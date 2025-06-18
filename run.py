# -*- coding: utf-8 -*-
"""
Created on Mon Jan 18 22:42:32 2021

@author: Ling Sun
"""

# 导入必要的库
import argparse
import time
import numpy as np 
import Constants
import torch
import torch.nn as nn
from graphConstruct import ConRelationGraph, ConHyperGraphList
from dataLoader import Split_data, DataLoader
from Metrics import Metrics
from HGAT import MSHGAT
from Optim import ScheduledOptim


# 设置随机种子，确保实验可重复性
torch.backends.cudnn.deterministic = True
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
np.random.seed(0)
torch.cuda.manual_seed(0)

# 初始化评估指标
metric = Metrics()


# 命令行参数设置
parser = argparse.ArgumentParser()
parser.add_argument('-data_name', default='twitter')  # 数据集名称
parser.add_argument('-epoch', type=int, default=50)   # 训练轮数
parser.add_argument('-batch_size', type=int, default=64)  # 批次大小
parser.add_argument('-d_model', type=int, default=64)  # 模型维度
parser.add_argument('-initialFeatureSize', type=int, default=64)  # 初始特征大小
parser.add_argument('-train_rate', type=float, default=0.8)  # 训练集比例
parser.add_argument('-valid_rate', type=float, default=0.1)  # 验证集比例
parser.add_argument('-n_warmup_steps', type=int, default=1000)  # 预热步数
parser.add_argument('-dropout', type=float, default=0.3)  # dropout比率
parser.add_argument('-log', default=None)  # 日志路径
parser.add_argument('-save_path', default= "./checkpoint/DiffusionPrediction.pt")  # 模型保存路径
parser.add_argument('-save_mode', type=str, choices=['all', 'best'], default='best')  # 保存模式
parser.add_argument('-no_cuda', action='store_true')  # 是否不使用CUDA
parser.add_argument('-pos_emb', type=bool, default=True)  # 是否使用位置嵌入

opt = parser.parse_args() 
opt.d_word_vec = opt.d_model  # 词向量维度等于模型维度
#print(opt)


def get_performance(crit, pred, gold):
    """
    计算模型性能
    参数:
        crit: 损失函数
        pred: 预测值
        gold: 真实值
    返回:
        loss: 损失值
        n_correct: 预测正确的样本数
    """

    loss = crit(pred, gold.contiguous().view(-1))  # 计算损失
    pred = pred.max(1)[1]  # 获取预测的最大概率的索引
    gold = gold.contiguous().view(-1)  # 展平真实值
    n_correct = pred.data.eq(gold.data)  # 比较预测值和真实值
    n_correct = n_correct.masked_select(gold.ne(Constants.PAD).data).sum().float()  # 计算非填充位置的正确预测数
    return loss, n_correct


def train_epoch(model, training_data, graph, hypergraph_list, loss_func, optimizer):
    """
    训练一个轮次
    参数:
        model: 模型
        training_data: 训练数据
        graph: 关系图
        hypergraph_list: 超图列表
        loss_func: 损失函数
        optimizer: 优化器
    返回:
        total_loss/n_total_words: 平均损失
        n_total_correct/n_total_words: 准确率
    """
    # 设置模型为训练模式
    model.train()

    total_loss = 0.0
    n_total_words = 0.0
    n_total_correct = 0.0
    batch_num = 0.0

    for i, batch in enumerate(training_data): # tqdm(training_data, mininterval=2, desc='  - (Training)   ', leave=False):
        # 准备数据
        tgt, tgt_timestamp, tgt_idx = (item.cuda() for item in batch)
        
        np.set_printoptions(threshold=np.inf)
        gold = tgt[:, 1:]  # 真实值为目标序列的下一个元素

        n_words = gold.data.ne(Constants.PAD).sum().float()  # 非填充词的数量
        n_total_words += n_words
        batch_num += tgt.size(0)
        
        # 训练
        optimizer.zero_grad()  # 清空梯度
        pred = model(tgt, tgt_timestamp,tgt_idx, graph, hypergraph_list)  # 前向传播
        
        # 计算损失
        loss, n_correct = get_performance(loss_func, pred, gold)
        loss.backward()  # 反向传播

        # 更新参数
        optimizer.step()
        optimizer.update_learning_rate()  # 更新学习率

        n_total_correct += n_correct
        total_loss += loss.item()

    return total_loss/n_total_words, n_total_correct/n_total_words

def train_model(MSHGAT, data_path):
    """
    训练模型
    参数:
        MSHGAT: 模型类
        data_path: 数据路径
    """
    # ========= 准备数据加载器 =========#
    user_size, total_cascades, timestamps, train, valid, test = Split_data(data_path, opt.train_rate, opt.valid_rate, load_dict=True)
    
    train_data = DataLoader(train, batch_size=opt.batch_size, load_dict=True, cuda=False)
    valid_data = DataLoader(valid, batch_size=opt.batch_size, load_dict=True, cuda=False)
    test_data = DataLoader(test, batch_size=opt.batch_size, load_dict=True, cuda=False)
    
    # 构建关系图和超图
    relation_graph = ConRelationGraph(data_path)
    hypergraph_list = ConHyperGraphList(total_cascades, timestamps, user_size)

    opt.user_size = user_size

    # ========= 准备模型 =========#
    model = MSHGAT(opt, dropout = opt.dropout)
    loss_func = nn.CrossEntropyLoss(size_average=False, ignore_index=Constants.PAD)
    
    # 设置优化器
    params = model.parameters()
    optimizerAdam = torch.optim.Adam(params, betas=(0.9, 0.98), eps=1e-09)
    optimizer = ScheduledOptim(optimizerAdam, opt.d_model, opt.n_warmup_steps)

    # 如果有CUDA，将模型和损失函数移至GPU
    if torch.cuda.is_available():
        model = model.cuda()
        loss_func = loss_func.cuda()

    validation_history = 0.0
    best_scores = {}
    # 开始训练循环
    for epoch_i in range(opt.epoch):
        print('\n[ Epoch', epoch_i, ']')

        start = time.time()
        train_loss, train_accu = train_epoch(model, train_data, relation_graph, hypergraph_list, loss_func, optimizer)
        print('  - (Training)   loss: {loss: 8.5f}, accuracy: {accu:3.3f} %, ' \
              'elapse: {elapse:3.3f} min'.format(
            loss=train_loss, accu=100 * train_accu,
            elapse=(time.time() - start) / 60))

        # 验证和测试
        if epoch_i >= 0: 
            start = time.time()
            scores = test_epoch(model, valid_data, relation_graph, hypergraph_list)
            print('  - ( Validation )) ')
            for metric in scores.keys():
                print(metric + ' ' + str(scores[metric]))
            print("Validation use time: ", (time.time() - start) / 60, "min")

            print('  - (Test) ')
            scores = test_epoch(model, test_data, relation_graph, hypergraph_list)
            for metric in scores.keys():
                print(metric + ' ' + str(scores[metric]))

            # 保存最佳模型
            if validation_history <= sum(scores.values()):
                print("Best Validation hit@100:{} at Epoch:{}".format(scores["hits@100"], epoch_i))
                validation_history = sum(scores.values())
                best_scores = scores
                print("Save best model!!!")
                torch.save(model.state_dict(), opt.save_path)
                
    # 打印最佳分数
    print(" -(Finished!!) \n Best scores: ")        
    for metric in best_scores.keys():
        print(metric + ' ' + str(best_scores[metric]))
                
def test_epoch(model, validation_data, graph, hypergraph_list, k_list=[10, 50, 100]):
    ''' 
    评估阶段的轮次操作
    参数:
        model: 模型
        validation_data: 验证数据
        graph: 关系图
        hypergraph_list: 超图列表
        k_list: 计算指标的k值列表
    返回:
        scores: 评估指标字典
    '''
    model.eval()  # 设置为评估模式

    scores = {}
    for k in k_list:
        scores['hits@' + str(k)] = 0
        scores['map@' + str(k)] = 0

    n_total_words = 0
    with torch.no_grad():  # 不计算梯度
        for i, batch in enumerate(validation_data):  #tqdm(validation_data, mininterval=2, desc='  - (Validation) ', leave=False):
            #print("Validation batch ", i)
            # 准备数据
            tgt, tgt_timestamp, tgt_idx =  batch
            y_gold = tgt[:, 1:].contiguous().view(-1).detach().cpu().numpy()  # 真实值

            # 前向传播
            pred = model(tgt, tgt_timestamp, tgt_idx, graph, hypergraph_list )
            y_pred = pred.detach().cpu().numpy()  # 预测值

            # 计算指标
            scores_batch, scores_len = metric.compute_metric(y_pred, y_gold, k_list)
            n_total_words += scores_len
            for k in k_list:
                scores['hits@' + str(k)] += scores_batch['hits@' + str(k)] * scores_len
                scores['map@' + str(k)] += scores_batch['map@' + str(k)] * scores_len

    # 计算平均值
    for k in k_list:
        scores['hits@' + str(k)] = scores['hits@' + str(k)] / n_total_words
        scores['map@' + str(k)] = scores['map@' + str(k)] / n_total_words

    return scores


def test_model(MSHGAT, data_path):
    """
    测试模型
    参数:
        MSHGAT: 模型类
        data_path: 数据路径
    """
    
    user_size, total_cascades, timestamps, train, valid, test = Split_data(data_path, opt.train_rate, opt.valid_rate, load_dict=True)
    
    test_data = DataLoader(test, batch_size=opt.batch_size, load_dict=True, cuda=False)
    
    relation_graph = ConRelationGraph(data_path)
    hypergraph_list = ConHyperGraphList(train, user_size)

    opt.user_size = user_size

    # 加载模型
    model = MSHGAT(opt, dropout = opt.dropout)
    model.load_state_dict(torch.load(opt.save_path))
    model.cuda()

    # 评估模型
    scores = test_epoch(model, test_data, relation_graph, hypergraph_list)
    print('  - (Test) ')
    for metric in scores.keys():
        print(metric + ' ' + str(scores[metric]))


if __name__ == "__main__": 
    model = MSHGAT  
    train_model(model, opt.data_name)



