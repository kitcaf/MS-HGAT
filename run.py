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
# action 参数 表示命令行中出现no_cuda参数时，-no_cuda 参数的值为True
parser.add_argument('-no_cuda', action='store_true')  # 是否不使用CUDA
parser.add_argument('-pos_emb', type=bool, default=True)  # 是否使用位置嵌入

opt = parser.parse_args() 
opt.d_word_vec = opt.d_model  # 词向量维度等于模型维度
#print(opt)


def get_performance(crit, pred, gold, prediction_steps=Constants.prediction_steps):
    """
    计算模型性能
    参数:
        crit: 损失函数
        pred: 预测值列表，包含prediction_steps个预测，每个形状为[batch_size, seq_len, n_node]
        gold: 真实值，形状为[batch_size, prediction_steps]
        prediction_steps: 预测步数
    返回:
        loss: 总损失值
        n_correct: 预测正确的样本数列表
    """
    batch_size = gold.size(0)
    
    # 扩展gold以包含多步预测的目标
    gold_expanded = []
    for step in range(prediction_steps):
        # 对于每一步，目标是序列中相应位置的元素
        gold_step = gold[:, step].contiguous()
        gold_expanded.append(gold_step)
    
    total_loss = 0
    n_correct_list = []
    
    # 计算每一步的损失和准确率
    for step in range(prediction_steps):
        step_pred = pred[step]  # [batch_size, seq_len, n_node]
        step_gold = gold_expanded[step]  # [batch_size]
        
        # 我们需要将预测调整为二维张量 [batch_size, n_node]
        # 只保留每个序列的最后一个非填充位置的预测
        step_pred = step_pred[:, -1, :]  # 取每个批次的最后一个位置
        
        # 计算损失
        step_loss = crit(step_pred, step_gold)
        total_loss += step_loss
        
        # 计算准确率
        step_pred = step_pred.max(1)[1]  # 获取预测的最大概率的索引
        n_correct = step_pred.data.eq(step_gold.data)  # 比较预测值和真实值
        n_correct = n_correct.masked_select(step_gold.ne(Constants.PAD).data).sum().float()  # 计算非填充位置的正确预测数
        n_correct_list.append(n_correct)
    
    return total_loss, n_correct_list


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
        n_total_correct/n_total_words: 准确率列表
    """
    # 设置模型为训练模式
    model.train()

    total_loss = 0.0
    n_total_words = 0.0
    n_total_correct = [0.0] * Constants.prediction_steps
    batch_num = 0.0

    print("开始训练批次...")
    for i, batch in enumerate(training_data): # tqdm(training_data, mininterval=2, desc='  - (Training)   ', leave=False):
        if i % 10 == 0:
            print(f"处理批次 {i}...")
            
        # 准备数据
        tgt, tgt_timestamp, tgt_idx = (item.cuda() for item in batch)
        
        np.set_printoptions(threshold=np.inf)
        # 修改：获取多步预测的目标
        gold = tgt[:, 1:1+Constants.prediction_steps]  # 真实值为目标序列的后续元素

        # 计算非填充词的数量（第一步的目标）
        n_words = gold[:, 0].data.ne(Constants.PAD).sum().float()
        n_total_words += n_words
        batch_num += tgt.size(0)
        
        # 训练
        optimizer.zero_grad()  # 清空梯度
        # 调用模型的forward方法
        try:
            print(f"批次 {i}: 调用模型前向传播...")
            pred = model(tgt, tgt_timestamp, tgt_idx, graph, hypergraph_list)  # 前向传播
            print(f"批次 {i}: 前向传播完成，计算损失...")
            
            # 检查预测输出的形状
            for step, p in enumerate(pred):
                print(f"批次 {i}, 步骤 {step}: 预测形状 = {p.shape}")
            
            # 计算损失和准确率
            loss, n_correct_list = get_performance(loss_func, pred, gold)
            print(f"批次 {i}: 损失计算完成，开始反向传播...")
            loss.backward()  # 反向传播
            print(f"批次 {i}: 反向传播完成，更新参数...")

            # 更新参数
            optimizer.step()
            optimizer.update_learning_rate()  # 更新学习率
            print(f"批次 {i}: 参数更新完成")

            # 累加每一步的正确预测数
            for step in range(Constants.prediction_steps):
                n_total_correct[step] += n_correct_list[step]
            total_loss += loss.item()
        except Exception as e:
            print(f"批次 {i} 处理出错: {str(e)}")
            import traceback
            traceback.print_exc()
            continue

    # 计算平均损失和每一步的准确率
    avg_loss = total_loss / n_total_words
    avg_accuracy = [n_correct / n_total_words for n_correct in n_total_correct]
    
    return avg_loss, avg_accuracy

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
    
    # 构建联系关系图和传播超图
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
        
        # 修改打印格式以显示每一步的准确率
        print('  - (Training)   loss: {loss: 8.5f}, elapse: {elapse:3.3f} min'.format(
            loss=train_loss, elapse=(time.time() - start) / 60))
        
        # 打印每一步的准确率
        for step in range(Constants.prediction_steps):
            print('    - Step {}: accuracy: {:3.3f} %'.format(
                step + 1, 100 * train_accu[step]))

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
            if validation_history <= sum(scores['avg'].values()):
                print("Best Validation hit@100:{} at Epoch:{}".format(scores['avg']["hits@100"], epoch_i))
                validation_history = sum(scores['avg'].values())
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
    print("开始评估...")

    # 为每一步预测创建评估指标
    scores = {}
    for step in range(Constants.prediction_steps):
        step_scores = {}
        for k in k_list:
            step_scores['hits@' + str(k)] = 0
            step_scores['map@' + str(k)] = 0
        scores[f'step_{step+1}'] = step_scores
    
    # 添加平均指标
    avg_scores = {}
    for k in k_list:
        avg_scores['hits@' + str(k)] = 0
        avg_scores['map@' + str(k)] = 0
    scores['avg'] = avg_scores

    n_total_words = 0
    with torch.no_grad():  # 不计算梯度
        for i, batch in enumerate(validation_data):  #tqdm(validation_data, mininterval=2, desc='  - (Validation) ', leave=False):
            if i % 10 == 0:
                print(f"评估批次 {i}...")
                
            try:
                # 准备数据
                tgt, tgt_timestamp, tgt_idx = batch
                # 修改：获取多步预测的目标
                y_gold_all = tgt[:, 1:1+Constants.prediction_steps].contiguous()
                
                # 前向传播
                print(f"评估批次 {i}: 调用模型前向传播...")
                pred_list = model(tgt, tgt_timestamp, tgt_idx, graph, hypergraph_list)
                print(f"评估批次 {i}: 前向传播完成")
                
                # 检查预测输出的形状
                for step, p in enumerate(pred_list):
                    print(f"评估批次 {i}, 步骤 {step}: 预测形状 = {p.shape}")
                
                # 计算每一步的指标
                for step in range(Constants.prediction_steps):
                    # 获取当前步的真实值
                    y_gold = y_gold_all[:, step].contiguous().detach().cpu().numpy()  # 当前步的真实值
                    
                    # 处理预测值，取每个批次的最后一个位置
                    y_pred = pred_list[step][:, -1, :].detach().cpu().numpy()  # 当前步的预测值
                    
                    print(f"评估批次 {i}, 步骤 {step}: 计算指标...")
                    # 计算当前步的指标
                    scores_batch, scores_len = metric.compute_metric(y_pred, y_gold, k_list)
                    print(f"评估批次 {i}, 步骤 {step}: 指标计算完成")
                    
                    if step == 0:  # 只在第一步计算总词数
                        n_total_words += scores_len
                    
                    # 累加指标
                    for k in k_list:
                        scores[f'step_{step+1}']['hits@' + str(k)] += scores_batch['hits@' + str(k)] * scores_len
                        scores[f'step_{step+1}']['map@' + str(k)] += scores_batch['map@' + str(k)] * scores_len
                        # 同时累加到平均指标
                        scores['avg']['hits@' + str(k)] += scores_batch['hits@' + str(k)] * scores_len
                        scores['avg']['map@' + str(k)] += scores_batch['map@' + str(k)] * scores_len
            except Exception as e:
                print(f"评估批次 {i} 处理出错: {str(e)}")
                import traceback
                traceback.print_exc()
                continue

    print("计算最终评估指标...")
    # 计算每一步的平均值
    for step in range(Constants.prediction_steps):
        for k in k_list:
            scores[f'step_{step+1}']['hits@' + str(k)] = scores[f'step_{step+1}']['hits@' + str(k)] / n_total_words
            scores[f'step_{step+1}']['map@' + str(k)] = scores[f'step_{step+1}']['map@' + str(k)] / n_total_words
    
    # 计算总平均值
    for k in k_list:
        scores['avg']['hits@' + str(k)] = scores['avg']['hits@' + str(k)] / (n_total_words * Constants.prediction_steps)
        scores['avg']['map@' + str(k)] = scores['avg']['map@' + str(k)] / (n_total_words * Constants.prediction_steps)

    print("评估完成")
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



