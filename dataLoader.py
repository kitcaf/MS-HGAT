# Part of this file is derived from 
# https://github.com/albertyang33/FOREST
"""
Created on Mon Jan 18 22:28:02 2021

@author: Ling Sun
"""
# 导入必要的库
import random
import numpy as np
import torch
from torch.autograd import Variable
import Constants
import pickle


class Options(object):
    """配置类：用于存储数据路径和嵌入维度等参数"""
    
    def __init__(self, data_name = 'douban'):
        """
        初始化配置对象
        参数:
            data_name: 数据集名称，默认为'douban'
        """
        self.data = 'data/'+data_name+'/cascades.txt'  # 级联数据文件路径
        self.u2idx_dict = 'data/'+data_name+'/u2idx.pickle'  # 用户到索引映射文件
        self.idx2u_dict = 'data/'+data_name+'/idx2u.pickle'  # 索引到用户映射文件
        self.save_path = ''  # 保存路径
        self.net_data = 'data/'+data_name+'/edges.txt'  # 网络边数据文件路径
        self.embed_dim = 64  # 嵌入维度

def Split_data(data_name, train_rate =0.8, valid_rate = 0.1, random_seed = 300, load_dict=True, with_EOS=True):
        """
        数据分割函数：将数据集分为训练集、验证集和测试集
        参数:
            data_name: 数据集名称
            train_rate: 训练集比例，默认0.8
            valid_rate: 验证集比例，默认0.1
            random_seed: 随机种子，默认300
            load_dict: 是否加载已有的用户映射字典，默认True
            with_EOS: 是否在序列末尾添加结束符，默认True
        返回:
            user_size: 用户数量
            t_cascades: 所有级联序列
            timestamps: 所有时间戳
            train: 训练数据
            valid: 验证数据
            test: 测试数据
        """
        options = Options(data_name)
        u2idx = {}  # 用户到索引的映射字典
        idx2u = []  # 索引到用户的映射列表
        if not load_dict:
            # 如果不加载已有字典，则构建新的映射
            user_size, u2idx, idx2u = buildIndex(options.data)
            with open(options.u2idx_dict, 'wb') as handle:
                pickle.dump(u2idx, handle, protocol=pickle.HIGHEST_PROTOCOL)
            with open(options.idx2u_dict, 'wb') as handle:
                pickle.dump(idx2u, handle, protocol=pickle.HIGHEST_PROTOCOL)
        else:
            # 加载已有的用户映射字典
            with open(options.u2idx_dict, 'rb') as handle:
                u2idx = pickle.load(handle)
            with open(options.idx2u_dict, 'rb') as handle:
                idx2u = pickle.load(handle)
            user_size = len(u2idx)
            
        t_cascades = []  # 存储级联序列
        timestamps = []  # 存储时间戳
        for line in open(options.data):
            if len(line.strip()) == 0:
                continue
            timestamplist = []  # 当前级联的时间戳列表
            userlist = []  # 当前级联的用户列表
            chunks = line.strip().split(',')
            for chunk in chunks:
                try:
                    # 处理Twitter和Douban数据集格式
                    if len(chunk.split())==2:
                        user, timestamp = chunk.split()
                    # 处理Android和Christianity数据集格式
                    elif len(chunk.split())==3:
                        root, user, timestamp = chunk.split()                                           
                        if root in u2idx:          
                            userlist.append(u2idx[root])                        
                            timestamplist.append(float(timestamp))
                except:
                    print(chunk)
                if user in u2idx:
                    userlist.append(u2idx[user])
                    timestamplist.append(float(timestamp))

            # 只保留长度大于1且不超过500的级联
            if len(userlist) > 1 and len(userlist)<=500:
                if with_EOS:
                    # 添加结束符
                    userlist.append(Constants.EOS)
                    timestamplist.append(Constants.EOS)
                t_cascades.append(userlist)
                timestamps.append(timestamplist)
                
        
        '''按时间戳排序'''        
        order = [i[0] for i in sorted(enumerate(timestamps), key=lambda x:x[1])]
        timestamps = sorted(timestamps)
        t_cascades[:] = [t_cascades[i] for i in order]
        cas_idx = [i for i in range(len(t_cascades))]
        
        '''数据分割'''
        # 分割训练集
        train_idx_ = int(train_rate*len(t_cascades))
        train = t_cascades[0:train_idx_]
        train_t = timestamps[0:train_idx_]
        train_idx = cas_idx[0:train_idx_]
        train = [train, train_t, train_idx]
        
        # 分割验证集
        valid_idx_ = int((train_rate+valid_rate)*len(t_cascades))
        valid = t_cascades[train_idx_:valid_idx_]
        valid_t = timestamps[train_idx_:valid_idx_]
        valid_idx = cas_idx[train_idx_:valid_idx_]
        valid = [valid, valid_t, valid_idx]
        
        # 分割测试集
        test = t_cascades[valid_idx_:]
        test_t = timestamps[valid_idx_:]
        test_idx = cas_idx[valid_idx_:]
        test = [test, test_t, test_idx]
            
        # 打乱训练数据
        random.seed(random_seed)
        random.shuffle(train)
        random.seed(random_seed)
        random.shuffle(train_t)
        random.seed(random_seed)
        random.shuffle(train_idx)
        
        # 打印数据集统计信息
        total_len =  sum(len(i)-1 for i in t_cascades)
        train_size = len(train_t)
        valid_size = len(valid_t)
        test_size = len(test_t)
        print("training size:%d\n   valid size:%d\n  testing size:%d" % (train_size, valid_size, test_size))
        print("total size:%d " %(len(t_cascades)))
        print("average length:%f" % (total_len/len(t_cascades)))
        print('maximum length:%f' % (max(len(cas) for cas in t_cascades)))
        print('minimum length:%f' % (min(len(cas) for cas in t_cascades)))    
        print("user size:%d"%(user_size-2))           
        
        return user_size, t_cascades, timestamps, train, valid, test
    
def buildIndex(data):
        """
        构建用户索引映射
        参数:
            data: 数据文件路径
        返回:
            user_size: 用户数量
            u2idx: 用户到索引的映射字典
            idx2u: 索引到用户的映射列表
        """
        user_set = set()  # 用户集合
        u2idx = {}  # 用户到索引的映射
        idx2u = []  # 索引到用户的映射

        lineid=0
        for line in open(data):
            lineid+=1
            if len(line.strip()) == 0:
                continue
            chunks = line.strip().split(',')
            for chunk in chunks:
                try:
                    # 处理不同格式的数据
                    if len(chunk.split())==2:
                        user, timestamp = chunk.split()
                    elif len(chunk.split())==3:
                        root, user, timestamp = chunk.split()                                           
                        user_set.add(root)
                except:
                    print(line)
                    print(chunk)
                    print(lineid)
                user_set.add(user)
        # 添加特殊标记：空白和结束符
        pos = 0
        u2idx['<blank>'] = pos
        idx2u.append('<blank>')
        pos += 1
        u2idx['</s>'] = pos
        idx2u.append('</s>')
        pos += 1

        # 为每个用户分配索引
        for user in user_set:
            u2idx[user] = pos
            idx2u.append(user)
            pos += 1
        user_size = len(user_set) + 2
        print("user_size : %d" % (user_size))
        return user_size, u2idx, idx2u
        
class DataLoader(object):
    ''' 数据迭代器：用于批量加载数据 '''

    def __init__(
        self, cas, batch_size=64, load_dict=True, cuda=True,  test=False, with_EOS=True): 
        """
        初始化数据加载器
        参数:
            cas: 级联数据，包含序列、时间戳和索引
            batch_size: 批次大小，默认64
            load_dict: 是否加载字典，默认True
            cuda: 是否使用CUDA，默认True
            test: 是否为测试模式，默认False
            with_EOS: 是否包含结束符，默认True
        """
        self._batch_size = batch_size
        self.cas = cas[0]  # 级联序列
        self.time = cas[1]  # 时间戳
        self.idx = cas[2]  # 索引
        self.test = test  # 测试模式标志
        self.with_EOS = with_EOS  # 是否包含结束符        
        self.cuda = cuda  # 是否使用CUDA
        
        self._n_batch = int(np.ceil(len(self.cas) / self._batch_size))  # 批次数量
        self._iter_count = 0  # 迭代计数器

     

    def __iter__(self):
        """迭代器方法"""
        return self

    def __next__(self):
        """Python 3迭代器接口"""
        return self.next()

    def __len__(self):
        """返回批次数量"""
        return self._n_batch

    def next(self):
        ''' 获取下一批数据 '''

        def pad_to_longest(insts):
            ''' 将序列填充到最大长度 '''

            max_len = 200  # 最大序列长度
            
            # 填充或截断序列
            inst_data = np.array([
                inst + [Constants.PAD] * (max_len - len(inst)) if len(inst)<max_len else inst[:max_len]  
               for inst in insts])
                
            # 转换为PyTorch张量
            inst_data_tensor = Variable(
                torch.LongTensor(inst_data), volatile=self.test)

            # 如果使用CUDA，则将张量移至GPU
            if self.cuda:
                inst_data_tensor = inst_data_tensor.cuda()

            return inst_data_tensor

        if self._iter_count < self._n_batch:
            # 计算当前批次的索引
            batch_idx = self._iter_count
            self._iter_count += 1

            # 计算批次的起始和结束索引
            start_idx = batch_idx * self._batch_size
            end_idx = (batch_idx + 1) * self._batch_size

            # 获取当前批次的数据
            seq_insts = self.cas[start_idx:end_idx]
            seq_timestamp = self.time[start_idx:end_idx]
            seq_data = pad_to_longest(seq_insts)
            seq_data_timestamp = pad_to_longest(seq_timestamp)
            seq_idx = Variable(
                torch.LongTensor(self.idx[start_idx:end_idx]), volatile=self.test)
            
            return seq_data, seq_data_timestamp, seq_idx
        else:
            # 迭代结束，重置计数器并抛出StopIteration异常
            self._iter_count = 0
            raise StopIteration()
