import numpy as np
import torch
import pickle
import Constants
import os 
from torch_geometric.data import Data
from dataLoader import Options
import scipy.sparse as sp
import torch.nn.functional as F

'''Friendship网络构建'''       
def ConRelationGraph(data):
        """
        构建用户关系图（友谊网络）
        参数:
            data: 数据集名称
        返回:
            data: PyG格式的图数据对象
        """
        options = Options(data)
        _u2idx = {}  # 用户到索引的映射字典
    
        with open(options.u2idx_dict, 'rb') as handle:
            _u2idx = pickle.load(handle)
        
        #TODO 
        edges_list = []  # 边列表
        if os.path.exists(options.net_data):
            with open(options.net_data, 'r') as handle:
                relation_list = handle.read().strip().split("\n")
                relation_list = [edge.split(',') for edge in relation_list]

                # 将用户名转换为索引，并过滤不在索引中的用户
                relation_list = [(_u2idx[edge[0]], _u2idx[edge[1]]) for edge in relation_list if edge[0] in _u2idx and edge[1] in _u2idx]

                # 添加反向边，构建无向图 relation_list = [[1, 2], [3, 4], [5, 6] ...]
                relation_list_reverse = [edge[::-1] for edge in relation_list]

                edges_list += relation_list_reverse
        else:
            return []  # 如果边文件不存在，返回空列表
        # 转换为PyTorch张量格式
        edges_list_tensor = torch.LongTensor(edges_list).t()
        edges_weight = torch.ones(edges_list_tensor.size(1)).float() # 权重这里其实是可以自动计算的
        """

        
        """
        data = Data(edge_index=edges_list_tensor, edge_attr=edges_weight)
        
        return data

'''扩散超图构建'''
def ConHyperGraphList(cascades, timestamps, user_size, step_split=Constants.step_split):
    #TODO
    '''
    将超图分割为子图，返回图列表
    参数:
        cascades: 级联序列
        timestamps: 时间戳
        user_size: 用户数量
        step_split: 分割步数
    返回:
        graphs: 包含子图和根节点列表的列表
    '''

    print(f"扩散超图构建的级联总数{len(cascades)}")
    times, root_list = ConHyperDiffsuionGraph(cascades, timestamps, user_size)
    zero_vec = torch.zeros_like(times)  # 零向量，与times形状相同
    one_vec = torch.ones_like(times)    # 一向量，与times形状相同
    
    time_sorted = []  # 排序后的时间戳
    graph_list = {}   # 子图字典，键为时间戳，值为子图
    
    # 收集并排序所有时间戳
    for time in timestamps:
        time_sorted += time[:-1]
    time_sorted = sorted(time_sorted)
    # 计算每个分割的长度
    split_length = len(time_sorted) // step_split
    
    # 根据时间戳分割超图
    for x in range(split_length, split_length * step_split , split_length):
        if x == split_length:
            # 第一个子图：从开始到第一个分割点
            sub_graph = torch.where(times > 0, one_vec, zero_vec) - torch.where(times > time_sorted[x], one_vec, zero_vec)
        else:
            # 其他子图：从上一个分割点到当前分割点
            sub_graph = torch.where(times > time_sorted[x-split_length], one_vec, zero_vec) - torch.where(times > time_sorted[x], one_vec, zero_vec)
          
        graph_list[time_sorted[x]] = sub_graph
    
    graphs = [graph_list, root_list]
    
    return graphs
    
    
def ConHyperDiffsuionGraph(cascades, timestamps, user_size):
    '''
    返回超图的邻接矩阵和时间邻接矩阵
    参数:
        cascades: 级联序列
        timestamps: 时间戳
        user_size: 用户数量
    返回:
        Times: 时间邻接矩阵
        root_list: 根节点列表
    '''
    e_size = len(cascades)+1  # 边数量
    n_size = user_size        # 节点数量
    rows = []                 # 行索引
    cols = []                 # 列索引
    vals_time = []            # 时间值
    root_list = [0]           # 根节点列表，初始化为0

        
    for i in range(e_size-1):
        root_list.append(cascades[i][0])  # 添加每个级联的根节点
        #TODO 后面改成3步预测时，这里就是-3
        rows += cascades[i][:-1]          # 添加行索引（用户）
        cols +=[i+1]*(len(cascades[i])-1) # 添加列索引（级联ID）
        #vals +=[1.0]*(len(cascades[i])-1)
        #TODO 后面改成3步预测时，同理也是-3
        vals_time += timestamps[i][:-1]    # 添加时间值
        
    root_list = torch.tensor(root_list)   # 转换为张量
    # 创建稀疏张量表示时间邻接矩阵
    Times = torch.sparse_coo_tensor(torch.Tensor([rows,cols]), torch.Tensor(vals_time), [n_size,e_size])
        
        
    return Times.to_dense(), root_list
 

def normalize(mx):
    """
    行归一化稀疏矩阵
    参数:
        mx: 输入矩阵
    返回:
        mx: 归一化后的矩阵
    """
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.  # 处理无穷大值
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    mx = torch.Tensor(mx).to_sparse()
    return mx


def get_NodeAttention(x, adjt, root_emb):
    """
    计算节点注意力
    参数:
        x: 节点特征
        adjt: 邻接矩阵的转置
        root_emb: 根节点嵌入
    返回:
        n2e_att: 节点到边的注意力权重
    """
    x1 = x[adjt.nonzero().t()[1]]   # 获取非零元素对应的节点特征
    #print(x1.shape)
    
    # 为每个根节点重复其嵌入向量
    q1 = torch.cat([root_emb[i].repeat(len(adjt[i].nonzero()),1) for i in torch.arange(root_emb.shape[0])], dim = 0)
    # 计算与根节点的相似度（欧氏距离）
    distance = torch.norm(q1.float()-x1.float(),dim = 1).cpu()
    n2e_att = torch.sparse_coo_tensor(adjt.nonzero().t(), distance, adjt.shape).to_dense() # e*n
        
    zero_vec = 9e15*torch.ones_like(n2e_att)  # 大值掩码
    n2e_att = torch.where(n2e_att > 0, n2e_att, zero_vec)
    n2e_att = F.softmax(-n2e_att, dim = 1) # e*n，负距离的softmax，将距离转换为相似度
    return n2e_att.cuda()

def get_EdgeAttention(adj):
    """
    获取边注意力
    参数:
        adj: 邻接矩阵
    返回:
        adj: 移至GPU的邻接矩阵
    """                
    return adj.cuda()
    