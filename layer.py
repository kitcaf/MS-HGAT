# -*- coding: utf-8 -*-
"""
Created on Mon Jan 18 22:29:40 2021

@author: Ling Sun
"""
# 导入必要的库
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from graphConstruct import  get_EdgeAttention, get_NodeAttention, normalize

class HGATLayer(nn.Module):
    """
    超图注意力层
    实现了基于注意力机制的超图卷积操作
    """

    def __init__(self, in_features, out_features, dropout, transfer, concat=True, bias=False, edge = True):
        """
        初始化超图注意力层
        参数:
            in_features: 输入特征维度
            out_features: 输出特征维度
            dropout: dropout比率
            transfer: 是否使用权重转换
            concat: 是否在输出时使用拼接，默认True
            bias: 是否使用偏置，默认False
            edge: 是否返回边嵌入，默认True
        """
        super(HGATLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.concat = concat
        self.edge = edge

        self.transfer = transfer

        if self.transfer:
            # 如果使用转换，创建权重矩阵
            self.weight = Parameter(torch.Tensor(self.in_features, self.out_features))
        else:
            self.register_parameter('weight', None)

        # 创建两个权重矩阵用于特征转换
        self.weight2 = Parameter(torch.Tensor(self.in_features, self.out_features))
        self.weight3 = Parameter(torch.Tensor(self.out_features, self.out_features))

        if bias:
            self.bias = Parameter(torch.Tensor(self.out_features))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        """重置参数"""
        stdv = 1. / math.sqrt(self.out_features)
        if self.weight is not None:
            self.weight.data.uniform_(-stdv, stdv)
        self.weight2.data.uniform_(-stdv, stdv)
        self.weight3.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, x, adj, root_emb):
        """
        前向传播
        参数:
            x: 节点特征矩阵
            adj: 邻接矩阵
            root_emb: 根节点嵌入
        返回:
            node: 节点嵌入
            edge: 边嵌入（如果self.edge为True）
        """
        
        if self.transfer:
            # 使用第一个权重矩阵转换特征
            x = x.matmul(self.weight)
        else:
            # 使用第二个权重矩阵转换特征
            x = x.matmul(self.weight2)
        
        if self.bias is not None:
            # 添加偏置
            x = x + self.bias  
            
        #n2e_att = get_NodeAttention(x, adj.t(), root_emb)

        # 对邻接矩阵的转置进行softmax，得到节点到边的权重
        adjt = F.softmax(adj.T,dim = 1)
        #adj = normalize(adj)
        

        # 节点特征聚合到边
        edge = torch.matmul(adjt, x)
        
        # dropout和激活
        edge = F.dropout(edge, self.dropout, training=self.training)
        edge = F.relu(edge,inplace = False)

        # 使用第三个权重矩阵转换边特征
        e1 = edge.matmul(self.weight3)

        
        # 对邻接矩阵进行softmax，得到边到节点的权重
        adj = F.softmax(adj,dim = 1)
        #adj = get_EdgeAttention(adj)

        # 边特征聚合到节点
        node = torch.matmul(adj, e1)
        node = F.dropout(node, self.dropout, training=self.training)
        

        if self.concat:
            # 如果使用拼接，则应用ReLU激活
            node = F.relu(node,inplace = False)
            
        if self.edge:
            # 如果需要返回边嵌入，再次从节点聚合到边
            edge = torch.matmul(adjt, node)        
            edge = F.dropout(edge, self.dropout, training=self.training)
            edge = F.relu(edge,inplace = False) 
            return node, edge
        else:
            return node

    def __repr__(self):
        """返回层的字符串表示"""
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'
