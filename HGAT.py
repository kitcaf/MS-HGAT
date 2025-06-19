# -*- coding: utf-8 -*-
"""
Created on Mon Jan 18 22:30:16 2021

@author: Ling Sun
"""

# 导入必要的库
import math
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from layer import HGATLayer
from torch_geometric.nn import GCNConv
import torch.nn.init as init
import Constants
from TransformerBlock import TransformerBlock
from torch.autograd import Variable


def get_previous_user_mask(seq, user_size):
    ''' 
    为之前激活的用户创建掩码 - 用于防止模型预测已经出现过的用户
    这是针对信息传播场景的一个重要约束 - 一个用户一旦被激活，就不应该再次被激活。
    参数:
        seq: 用户序列，形状为[batch_size, seq_len]
        user_size: 用户总数（包含PAD和EOS标记，即user_size = 实际用户数 + 2）
    返回:
        masked_seq: 掩码张量，形状为[batch_size, seq_len, user_size]，对已出现用户位置赋予大的负值
    '''
    assert seq.dim() == 2
    prev_shape = (seq.size(0), seq.size(1), seq.size(1))
    seqs = seq.repeat(1, 1, seq.size(1)).view(seq.size(0), seq.size(1), seq.size(1))
    previous_mask = np.tril(np.ones(prev_shape)).astype('float32')  # 下三角矩阵，确保只看之前位置的用户
    previous_mask = torch.from_numpy(previous_mask)
    if seq.is_cuda:
        previous_mask = previous_mask.cuda()
    masked_seq = previous_mask * seqs.data.float()

    # 强制第0维度(PAD)被掩码
    PAD_tmp = torch.zeros(seq.size(0), seq.size(1), 1)
    if seq.is_cuda:
        PAD_tmp = PAD_tmp.cuda()
    masked_seq = torch.cat([masked_seq, PAD_tmp], dim=2)
    ans_tmp = torch.zeros(seq.size(0), seq.size(1), user_size)
    if seq.is_cuda:
        ans_tmp = ans_tmp.cuda()
    masked_seq = ans_tmp.scatter_(2, masked_seq.long(), float(-1000))  # 对已出现的用户添加大的负值
    masked_seq = Variable(masked_seq, requires_grad=False)
    # print("masked_seq ",masked_seq.size())
    return masked_seq.cuda()


# 融合门
class Fusion(nn.Module):
    def __init__(self, input_size, out=1, dropout=0.2):
        """
        初始化融合门模块
        参数:
            input_size: 输入特征维度
            out: 输出维度，默认为1
            dropout: dropout比率，默认0.2
        """
        super(Fusion, self).__init__()
        self.linear1 = nn.Linear(input_size, input_size)
        self.linear2 = nn.Linear(input_size, out)
        self.dropout = nn.Dropout(dropout)
        self.init_weights()

    def init_weights(self):
        """初始化权重"""
        init.xavier_normal_(self.linear1.weight)
        init.xavier_normal_(self.linear2.weight)

    def forward(self, hidden, dy_emb):
        """
        前向传播
        参数:
            hidden: 隐藏状态
            dy_emb: 动态嵌入
        返回:
            out: 融合后的嵌入
        """
        emb = torch.cat([hidden.unsqueeze(dim=0), dy_emb.unsqueeze(dim=0)], dim=0)
        emb_score = F.softmax(self.linear2(torch.tanh(self.linear1(emb))), dim=0)  # 计算注意力分数
        emb_score = self.dropout(emb_score)
        out = torch.sum(emb_score * emb, dim=0)  # 加权求和
        return out


'''学习友谊网络'''


class GraphNN(nn.Module):
    def __init__(self, ntoken, ninp, dropout=0.5, is_norm=True):
        """
        初始化图神经网络
        参数:
            ntoken: 节点数量
            ninp: 输入特征维度
            dropout: dropout比率，默认0.5
            is_norm: 是否使用批归一化，默认True
        """
        super(GraphNN, self).__init__()
        self.embedding = nn.Embedding(ntoken, ninp, padding_idx=0)
        # 输入:ninp, 输出:ninp*2
        self.gnn1 = GCNConv(ninp, ninp * 2)  # 第一层图卷积
        self.gnn2 = GCNConv(ninp * 2, ninp)  # 第二层图卷积
        self.is_norm = is_norm

        self.dropout = nn.Dropout(dropout)
        if self.is_norm:
            self.batch_norm = torch.nn.BatchNorm1d(ninp)
        self.init_weights()

    def init_weights(self):
        """初始化权重"""
        init.xavier_normal_(self.embedding.weight)

    def forward(self, graph):
        """
        前向传播
        参数:
            graph: 图数据
        返回:
            graph_output: 图节点的输出特征
        """
        graph_edge_index = graph.edge_index.cuda()
        graph_x_embeddings = self.gnn1(self.embedding.weight, graph_edge_index)  # 第一层GCN
        graph_x_embeddings = self.dropout(graph_x_embeddings)
        graph_output = self.gnn2(graph_x_embeddings, graph_edge_index)  # 第二层GCN
        if self.is_norm:
            graph_output = self.batch_norm(graph_output)  # 批归一化
        # print(graph_output.shape)
        return graph_output.cuda()


'''学习扩散网络'''


class HGNN_ATT(nn.Module):
    def __init__(self, input_size, n_hid, output_size, dropout=0.3, is_norm=True):
        """
        初始化超图注意力网络
        参数:
            input_size: 输入特征维度
            n_hid: 隐藏层维度
            output_size: 输出特征维度
            dropout: dropout比率，默认0.3
            is_norm: 是否使用批归一化，默认True
        """
        super(HGNN_ATT, self).__init__()
        self.dropout = dropout
        self.is_norm = is_norm
        if self.is_norm:
            self.batch_norm1 = torch.nn.BatchNorm1d(output_size)
        self.gat1 = HGATLayer(input_size, output_size, dropout=self.dropout, transfer=False, concat=True, edge=True)
        self.fus1 = Fusion(output_size)

    def forward(self, x, hypergraph_list):
        """
        前向传播
        参数:
            x: 节点特征
            hypergraph_list: 超图列表
        返回:
            embedding_list: 嵌入列表
        """
        root_emb = F.embedding(hypergraph_list[1].cuda(), x)  # 获取根节点嵌入

        hypergraph_list = hypergraph_list[0]
        embedding_list = {}
        for sub_key in hypergraph_list.keys():
            sub_graph = hypergraph_list[sub_key]
            # 使用超图注意力层处理子图
            sub_node_embed, sub_edge_embed = self.gat1(x, sub_graph.cuda(), root_emb)
            sub_node_embed = F.dropout(sub_node_embed, self.dropout, training=self.training)

            if self.is_norm:
                # 批归一化
                sub_node_embed = self.batch_norm1(sub_node_embed)
                sub_edge_embed = self.batch_norm1(sub_edge_embed)

            # 融合原始特征和节点嵌入
            x = self.fus1(x, sub_node_embed)
            embedding_list[sub_key] = [x.cpu(), sub_edge_embed.cpu()]

        return embedding_list


class MSHGAT(nn.Module):
    def __init__(self, opt, dropout=0.3):
        """
        初始化多尺度超图注意力网络
        参数:
            opt: 配置选项
            dropout: dropout比率，默认0.3
        """
        super(MSHGAT, self).__init__()
        self.hidden_size = opt.d_word_vec
        self.n_node = opt.user_size  # 节点数量，等于user_size（用户数+2），包含PAD和EOS标记
        self.pos_dim = 8  # 位置嵌入维度
        self.dropout = nn.Dropout(dropout)
        self.initial_feature = opt.initialFeatureSize
        self.step_shift = Constants.STEP_SHIFT  # 预测步长

        # 超图注意力网络
        self.hgnn = HGNN_ATT(self.initial_feature, self.hidden_size * 2, self.hidden_size, dropout=dropout)
        # 图神经网络
        self.gnn = GraphNN(self.n_node, self.initial_feature, dropout=dropout)
        # 融合模块
        self.fus = Fusion(self.hidden_size + self.pos_dim)
        self.fus2 = Fusion(self.hidden_size)
        # 位置嵌入
        self.pos_embedding = nn.Embedding(1000, self.pos_dim)
        # Transformer块
        self.decoder_attention1 = TransformerBlock(input_size=self.hidden_size + self.pos_dim, n_heads=8)
        self.decoder_attention2 = TransformerBlock(input_size=self.hidden_size + self.pos_dim, n_heads=8)

        # 添加3步预测解码器组件
        # 解码器Transformer块
        self.decoder_step1 = TransformerBlock(input_size=self.hidden_size + self.pos_dim, n_heads=8)
        self.decoder_step2 = TransformerBlock(input_size=self.hidden_size + self.pos_dim, n_heads=8)
        self.decoder_step3 = TransformerBlock(input_size=self.hidden_size + self.pos_dim, n_heads=8)
        
        # 步长预测的线性层
        self.step_linear1 = nn.Linear(self.hidden_size + self.pos_dim, self.hidden_size + self.pos_dim)
        self.step_linear2 = nn.Linear(self.hidden_size + self.pos_dim, self.hidden_size + self.pos_dim)
        self.step_linear3 = nn.Linear(self.hidden_size + self.pos_dim, self.hidden_size + self.pos_dim)

        # 输出层 - 将隐藏状态映射到用户预测概率，输出维度为user_size
        self.linear2 = nn.Linear(self.hidden_size + self.pos_dim, self.n_node)
        # 嵌入层
        self.embedding = nn.Embedding(self.n_node, self.initial_feature, padding_idx=0)
        self.reset_parameters()

    def reset_parameters(self):
        """重置参数"""
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def forward(self, input, input_timestamp, input_idx, graph, hypergraph_list):
        """
        前向传播
        参数:
            input: 输入序列，形状为[batch_size, seq_len]
            input_timestamp: 输入时间戳，形状为[batch_size, seq_len]
            input_idx: 输入索引，级联的索引
            graph: 关系图
            hypergraph_list: 超图列表
        返回:
            output: 用户预测概率，形状为[batch_size*seq_len, user_size]
        """

        # 对于3步预测，我们需要截取输入序列，去掉最后Constants.STEP_SHIFT+1个元素
        # 因为我们要预测每个位置后3步的用户，所以输入序列需要比原来短3+1步
        # 例如，如果原序列是[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]，输入序列应该是[1, 2, 3, 4, 5, 6, 7]
        input = input[:, :-Constants.STEP_SHIFT]  # 移除最后STEP_SHIFT+1个元素
        print(f"input输入结果{input.shape}")
        # print(input)
        # print(input_timestamp)
        input_timestamp = input_timestamp[:, :-Constants.STEP_SHIFT]  # 相应地也移除时间戳的最后STEP_SHIFT+1个元素
        hidden = self.dropout(self.gnn(graph))  # 使用GNN处理关系图，得到节点嵌入，形状为[n_node, hidden_size]
        memory_emb_list = self.hgnn(hidden, hypergraph_list)  # 使用HGNN处理超图，得到不同时间步的嵌入
        # print(sorted(memory_emb_list.keys()))

        mask = (input == Constants.PAD)  # 创建填充掩码，标记哪些位置是PAD
        batch_t = torch.arange(input.size(1)).expand(input.size()).cuda()  # 位置索引，形状为[batch_size, seq_len-1]
        order_embed = self.dropout(self.pos_embedding(batch_t))  # 位置嵌入，形状为[batch_size, seq_len-1, pos_dim]
        batch_size, max_len = input.size()  # 获取批次大小和序列长度

        zero_vec = torch.zeros_like(input)  # 零向量，用于时间戳处理
        dyemb = torch.zeros(batch_size, max_len, self.hidden_size).cuda()  # 动态嵌入，形状为[batch_size, seq_len-1, hidden_size]
        cas_emb = torch.zeros(batch_size, max_len, self.hidden_size).cuda()  # 级联嵌入，形状为[batch_size, seq_len-1, hidden_size]

        # 根据时间戳处理嵌入
        for ind, time in enumerate(sorted(memory_emb_list.keys())):
            if ind == 0:
                # 第一个时间段
                sub_input = torch.where(input_timestamp <= time, input, zero_vec)
                sub_emb = F.embedding(sub_input.cuda(), hidden.cuda())
                temp = sub_input == 0
                sub_cas = sub_emb.clone()
            else:
                # 后续时间段
                cur = torch.where(input_timestamp <= time, input, zero_vec) - sub_input
                temp = cur == 0

                sub_cas = torch.zeros_like(cur)
                sub_cas[~temp] = 1
                sub_cas = torch.einsum('ij,i->ij', sub_cas, input_idx)  # 张量乘积
                sub_cas = F.embedding(sub_cas.cuda(), list(memory_emb_list.values())[ind - 1][1].cuda())
                sub_emb = F.embedding(cur.cuda(), list(memory_emb_list.values())[ind - 1][0].cuda())
                sub_input = cur + sub_input

            sub_cas[temp] = 0
            sub_emb[temp] = 0
            dyemb += sub_emb  # 累加动态嵌入
            cas_emb += sub_cas  # 累加级联嵌入

            if ind == len(memory_emb_list) - 1:
                # 最后一个时间段
                sub_input = input - sub_input
                temp = sub_input == 0

                sub_cas = torch.zeros_like(sub_input)
                sub_cas[~temp] = 1
                sub_cas = torch.einsum('ij,i->ij', sub_cas, input_idx)
                sub_cas = F.embedding(sub_cas.cuda(), list(memory_emb_list.values())[ind - 1][1].cuda())
                sub_cas[temp] = 0
                sub_emb = F.embedding(sub_input.cuda(), list(memory_emb_list.values())[ind][0].cuda())
                sub_emb[temp] = 0

                dyemb += sub_emb
                cas_emb += sub_cas
        # dyemb = self.fus2(dyemb,cas_emb)

        # 拼接动态嵌入和位置嵌入，形状为[batch_size, seq_len-1, hidden_size+pos_dim]
        diff_embed = torch.cat([dyemb, order_embed], dim=-1).cuda()
        # 拼接关系图嵌入和位置嵌入，形状为[batch_size, seq_len-1, hidden_size+pos_dim]
        fri_embed = torch.cat([F.embedding(input.cuda(), hidden.cuda()), order_embed], dim=-1).cuda()

        # 使用Transformer块处理扩散嵌入
        diff_att_out = self.decoder_attention1(diff_embed.cuda(), diff_embed.cuda(), diff_embed.cuda(),
                                               mask=mask.cuda())
        diff_att_out = self.dropout(diff_att_out.cuda())

        # 使用Transformer块处理友谊嵌入
        fri_att_out = self.decoder_attention2(fri_embed.cuda(), fri_embed.cuda(), fri_embed.cuda(), mask=mask.cuda())
        fri_att_out = self.dropout(fri_att_out.cuda())

        # 融合两种注意力输出，形状为[batch_size, seq_len-1, hidden_size+pos_dim]
        att_out = self.fus(diff_att_out, fri_att_out)

        # ===== 添加3步预测解码器 =====
        # 创建用于存储3步预测结果的张量
        step1_features = torch.zeros_like(att_out)
        step2_features = torch.zeros_like(att_out)
        step3_features = torch.zeros_like(att_out)
        
        # 第1步预测：使用当前特征预测下一步
        step1_hidden = self.step_linear1(att_out)
        step1_features = self.decoder_step1(step1_hidden, step1_hidden, step1_hidden, mask=mask.cuda())
        step1_features = self.dropout(step1_features)
        
        # 第2步预测：使用第1步预测结果预测下一步
        step2_hidden = self.step_linear2(step1_features)
        step2_features = self.decoder_step2(step2_hidden, step2_hidden, step2_hidden, mask=mask.cuda())
        step2_features = self.dropout(step2_features)
        
        # 第3步预测：使用第2步预测结果预测下一步
        step3_hidden = self.step_linear3(step2_features)
        step3_features = self.decoder_step3(step3_hidden, step3_hidden, step3_hidden, mask=mask.cuda())
        step3_features = self.dropout(step3_features)
        
        # 将最终的第3步特征映射到用户预测空间
        output_u = self.linear2(step3_features)  # 形状为[batch_size, seq_len-1, user_size]
        
        # 获取之前用户的掩码 - 防止预测已出现的用户
        mask = get_previous_user_mask(input.cpu(), self.n_node)
        
        # 添加掩码并重塑输出，将3D张量展平为2D张量
        # 加mask是为了使已出现用户的预测概率变低（加上一个大的负值）
        # view(-1, output_u.size(-1))将[batch_size, seq_len-1, user_size]展平为[batch_size*(seq_len-1), user_size]
        output = (output_u + mask).view(-1, output_u.size(-1)).cuda()
        print(f"output输出结果{output.shape}")
        return  output # 返回预测结果，维度为[batch_size*(seq_len-1), user_size]

