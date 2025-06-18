# 导入必要的库
import torch
import torch.nn.init as init
import torch.nn.functional as F
import torch.nn as nn
import copy
import math

def MultileayerModule(module, N):
    """
    创建模块的多层副本
    参数:
        module: 要复制的模块
        N: 层数
    返回:
        nn.ModuleList: 包含N个模块副本的列表
    """
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        """ 
        初始化多头注意力模块
        参数:
            h: 头的数量
            d_model: 模型维度
            dropout: dropout比率，默认0.1
        """
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        self.d_k = d_model // h  # 每个头的维度
        self.h = h  # 头的数量
        self.linears = MultileayerModule(nn.Linear(d_model, d_model), 4)  # 4个线性层
        self.attn = None  # 注意力权重
        self.dropout = nn.Dropout(p=dropout)
        self.init_weights()

    def init_weights(self):
        """初始化权重"""
        for l in self.linears:
            init.xavier_normal_(l.weight)

    def forward(self, query, key, value, mask=None):
        """
        前向传播
        参数:
            query: 查询张量
            key: 键张量
            value: 值张量
            mask: 掩码张量
        返回:
            outputs: 注意力输出
        """
        if mask is not None:
            mask = mask.unsqueeze(1)  # 扩展掩码维度
        nbatches = query.size(0)  # 批次大小

        # h x d_k，将线性变换后的张量分割为多头
        query, key, value = [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
                             for l, x in zip(self.linears, (query, key, value))]
        x, self.attn = self.attention(query, key, value, dropout=self.dropout)
        # 拼接多头的结果
        x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.h * self.d_k)
        outputs = self.linears[-1](x)  # 最后一个线性层
        outputs = torch.squeeze(outputs)
        return outputs

    def attention(self, query, key, value, mask=None, dropout=None):
        """
        计算缩放点积注意力
        参数:
            query: 查询张量
            key: 键张量
            value: 值张量
            mask: 掩码张量
            dropout: dropout层
        返回:
            注意力加权的值和注意力权重
        """
        d_k = query.size(-1)
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)  # 缩放点积注意力
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)  # 掩码处理
        p_attn = F.softmax(scores, dim=-1)  # 注意力权重
        if dropout is not None:
            p_attn = dropout(p_attn)
        return torch.matmul(p_attn, value), p_attn  # 注意力加权的值和注意力权重


class PositionalEncoding(nn.Module):
    "实现位置编码函数"

    def __init__(self, d_model, dropout, max_len=5000):
        """
        初始化位置编码
        参数:
            d_model: 模型维度
            dropout: dropout比率
            max_len: 最大序列长度，默认5000
        """
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # 在对数空间中计算位置编码
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)  # 偶数位置使用正弦
        pe[:, 1::2] = torch.cos(position * div_term)  # 奇数位置使用余弦
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        前向传播
        参数:
            x: 输入张量
        返回:
            添加位置编码后的张量
        """
        x = x + self.pe[:, :x.size(1)]  # 添加位置编码
        return self.dropout(x)


class TransformerBlock(nn.Module):
    """Transformer块实现"""

    def __init__(self, input_size, d_k=64, d_v=64, n_heads=2, is_layer_norm=True, attn_dropout=0.1):
        """
        初始化Transformer块
        参数:
            input_size: 输入特征维度
            d_k: 键的维度，默认64
            d_v: 值的维度，默认64
            n_heads: 注意力头数，默认2
            is_layer_norm: 是否使用层归一化，默认True
            attn_dropout: 注意力dropout比率，默认0.1
        """
        super(TransformerBlock, self).__init__()
        self.n_heads = n_heads
        self.d_k = d_k if d_k is not None else input_size
        self.d_v = d_v if d_v is not None else input_size

        self.is_layer_norm = is_layer_norm
        if is_layer_norm:
            self.layer_norm = nn.LayerNorm(normalized_shape=input_size)

        # self.pos_encoding = PositionalEncoding(d_model=input_size, dropout=0.5)
        # 查询、键、值的权重矩阵
        self.W_q = nn.Parameter(torch.Tensor(input_size, n_heads * d_k))
        self.W_k = nn.Parameter(torch.Tensor(input_size, n_heads * d_k))
        self.W_v = nn.Parameter(torch.Tensor(input_size, n_heads * d_v))

        # 输出权重矩阵
        self.W_o = nn.Parameter(torch.Tensor(d_v*n_heads, input_size))
        # 前馈网络
        self.linear1 = nn.Linear(input_size, input_size)
        self.linear2 = nn.Linear(input_size, input_size)

        self.dropout = nn.Dropout(attn_dropout)
        self.__init_weights__()
        #print(self)

    def __init_weights__(self):
        """初始化权重"""
        init.xavier_normal_(self.W_q)
        init.xavier_normal_(self.W_k)
        init.xavier_normal_(self.W_v)
        init.xavier_normal_(self.W_o)

        init.xavier_normal_(self.linear1.weight)
        init.xavier_normal_(self.linear2.weight)

    def FFN(self, X):
        """
        前馈网络
        参数:
            X: 输入张量
        返回:
            output: 前馈网络输出
        """
        output = self.linear2(F.relu(self.linear1(X)))
        output = self.dropout(output)
        return output

    def scaled_dot_product_attention(self, Q, K, V, mask, episilon=1e-6):
        '''
        缩放点积注意力
        参数:
            Q: (*, max_q_words, n_heads, input_size) - 查询张量
            K: (*, max_k_words, n_heads, input_size) - 键张量
            V: (*, max_v_words, n_heads, input_size) - 值张量
            mask: (*, max_q_words) - 掩码张量
            episilon: 小常数，防止除零
        返回:
            V_att: 注意力加权的值
        '''
        temperature = self.d_k ** 0.5  # 缩放因子
        Q_K = torch.einsum("bqd,bkd->bqk", Q, K) / (temperature + episilon)  # 批量矩阵乘法
        if mask is not None:
            pad_mask = mask.unsqueeze(dim=-1).expand(-1, -1, K.size(1))
            mask = torch.triu(torch.ones(pad_mask.size()), diagonal=1).bool().cuda()  # 上三角掩码（用于因果注意力）
            mask_ = mask + pad_mask
            Q_K = Q_K.masked_fill(mask_, -2**32+1)  # 掩码处理

        Q_K_score = F.softmax(Q_K, dim=-1)  # (batch_size, max_q_words, max_k_words)
        Q_K_score = self.dropout(Q_K_score)
        # 维度为3的两个矩阵的乘法
        V_att = Q_K_score.bmm(V)  # (*, max_q_words, input_size)
        return V_att


    def multi_head_attention(self, Q, K, V, mask):
        '''
        多头注意力机制
        参数:
            Q: 查询张量
            K: 键张量
            V: 值张量
            mask: (bsz, max_q_words) - 掩码张量
        返回:
            output: 多头注意力输出
        '''
        bsz, q_len, _ = Q.size()
        bsz, k_len, _ = K.size()
        bsz, v_len, _ = V.size()

        # 线性变换并分割为多头
        Q_ = Q.matmul(self.W_q).view(bsz, q_len, self.n_heads, self.d_k)
        K_ = K.matmul(self.W_k).view(bsz, k_len, self.n_heads, self.d_k)
        V_ = V.matmul(self.W_v).view(bsz, v_len, self.n_heads, self.d_v)

        # 调整维度顺序以便并行处理多头
        Q_ = Q_.permute(0, 2, 1, 3).contiguous().view(bsz*self.n_heads, q_len, self.d_k)
        K_ = K_.permute(0, 2, 1, 3).contiguous().view(bsz*self.n_heads, q_len, self.d_k)
        V_ = V_.permute(0, 2, 1, 3).contiguous().view(bsz*self.n_heads, q_len, self.d_v)

        if mask is not None:
            # 扩展掩码以适应多头
            mask = mask.unsqueeze(dim=1).expand(-1, self.n_heads, -1)  # 为头轴广播
            mask = mask.reshape(-1, mask.size(-1))

        # 计算注意力
        V_att = self.scaled_dot_product_attention(Q_, K_, V_, mask)
        # 重新调整维度
        V_att = V_att.view(bsz, self.n_heads, q_len, self.d_v)
        V_att = V_att.permute(0, 2, 1, 3).contiguous().view(bsz, q_len, self.n_heads*self.d_v)

        # 最终线性变换
        output = self.dropout(V_att.matmul(self.W_o)) # (batch_size, max_q_words, input_size)
        return output


    def forward(self, Q, K, V, mask=None):
        '''
        前向传播
        参数:
            Q: (batch_size, max_q_words, input_size) - 查询张量
            K: (batch_size, max_k_words, input_size) - 键张量
            V: (batch_size, max_v_words, input_size) - 值张量
            mask: 掩码张量
        返回:  
            output: (batch_size, max_q_words, input_size) 与Q相同大小的输出
        '''
        # 多头注意力
        V_att = self.multi_head_attention(Q, K, V, mask)

        if self.is_layer_norm:
            # 第一个残差连接和层归一化
            X = self.layer_norm(Q + V_att)  # (batch_size, max_r_words, embedding_dim)
            # 第二个残差连接和层归一化
            output = self.layer_norm(self.FFN(X) + X)
        else:
            # 不使用层归一化的残差连接
            X = Q + V_att
            output = self.FFN(X) + X
        return output
