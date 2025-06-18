# 导入必要的库
import numpy as np


class Metrics(object):
    """
    评估指标类：用于计算模型预测的准确性指标
    """

    def __init__(self):
        """初始化评估指标类"""
        super().__init__()
        self.PAD = 0  # 填充标记的索引

    def apk(self, actual, predicted, k=10):
        """
        计算平均精度@k
        该函数计算两个列表之间的平均精度@k
        
        参数
        ----------
        actual : list
                 要预测的元素列表（顺序不重要）
        predicted : list
                    预测的元素列表（顺序很重要）
        k : int, 可选
            预测元素的最大数量
            
        返回值
        -------
        score : double
                输入列表之间的平均精度@k
        """
        score = 0.0
        num_hits = 0.0

        for i, p in enumerate(predicted):
            if p in actual and p not in predicted[:i]:
                num_hits += 1.0
                score += num_hits / (i + 1.0)

        # if not actual:
        # 	return 0.0
        return score / min(len(actual), k)


    def compute_metric(self, y_prob, y_true, k_list=[10, 50, 100]):
        '''
        计算多个评估指标
        
        参数:
            y_true: (#样本, ) - 真实标签
            y_pred: (#样本, #用户) - 预测概率
            k_list: 要计算指标的k值列表
            
        返回:
            scores: 包含各种指标的字典
            scores_len: 有效样本数量
        '''
        scores_len = 0
        y_prob = np.array(y_prob)
        y_true = np.array(y_true)

        # 初始化得分字典
        scores = {'hits@'+str(k):[] for k in k_list}
        scores.update({'map@'+str(k):[] for k in k_list})
        
        # 对每个样本计算指标
        for p_, y_ in zip(y_prob, y_true):
            if y_ != self.PAD:
                scores_len += 1.0
                p_sort = p_.argsort()  # 按预测概率排序
                for k in k_list:
                    topk = p_sort[-k:][::-1]  # 获取前k个预测
                    scores['hits@' + str(k)].extend([1. if y_ in topk else 0.])  # 命中率@k
                    scores['map@'+str(k)].extend([self.apk([y_], topk, k)])  # 平均精度@k

        # 计算平均值
        scores = {k: np.mean(v) for k, v in scores.items()}
        return scores, scores_len


