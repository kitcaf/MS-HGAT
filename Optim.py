'''优化器的包装类'''
import numpy as np

class ScheduledOptim(object):
    '''
    学习率调度的简单包装类
    实现了Transformer论文中的学习率调度策略
    '''

    def __init__(self, optimizer, d_model, n_warmup_steps):
        """
        初始化调度优化器
        参数:
            optimizer: 基础优化器（如Adam）
            d_model: 模型维度，用于计算学习率
            n_warmup_steps: 预热步数
        """
        self.optimizer = optimizer
        self.d_model = d_model
        self.n_warmup_steps = n_warmup_steps
        self.n_current_steps = 0

    def step(self):
        "执行基础优化器的步进"
        self.optimizer.step()

    def zero_grad(self):
        "清空基础优化器的梯度"
        self.optimizer.zero_grad()

    def update_learning_rate(self):
        ''' 
        按步骤调整学习率
        实现公式: lr = d_model^(-0.5) * min(step^(-0.5), step*warmup_steps^(-1.5))
        '''
        self.n_current_steps += 1
        new_lr = np.power(self.d_model, -0.5) * np.min([
            np.power(self.n_current_steps, -0.5),
            np.power(self.n_warmup_steps, -1.5) * self.n_current_steps])

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = new_lr
