# -*- coding: utf-8 -*-
"""
自定义 MAML (Model-Agnostic Meta-Learning) 实现
替换 learn2learn 库，避免依赖已停止维护的 gym 库
"""
import torch
import torch.nn as nn
from collections import OrderedDict
from copy import deepcopy


class MAML(nn.Module):
    """
    自定义 MAML 实现
    
    参数:
        model: 要进行元学习的模型
        lr: 内循环学习率 (task-level learning rate)
        first_order: 是否使用一阶近似（默认 False）
        allow_unused: 是否允许未使用的参数（默认 False）
        allow_nograd: 是否允许没有梯度的参数（默认 False）
    """
    
    def __init__(self, model, lr=0.01, first_order=False, allow_unused=False, allow_nograd=False):
        super(MAML, self).__init__()
        self.module = model
        self.lr = lr
        self.first_order = first_order
        self.allow_unused = allow_unused
        self.allow_nograd = allow_nograd
        
    def forward(self, *args, **kwargs):
        """前向传播，直接调用底层模型"""
        return self.module(*args, **kwargs)
    
    def clone(self):
        """
        克隆当前模型，用于任务级别的适应
        返回一个新的 MAML 实例，共享相同的参数
        """
        return MAMLWrapper(self)
    
    def parameters(self):
        """返回模型参数"""
        return self.module.parameters()
    
    def named_parameters(self):
        """返回命名参数"""
        return self.module.named_parameters()
    
    def state_dict(self):
        """返回模型状态字典"""
        return self.module.state_dict()
    
    def load_state_dict(self, state_dict):
        """加载模型状态"""
        return self.module.load_state_dict(state_dict)
    
    def train(self, mode=True):
        """设置训练模式"""
        self.module.train(mode)
        return self
    
    def eval(self):
        """设置评估模式"""
        self.module.eval()
        return self


class MAMLWrapper:
    """
    MAML 任务包装器
    用于在单个任务上进行快速适应（inner loop）
    """
    
    def __init__(self, maml):
        self.maml = maml
        self.module = maml.module
        self.lr = maml.lr
        self.first_order = maml.first_order
        self.allow_unused = maml.allow_unused
        self.allow_nograd = maml.allow_nograd
        
        # 存储快速权重（adapted parameters）
        self.fast_weights = None
        
    def __call__(self, *args, **kwargs):
        """
        使用快速权重进行前向传播
        如果没有快速权重，使用原始权重
        """
        if self.fast_weights is None:
            return self.module(*args, **kwargs)
        else:
            return self._forward_with_fast_weights(*args, **kwargs)
    
    def _forward_with_fast_weights(self, *args, **kwargs):
        """使用快速权重进行前向传播"""
        # 临时替换模型参数
        original_params = {}
        for name, param in self.module.named_parameters():
            original_params[name] = param.data.clone()
            if name in self.fast_weights:
                param.data = self.fast_weights[name]
        
        # 前向传播
        output = self.module(*args, **kwargs)
        
        # 恢复原始参数
        for name, param in self.module.named_parameters():
            param.data = original_params[name]
        
        return output
    
    def adapt(self, loss, allow_unused=None, allow_nograd=None):
        """
        在当前任务上进行一步梯度更新（inner loop）
        
        参数:
            loss: 支持集上的损失
            allow_unused: 是否允许未使用的参数
            allow_nograd: 是否允许没有梯度的参数
        """
        if allow_unused is None:
            allow_unused = self.allow_unused
        if allow_nograd is None:
            allow_nograd = self.allow_nograd
        
        # 计算梯度
        grads = torch.autograd.grad(
            loss,
            self.module.parameters(),
            create_graph=not self.first_order,
            allow_unused=allow_unused
        )
        
        # 初始化快速权重
        if self.fast_weights is None:
            self.fast_weights = OrderedDict()
        
        # 更新快速权重
        for (name, param), grad in zip(self.module.named_parameters(), grads):
            if grad is not None:
                if name not in self.fast_weights:
                    self.fast_weights[name] = param.clone()
                
                # 梯度下降更新
                self.fast_weights[name] = self.fast_weights[name] - self.lr * grad
            elif not allow_nograd:
                raise ValueError(f"Parameter {name} has no gradient, but allow_nograd=False")
    
    def parameters(self):
        """返回当前参数（快速权重或原始权重）"""
        if self.fast_weights is None:
            return self.module.parameters()
        else:
            return self.fast_weights.values()


class MetaLearner:
    """
    元学习器辅助类
    提供一些常用的元学习工具函数
    """
    
    @staticmethod
    def clone_model(model):
        """深拷贝模型"""
        return deepcopy(model)
    
    @staticmethod
    def update_parameters(model, lr, grads=None, params=None):
        """
        使用梯度更新模型参数
        
        参数:
            model: 要更新的模型
            lr: 学习率
            grads: 梯度列表（可选）
            params: 参数列表（可选）
        """
        if grads is not None:
            params = list(model.parameters())
            for param, grad in zip(params, grads):
                if grad is not None:
                    param.data = param.data - lr * grad
    
    @staticmethod
    def compute_gradient(loss, parameters, create_graph=True, allow_unused=True):
        """
        计算损失相对于参数的梯度
        
        参数:
            loss: 损失值
            parameters: 参数列表
            create_graph: 是否创建计算图（用于二阶导数）
            allow_unused: 是否允许未使用的参数
            
        返回:
            梯度列表
        """
        return torch.autograd.grad(
            loss,
            parameters,
            create_graph=create_graph,
            allow_unused=allow_unused
        )


if __name__ == "__main__":
    # 简单测试
    print("=" * 60)
    print("自定义 MAML 实现测试")
    print("=" * 60)
    
    # 创建一个简单的测试模型
    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(10, 1)
        
        def forward(self, x):
            return self.linear(x)
    
    # 测试 MAML
    model = SimpleModel()
    maml = MAML(model, lr=0.01)
    
    print(f"✓ MAML 创建成功")
    print(f"  学习率: {maml.lr}")
    print(f"  参数数量: {sum(p.numel() for p in maml.parameters())}")
    
    # 测试 clone
    task_model = maml.clone()
    print(f"✓ 任务模型克隆成功")
    
    # 测试 adapt
    x = torch.randn(5, 10)
    y = torch.randn(5, 1)
    output = task_model(x)
    loss = nn.MSELoss()(output, y)
    task_model.adapt(loss, allow_unused=True, allow_nograd=True)
    print(f"✓ 任务适应成功")
    
    print("=" * 60)
    print("所有测试通过！")
    print("=" * 60)
