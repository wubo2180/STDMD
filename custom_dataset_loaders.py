# -*- coding: utf-8 -*-
"""
自定义数据集加载器
从本地 data 文件夹加载时序图数据，替换 torch_geometric_temporal 的远程加载
"""
import json
import numpy as np
import torch
from torch_geometric.data import Data
from typing import List, Union
import os


class StaticGraphTemporalSignal:
    """
    静态图的时序信号
    图结构不变，节点特征随时间变化
    """
    
    def __init__(self, edge_index, edge_weight, features, targets):
        """
        参数:
            edge_index: 边索引 [2, num_edges]
            edge_weight: 边权重 [num_edges]
            features: 节点特征序列 [num_timesteps, num_nodes, num_features]
            targets: 目标值序列 [num_timesteps, num_nodes]
        """
        self.edge_index = edge_index
        self.edge_weight = edge_weight
        self.features = features
        self.targets = targets
        self._num_timesteps = len(features)
        
    def __len__(self):
        return self._num_timesteps
    
    def __getitem__(self, time_index):
        """获取指定时间步的数据快照"""
        if isinstance(time_index, slice):
            # 支持切片操作
            return [self._get_snapshot(t) for t in range(*time_index.indices(len(self)))]
        else:
            return self._get_snapshot(time_index)
    
    def _get_snapshot(self, time_index):
        """创建单个时间步的数据快照"""
        x = torch.FloatTensor(self.features[time_index])
        edge_index = torch.LongTensor(self.edge_index)
        edge_attr = torch.FloatTensor(self.edge_weight) if self.edge_weight is not None else None
        y = torch.FloatTensor(self.targets[time_index])
        
        snapshot = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)
        return snapshot
    
    def __iter__(self):
        """支持迭代"""
        for t in range(self._num_timesteps):
            yield self._get_snapshot(t)


class EnglandCovidDatasetLoader:
    """英国COVID-19数据集加载器"""
    
    def __init__(self, data_path="data/england_covid.json"):
        self.data_path = data_path
        self._read_data()
    
    def _read_data(self):
        """从本地JSON文件读取数据"""
        with open(self.data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # 解析边信息
        self._edge_index = np.array(data['edges']).T  # [2, num_edges]
        self._edge_weight = np.array(data.get('weights', [1.0] * len(data['edges'])))
        
        # 处理特征和标签，兼容不同的数据格式
        if 'FX' in data:
            self._features = np.array(data['FX'])
            self._targets = np.array(data['y'])
        elif 'X' in data:
            X = np.array(data['X'])
            if len(X.shape) == 2:
                num_timesteps, num_nodes = X.shape
                features = np.zeros((num_timesteps, num_nodes, 4))
                for t in range(num_timesteps):
                    features[t, :, 0] = X[t, :]
                    if t > 0:
                        features[t, :, 1] = X[t-1, :]
                    if t > 1:
                        features[t, :, 2] = X[t-2, :]
                    if t > 2:
                        features[t, :, 3] = (X[t-3, :] + X[t-2, :] + X[t-1, :]) / 3
                self._features = features
                self._targets = X
            else:
                self._features = X
                self._targets = np.array(data.get('y', X[:, :, 0]))
        else:
            raise ValueError("数据文件中必须包含 'FX' 或 'X' 字段")
        
        print(f"✓ 加载 England COVID-19 数据集:")
        print(f"  时间步数: {len(self._features)}")
        print(f"  节点数: {self._features.shape[1]}")
        print(f"  特征维度: {self._features.shape[2]}")
        print(f"  边数: {self._edge_index.shape[1]}")
        print(f"  标签形状: {self._targets.shape}")
    
    def get_dataset(self, num_timesteps=None):
        """
        获取数据集
        
        参数:
            num_timesteps: 要使用的时间步数（None表示全部）
        """
        if num_timesteps is not None:
            features = self._features[:num_timesteps]
            targets = self._targets[:num_timesteps]
        else:
            features = self._features
            targets = self._targets
        
        return StaticGraphTemporalSignal(
            edge_index=self._edge_index,
            edge_weight=self._edge_weight,
            features=features,
            targets=targets
        )


class PedalMeDatasetLoader:
    """PedalMe 伦敦数据集加载器"""
    
    def __init__(self, data_path="data/pedalme_london.json"):
        self.data_path = data_path
        self._read_data()
    
    def _read_data(self):
        """从本地JSON文件读取数据"""
        with open(self.data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # 解析边信息
        self._edge_index = np.array(data['edges']).T  # [2, num_edges]
        self._edge_weight = np.array(data.get('weights', [1.0] * len(data['edges'])))
        
        # 处理特征和标签，兼容不同的数据格式
        if 'FX' in data:
            # 格式1: FX (特征) + y (标签)
            self._features = np.array(data['FX'])  # [num_timesteps, num_nodes, num_features]
            self._targets = np.array(data['y'])    # [num_timesteps, num_nodes]
        elif 'X' in data:
            # 格式2: 只有 X (可能是标签)
            X = np.array(data['X'])  # [num_timesteps, num_nodes]
            
            # X 看起来是 2D 的 [timesteps, nodes]，需要扩展为 3D
            if len(X.shape) == 2:
                # 使用 X 作为标签，创建简单的特征（用历史值）
                num_timesteps, num_nodes = X.shape
                # 创建特征：使用当前值和简单统计
                features = np.zeros((num_timesteps, num_nodes, 4))
                for t in range(num_timesteps):
                    features[t, :, 0] = X[t, :]  # 当前值
                    if t > 0:
                        features[t, :, 1] = X[t-1, :]  # 前一时刻
                    if t > 1:
                        features[t, :, 2] = X[t-2, :]  # 前两时刻
                    if t > 2:
                        features[t, :, 3] = (X[t-3, :] + X[t-2, :] + X[t-1, :]) / 3  # 移动平均
                
                self._features = features
                self._targets = X  # 使用 X 作为预测目标
            else:
                # X 已经是 3D [timesteps, nodes, features]
                self._features = X
                # 如果有 y，使用它；否则使用 X 的第一个特征
                if 'y' in data:
                    self._targets = np.array(data['y'])
                else:
                    self._targets = X[:, :, 0]  # 使用第一个特征作为目标
        else:
            raise ValueError("数据文件中必须包含 'FX' 或 'X' 字段")
        
        print(f"✓ 加载 PedalMe London 数据集:")
        print(f"  时间步数: {len(self._features)}")
        print(f"  节点数: {self._features.shape[1]}")
        print(f"  特征维度: {self._features.shape[2]}")
        print(f"  边数: {self._edge_index.shape[1]}")
        print(f"  标签形状: {self._targets.shape}")
    
    def get_dataset(self, num_timesteps=None):
        """
        获取数据集
        
        参数:
            num_timesteps: 要使用的时间步数（None表示全部）
        """
        if num_timesteps is not None:
            features = self._features[:num_timesteps]
            targets = self._targets[:num_timesteps]
        else:
            features = self._features
            targets = self._targets
        
        return StaticGraphTemporalSignal(
            edge_index=self._edge_index,
            edge_weight=self._edge_weight,
            features=features,
            targets=targets
        )


class WikiMathsDatasetLoader:
    """WikiMaths 数学数据集加载器"""
    
    def __init__(self, data_path="data/wikivital_mathematics.json"):
        self.data_path = data_path
        print(f"\n[WikiMathsDatasetLoader] 初始化")
        print(f"  数据路径: {self.data_path}")
        print(f"  文件存在: {os.path.exists(self.data_path)}")
        self._read_data()
    
    def _read_data(self):
        """从本地JSON文件读取数据"""
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(
                f"数据文件不存在: {self.data_path}\n"
                f"请确保数据文件在正确的位置。"
            )
        
        print(f"  正在读取文件...")
        with open(self.data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # 解析边信息
        self._edge_index = np.array(data['edges']).T  # [2, num_edges]
        self._edge_weight = np.array(data.get('weights', [1.0] * len(data['edges'])))
        
        # 处理特征和标签，兼容不同的数据格式
        if 'FX' in data:
            self._features = np.array(data['FX'])
            self._targets = np.array(data['y'])
        elif 'X' in data:
            X = np.array(data['X'])
            if len(X.shape) == 2:
                num_timesteps, num_nodes = X.shape
                features = np.zeros((num_timesteps, num_nodes, 4))
                for t in range(num_timesteps):
                    features[t, :, 0] = X[t, :]
                    if t > 0:
                        features[t, :, 1] = X[t-1, :]
                    if t > 1:
                        features[t, :, 2] = X[t-2, :]
                    if t > 2:
                        features[t, :, 3] = (X[t-3, :] + X[t-2, :] + X[t-1, :]) / 3
                self._features = features
                self._targets = X
            else:
                self._features = X
                self._targets = np.array(data.get('y', X[:, :, 0]))
        else:
            raise ValueError("数据文件中必须包含 'FX' 或 'X' 字段")
        
        print(f"✓ 加载 WikiMaths 数据集:")
        print(f"  时间步数: {len(self._features)}")
        print(f"  节点数: {self._features.shape[1]}")
        print(f"  特征维度: {self._features.shape[2]}")
        print(f"  边数: {self._edge_index.shape[1]}")
        print(f"  标签形状: {self._targets.shape}")
    
    def get_dataset(self, num_timesteps=None):
        """
        获取数据集
        
        参数:
            num_timesteps: 要使用的时间步数（None表示全部）
        """
        if num_timesteps is not None:
            features = self._features[:num_timesteps]
            targets = self._targets[:num_timesteps]
        else:
            features = self._features
            targets = self._targets
        
        print(f"\n[WikiMathsDatasetLoader] 创建 StaticGraphTemporalSignal")
        print(f"  使用的类: {StaticGraphTemporalSignal}")
        print(f"  类所在模块: {StaticGraphTemporalSignal.__module__}")
        
        dataset = StaticGraphTemporalSignal(
            edge_index=self._edge_index,
            edge_weight=self._edge_weight,
            features=features,
            targets=targets
        )
        
        print(f"  创建的对象类型: {type(dataset)}")
        print(f"  对象模块: {type(dataset).__module__}")
        
        return dataset


def temporal_signal_split(dataset, train_ratio=0.8):
    """
    将时序数据集分割为训练集和测试集
    
    参数:
        dataset: StaticGraphTemporalSignal 对象（自定义或torch_geometric_temporal版本）
        train_ratio: 训练集比例
    
    返回:
        train_dataset, test_dataset
    """
    # 检查是否是我们自定义的 StaticGraphTemporalSignal
    is_custom = type(dataset).__module__ == 'custom_dataset_loaders'
    
    if not is_custom:
        # 如果是 torch_geometric_temporal 的版本，转换为自定义版本
        print(f"\n⚠️ 警告: 检测到使用 torch_geometric_temporal 的 StaticGraphTemporalSignal")
        print(f"   正在转换为自定义实现...")
        
        # 提取数据
        try:
            # 从 torch_geometric_temporal 对象中提取数据
            snapshots = list(dataset)
            edge_index = snapshots[0].edge_index.numpy()
            edge_weight = snapshots[0].edge_attr.numpy() if hasattr(snapshots[0], 'edge_attr') and snapshots[0].edge_attr is not None else None
            
            features = []
            targets = []
            for snapshot in snapshots:
                features.append(snapshot.x.numpy())
                targets.append(snapshot.y.numpy())
            
            features = np.array(features)
            targets = np.array(targets)
            
            # 创建自定义版本
            dataset = StaticGraphTemporalSignal(
                edge_index=edge_index,
                edge_weight=edge_weight,
                features=features,
                targets=targets
            )
            print(f"   ✓ 转换完成")
            
        except Exception as e:
            raise TypeError(f"无法转换 torch_geometric_temporal 的 StaticGraphTemporalSignal: {e}")
    
    # 获取时间步数
    try:
        num_timesteps = len(dataset)
    except TypeError:
        # 如果 __len__ 方法有问题，直接从 features 获取
        num_timesteps = dataset._num_timesteps if hasattr(dataset, '_num_timesteps') else len(dataset.features)
    
    train_size = int(num_timesteps * train_ratio)
    
    # 训练集
    train_features = dataset.features[:train_size]
    train_targets = dataset.targets[:train_size]
    train_dataset = StaticGraphTemporalSignal(
        edge_index=dataset.edge_index,
        edge_weight=dataset.edge_weight,
        features=train_features,
        targets=train_targets
    )
    
    # 测试集
    test_features = dataset.features[train_size:]
    test_targets = dataset.targets[train_size:]
    test_dataset = StaticGraphTemporalSignal(
        edge_index=dataset.edge_index,
        edge_weight=dataset.edge_weight,
        features=test_features,
        targets=test_targets
    )
    
    print(f"\n数据集分割:")
    print(f"  训练集: {len(train_dataset)} 时间步")
    print(f"  测试集: {len(test_dataset)} 时间步")
    
    return train_dataset, test_dataset


# 测试代码
if __name__ == "__main__":
    print("=" * 60)
    print("测试自定义数据集加载器")
    print("=" * 60)
    print()
    
    # 测试 EnglandCovid
    print("[1/3] 测试 England COVID-19 数据集")
    print("-" * 60)
    try:
        loader = EnglandCovidDatasetLoader()
        dataset = loader.get_dataset()
        print(f"✓ 数据集大小: {len(dataset)} 时间步")
        
        # 测试获取单个快照
        snapshot = dataset[0]
        print(f"✓ 快照测试: x={snapshot.x.shape}, y={snapshot.y.shape}")
        print()
    except Exception as e:
        print(f"✗ 加载失败: {e}")
        print()
    
    # 测试 PedalMe
    print("[2/3] 测试 PedalMe London 数据集")
    print("-" * 60)
    try:
        loader = PedalMeDatasetLoader()
        dataset = loader.get_dataset()
        print(f"✓ 数据集大小: {len(dataset)} 时间步")
        
        snapshot = dataset[0]
        print(f"✓ 快照测试: x={snapshot.x.shape}, y={snapshot.y.shape}")
        print()
    except Exception as e:
        print(f"✗ 加载失败: {e}")
        print()
    
    # 测试 WikiMaths
    print("[3/3] 测试 WikiMaths 数据集")
    print("-" * 60)
    try:
        loader = WikiMathsDatasetLoader()
        dataset = loader.get_dataset()
        print(f"✓ 数据集大小: {len(dataset)} 时间步")
        
        snapshot = dataset[0]
        print(f"✓ 快照测试: x={snapshot.x.shape}, y={snapshot.y.shape}")
        
        # 测试数据分割
        train_dataset, test_dataset = temporal_signal_split(dataset, train_ratio=0.8)
        print()
    except Exception as e:
        print(f"✗ 加载失败: {e}")
        print()
    
    print("=" * 60)
    print("✓ 所有测试完成！")
    print("=" * 60)
