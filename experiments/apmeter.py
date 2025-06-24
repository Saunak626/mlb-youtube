import math
import meter
import numpy as np
import torch

# 注意：此文件是为旧版本PyTorch编写的。以下代码已进行部分现代化改造和注释。

class APMeter(meter.Meter):
    """
    APMeter (Average Precision Meter) 用于逐类计算平均精度。

    APMeter 设计用于处理 N x K 维度的 `output` 和 `target` 张量，
    其中 N 是样本数，K 是类别数。
    - `output`: 模型的输出分数。对于模型确信为正例的样本，分数应该更高。
    - `target`: 真实的二进制标签 (0 或 1)。
    - `weight` (可选): 每个样本的权重。
    """
    def __init__(self):
        super(APMeter, self).__init__()
        self.reset()

    def reset(self):
        """重置 meter，清空所有已存储的分数、目标和权重。"""
        # 修复：使用现代的、更安全的方式创建空Tensor。
        self.scores = torch.tensor([], dtype=torch.float32)
        self.targets = torch.tensor([], dtype=torch.int64)
        self.weights = torch.tensor([], dtype=torch.float32)

    def add(self, output, target, weight=None):
        """
        向 meter 中添加一个新的批次 (batch) 的数据。

        参数:
            output (Tensor): N x K 的张量，表示模型对N个样本属于K个类别的预测分数。
            target (Tensor): N x K 的二进制张量，表示N个样本的真实标签。
            weight (optional, Tensor): N x 1 的张量，表示每个样本的权重。
        """
        if not torch.is_tensor(output):
            output = torch.from_numpy(output)
        if not torch.is_tensor(target):
            target = torch.from_numpy(target)

        if weight is not None:
            if not torch.is_tensor(weight):
                weight = torch.from_numpy(weight)
            weight = weight.squeeze()
        if output.dim() == 1:
            output = output.view(-1, 1)
        else:
            assert output.dim() == 2, 'output 必须是一维或二维张量'
        if target.dim() == 1:
            target = target.view(-1, 1)
        else:
            assert target.dim() == 2, 'target 必须是一维或二维张量'
        if weight is not None:
            assert weight.dim() == 1, '权重必须是一维的'
            assert weight.numel() == target.size(0), '权重和目标的样本数必须相同'
            assert torch.min(weight) >= 0, '权重必须为非负数'
        
        # 确保target是二进制的 (只包含0和1)
        # assert torch.equal(target**2, target), 'target 必须是二进制的 (0或1)'
        
        if self.scores.numel() > 0:
            assert target.size(1) == self.targets.size(1), '新添加的数据维度必须与之前的数据匹配'

        # 存储分数和目标
        # 使用 torch.cat 实现更简洁高效的拼接
        self.scores = torch.cat([self.scores, output], 0)
        self.targets = torch.cat([self.targets, target.long()], 0)
        if weight is not None:
            self.weights = torch.cat([self.weights, weight], 0)

    def value(self):
        """
        计算并返回所有已添加样本的每个类别的平均精度。
        返回:
            ap (FloatTensor): 1 x K 的张量，包含每个类别的平均精度(AP)。
        """
        if self.scores.numel() == 0:
            return 0
        
        ap = torch.zeros(self.scores.size(1))
        # 修复：torch.range 已废弃，使用 torch.arange
        rg = torch.arange(1, self.scores.size(0) + 1).float()
        
        if self.weights.numel() > 0:
            # .new() 已不推荐，使用 .new_zeros() 或 .clone().detach()
            weight = self.weights.new_zeros(self.weights.size())
            weighted_truth = self.weights.new_zeros(self.weights.size())

        # 为每个类别计算AP
        for k in range(self.scores.size(1)):
            # 获取当前类别的分数和目标
            scores = self.scores[:, k]
            targets = self.targets[:, k]

            # 根据分数从高到低排序
            _, sortind = torch.sort(scores, 0, descending=True)
            truth = targets[sortind]
            
            if self.weights.numel() > 0:
                weight = self.weights[sortind]
                weighted_truth = truth.float() * weight
                rg = weight.cumsum(0)

            # 计算真正例(True Positive)的累积和
            if self.weights.numel() > 0:
                tp = weighted_truth.cumsum(0)
            else:
                tp = truth.float().cumsum(0)

            # 计算精确率曲线 (Precision Curve)
            precision = tp.div(rg)

            # 计算平均精度 (Average Precision)
            # 修复：.byte() 已废弃，使用 .bool()
            ap[k] = precision[truth.bool()].sum() / max(truth.sum(), 1)
            
        return ap
