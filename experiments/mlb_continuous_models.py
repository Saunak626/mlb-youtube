import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable # 警告：Variable 已被废弃

import numpy as np
# 假设 temporal_structure_filter 是一个本地文件，我们将其导入
# 在原始代码中，这部分逻辑可能与 models.py 中的 TSF 类似
import models as tsf

# 注意：此文件同样是为旧版本PyTorch编写的。
# 下面的代码进行了一些现代化修改和注释。

class SuperEvent(nn.Module):
    """
    Super-event 模型，用于连续视频中的动作检测。
    这个模型的核心思想是结合一个全局的 "super-event" 概念和逐帧的分类。
    "super-event" 捕捉了视频片段的整体上下文信息。
    """
    def __init__(self, classes=65): # 原始类别数为65，但本项目中应为8
        super(SuperEvent, self).__init__()

        self.classes = classes
        self.dropout = nn.Dropout(0.7)

        # 两个并行的 TSF 模块来学习不同的时间结构
        self.super_event = tsf.TSF(3)
        self.super_event2 = tsf.TSF(3)

        # 可学习的权重，用于结合两个TSF模块的输出
        self.cls_wts = nn.Parameter(torch.Tensor(classes))
        
        # Super-event 的权重矩阵，用于将 TSF 输出映射到每个类别
        self.sup_mat = nn.Parameter(torch.Tensor(1, classes, 1024))
        
        # 逐帧分类器，一个简单的 1x1x1 卷积
        self.per_frame = nn.Conv3d(1024, classes, (1,1,1))
        
        # 初始化参数
        stdv = 1./np.sqrt(1024)
        self.sup_mat.data.uniform_(-stdv, stdv)
        self.per_frame.weight.data.uniform_(-stdv, stdv)
        self.per_frame.bias.data.uniform_(-stdv, stdv)
        
    def forward(self, inp):
        """
        :param inp: tuple, (特征, 长度)
        """
        # 修复：输入不再是元组，直接是特征张量
        features = inp
        # 修复：移除Variable，直接对输入应用dropout
        features = self.dropout(features)

        # 1. 计算 Super-event 特征
        # 将输入传递给两个并行的 TSF 模块
        # 注意：这里的输入维度需要与 TSF 模块的期望输入匹配
        tsf_input = (features.squeeze(-1).squeeze(-1).permute(0, 2, 1), None) # 模拟 (B, T, D) 输入
        se1 = self.super_event(tsf_input)
        se2 = self.super_event2(tsf_input)
        
        super_event_features = torch.stack([se1, se2], dim=1) # (B, 2, D*3)
        
        # 2. 结合 Super-event 特征
        # 使用 sigmoid 确保权重在0-1之间
        cls_wts = torch.stack([torch.sigmoid(self.cls_wts), 1-torch.sigmoid(self.cls_wts)], dim=1)
        # 扩展权重以匹配批次大小
        cls_wts = cls_wts.expand(features.size(0), -1, -1)
        # 使用批量矩阵乘法进行加权平均
        super_event_features = torch.bmm(cls_wts, super_event_features) # (B, C, D*3)

        # 3. 应用 super-event 权重
        # 将 super-event 特征与权重矩阵相乘并求和
        super_event_out = torch.sum(self.sup_mat * super_event_features, dim=2)
        # 扩展维度以匹配逐帧输出
        super_event_out = super_event_out.unsqueeze(2).unsqueeze(3).unsqueeze(4)

        # 4. 计算逐帧分类结果
        per_frame_out = self.per_frame(features)
        
        # 5. 返回最终结果
        return super_event_out + per_frame_out


def get_baseline_model(classes=8):
    """一个简单的基线模型，只包含一个逐帧分类器。"""
    model = nn.Sequential(
        nn.Dropout(0.7),
        nn.Conv3d(1024, classes, (1,1,1))
    )
    return model

def get_tsf_model(classes=8):
    """构建 SuperEvent 模型。"""
    # 原始代码中这里是 PerFramev4，但该类未定义，
    # 根据上下文推断它应该是 SuperEvent 模型。
    model = SuperEvent(classes=classes)
    return model

