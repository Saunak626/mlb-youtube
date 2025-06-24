from __future__ import division

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable

import numpy as np

# 注意：此文件是为旧版本PyTorch(0.3.1)编写的。
# 下面的代码进行了一些现代化修改(如移除Variable)，并添加了大量注释以帮助理解。
# 但要使其在现代PyTorch中完全正常工作，可能还需要进一步的调试。

def compute_pad(stride, k, s):
    """计算卷积所需的填充(padding)大小，以保持输出尺寸与输入相同。"""
    return max((k - s)//2, 0)


class TSF(nn.Module):
    """
    TSF (Temporal Structure Filter) 模块，是本代码库中很多模型的基石。
    它实现了一种可学习的时间滤波器，用于捕捉视频序列中的时序模式。
    其核心思想是，滤波器的形状（由高斯函数定义）是由可学习的参数（delta, gamma, center）决定的。
    """
    def __init__(self, N=3, mx=False):
        """
        :param N: int, 高斯滤波器的数量。
        :param mx: bool, 是否在最后使用最大池化。
        """
        super(TSF, self).__init__()
        self.N = N
        self.mx = mx
        
        # 定义可学习的参数来控制高斯滤波器的形状
        # delta: 控制高斯函数的宽度 (标准差)
        # gamma: 控制高斯函数的高度 (缩放因子)
        # center: 控制高斯函数的中心位置 (均值)
        self.delta = nn.Parameter(torch.Tensor(N))
        self.gamma = nn.Parameter(torch.Tensor(N))
        self.center = nn.Parameter(torch.Tensor(N))
        
        # 初始化这些参数
        self.delta.data.uniform_(0.1, 1)
        self.gamma.data.uniform_(0.5, 2)
        self.center.data.uniform_(-0.5, 0.5)

    def get_filters(self, delta, gamma, center, length, time):
        """
        根据当前可学习的参数，生成实际的高斯滤波器。
        :param length: int, 滤波器的总长度 (时间维度)。
        :param time: Tensor, 一个表示时间轴上各个点位置的张量。
        :return: Tensor, 生成的高斯滤波器组。
        """
        # 确保参数在合理范围内
        delta = torch.clamp(self.delta, min=0.01)
        center = torch.clamp(self.center, min=-1, max=1)

        # 调整中心位置，使其从[-1, 1]映射到[0, length]
        c = center * (length/2) + (length/2)
        c = c.unsqueeze(1).expand(-1, length)

        # 调整delta，使其与长度相关
        d = delta * length/2
        d = d.unsqueeze(1).expand(-1, length)

        # 计算高斯函数
        # exp(-((t-c)^2 / (2d^2)))
        x = torch.exp(-torch.pow(time.expand(self.N, -1) - c, 2) / (2*torch.pow(d,2)))
        
        # 乘以缩放因子 gamma
        g = gamma.unsqueeze(1).expand(-1, length)
        x = g * x
        
        # 归一化
        x = x / torch.sum(x, dim=1, keepdim=True).expand(-1, length)
        
        return x

    def forward(self, inp):
        """
        前向传播。
        :param inp: tuple, 包含 (特征, 序列长度)
        """
        x, lens = inp
        b, t, d = x.size() # batch, time, dimension
        
        # 修复: 移除 Variable 包装
        # time = Variable(torch.linspace(0, t-1, t).cuda(), requires_grad=False)
        device = x.device # 获取输入张量所在的设备 (cpu或gpu)
        time = torch.linspace(0, t-1, t, device=device)

        # 生成滤波器
        filters = self.get_filters(self.delta, self.gamma, self.center, t, time)
        filters = filters.unsqueeze(1).expand(-1,d,-1) # 调整形状以进行批量乘法

        # 将滤波器应用于输入特征
        x_t = x.permute(0,2,1)
        x_t = torch.bmm(x_t, filters.permute(0,2,1))
        
        # 如果设置了最大池化
        if self.mx:
            x_t, _ = torch.max(x_t, dim=2)
            
        return x_t.view(b, -1)


class SubConv(TSF):
    """
    Sub-event Convolution, 继承自TSF。
    它将TSF生成的滤波器用作一维卷积核。
    这是论文中 "sub-events" 模型的核心实现。
    """
    def __init__(self, inp, num_f, length):
        super(SubConv, self).__init__(N=num_f)
        self.length = length
        self.conv = nn.Conv1d(inp, num_f, length, padding=length//2)

    def forward(self, x):
        # 重写前向传播，将TSF生成的滤波器作为卷积核
        b,c,t = x.size() # batch, channel, time
        
        # 修复: 移除 Variable 包装
        # time = Variable(torch.linspace(0, self.length-1, self.length).cuda(), requires_grad=False)
        device = x.device
        time = torch.linspace(0, self.length-1, self.length, device=device)
        
        filters = self.get_filters(self.delta, self.gamma, self.center, self.length, time)
        
        # 将滤波器赋值给卷积层的权重
        self.conv.weight.data = filters.unsqueeze(1).expand(-1,c,-1)
        
        return self.conv(x)


class ContSubConv(nn.Module):
    """
    Continuous Sub-event Convolution.
    用于连续视频（逐帧预测）的子事件模型。
    """
    def __init__(self, inp, num_f, length, classes):
        super(ContSubConv, self).__init__()
        self.sub_conv = SubConv(inp, num_f, length)
        self.dropout = nn.Dropout(0.7)
        self.fc = nn.Linear(num_f, classes)
        self.sig = nn.Sigmoid()

    def forward(self, inp):
        x = self.sub_conv(inp[0].permute(0,4,1,2,3).squeeze(3).squeeze(3))
        x = self.dropout(x)
        x = x.permute(0,2,1)
        x = self.fc(x)
        return self.sig(x)


class TConv(nn.Module):
    """
    Temporal Convolution.
    实现了一个简单的一维时间卷积网络，后面接一个全连接层。
    """
    def __init__(self, inp, classes):
        super(TConv, self).__init__()
        self.conv = nn.Conv1d(inp, 256, 5, padding=2)
        self.fc = nn.Linear(256, classes)

    def forward(self, x, lens):
        x = x.permute(0,2,1)
        x = F.relu(self.conv(x))
        x = x.permute(0,2,1)
        
        # 根据每个视频的实际长度进行平均池化
        out = []
        for i,l in enumerate(lens):
            out.append(torch.mean(x[i, :l], dim=0))
        out = torch.stack(out, dim=0)
        
        return self.fc(out)


class Pyramid(nn.Module):
    """
    Temporal Pyramid Pooling.
    在时间维度上进行多尺度的池化，然后将所有池化结果拼接起来。
    这有助于模型捕捉不同时间尺度的信息。
    """
    def __init__(self, inp, classes):
        super(Pyramid, self).__init__()
        self.fc = nn.Linear(inp*4, classes)

    def forward(self, x, lens):
        b, t, d = x.size()
        
        # 定义金字塔的三个级别
        lvls = [1,2,4]
        
        feats = []
        for i in range(b):
            vid_feats = []
            for l in lvls:
                # 将视频分割成 l 个部分
                splits = torch.linspace(0, lens[i], l+1).long()
                for j in range(l):
                    # 对每个部分进行平均池化
                    st = splits[j]
                    en = splits[j+1]
                    if st==en: continue
                    vid_feats.append(torch.mean(x[i, st:en, :], dim=0))
            # 拼接所有池化结果
            feats.append(torch.cat(vid_feats, dim=0))

        feats = torch.stack(feats)
        
        return self.fc(feats)


# --- 模型构建函数 ---
# 这些函数将上述模块组合成完整的模型，供训练脚本调用。

def baseline(inp=1024, classes=1):
    """一个最简单的基线模型，只有一个线性层。"""
    return nn.Linear(inp, classes)

def sub_event(inp=1024, classes=1):
    """
    构建 Sub-event 模型。
    这是论文中的核心模型之一。
    由一个 dropout 层和一个 TSF 模块组成。
    """
    return nn.Sequential(nn.Dropout(0.7), TSF(N=512, mx=True))

def cont_sub_event(inp=1024, classes=8):
    """构建用于连续视频的子事件模型。"""
    return ContSubConv(inp, 512, 11, classes)

def tconv(inp=1024, classes=1):
    """构建 Temporal Convolution 模型。"""
    return TConv(inp, classes)

def max_pool(inp, classes):
    """一个使用最大池化的简单模型。"""
    # 这个实现看起来不完整或有误，因为它没有实际的层，
    # 只是返回了一个恒等函数。
    # 实际的池化逻辑可能在训练循环中。
    return lambda x, lens: torch.mean(x, dim=1)

def pyramid(inp, classes):
    """构建 Temporal Pyramid Pooling 模型。"""
    return Pyramid(inp, classes)
