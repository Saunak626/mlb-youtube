import torch
import torch.utils.data as data_utl
from torch.utils.data.dataloader import default_collate
from skimage import io

import numpy as np
import json
import random

import os
import os.path

# 注意：这个文件是为旧版本PyTorch编写的，并且包含一些bug。
# 注释将解释其原始意图并指出问题。

def video_to_tensor(pic):
    """
    将一个numpy数组转换为torch张量。
    输入: numpy.ndarray, 形状为 (T, H, W, C) - 时间, 高, 宽, 通道
    输出: torch.FloatTensor, 形状为 (C, T, H, W) - 通道, 时间, 高, 宽
    这是PyTorch中常用的视频张量格式。
    """
    return torch.from_numpy(pic.transpose([3,0,1,2]))


# 定义一个从标签名到索引的映射字典
l2i = {'ball':0,'swing':1,'strike':2,'hit':3,'foul':4,'in play':5,'bunt':6,'hit by pitch':7}
class SegmentedPitchResultMultiLabel(data_utl.Dataset):
    """
    这是一个自定义的PyTorch数据集类，用于加载分割好的视频片段及其多标签。
    它会同时处理正样本（有动作）和负样本（无动作）。

    标签定义:
      0 - ball (好球)
      1 - swing (挥棒)
      2 - strike (好球/三振)
      3 - hit (安打)
      4 - foul (界外球)
      5 - in play (活球)
      6 - bunt (触击)
      7 - hit by pitch (触身球)
    """

    def __init__(self, positive, negative, split, root):
        """
        初始化数据集。
        :param positive: str, 指向正样本JSON文件的路径 (mlb-youtube-segmented.json)
        :param negative: str, 指向负样本JSON文件的路径 (mlb-youtube-negative.json)
        :param split: str, 'training' 或 'testing'，用于划分数据集
        :param root: str, 存放视频特征文件 (.npy) 的根目录
        """
        self.root = root
        
        # 1. 加载和筛选正样本
        with open(positive, 'r') as f: 
            self.act_dict = json.load(f)
            # 这里的删除操作在迭代中进行，效率较低且不安全，更好的做法是创建一个新字典
            keys_to_del = [k for k, v in self.act_dict.items() if v.get('subset') != split]
            for k in keys_to_del:
                del self.act_dict[k]
            
        # 2. 加载负样本并合并到主字典中
        with open(negative, 'r') as f:
            self.negs = json.load(f)
        for n in self.negs.keys():
            self.negs[n]['labels'] = []  # 负样本没有标签
            self.act_dict[n] = self.negs[n]

        # 获取所有视频片段的ID/名称列表
        self.videos = list(self.act_dict.keys())
        
    def __getitem__(self, index):
        """
        根据索引获取一个数据样本。
        :param index: int, 索引
        :return: tuple, (特征, 标签, 视频名)
        """
        # 获取视频ID和对应的标签
        name = self.videos[index]
        labels = self.act_dict[name]['labels']

        # 构建特征文件的完整路径
        path = os.path.join(self.root, name+'.npy')
        
        # 创建一个长度为8的全零数组，用于存放多标签
        multilabel = np.zeros((8,))

        # 加载特征文件
        # 原始代码中这里返回的是'feat'，但'feat'未定义。应返回加载的'img'。
        feat = np.load(path)
        
        # 如果存在标签，则将对应位置设为1
        if len(labels) > 0:
            for lab in labels:
                if lab in l2i: # 确保标签在字典中
                    multilabel[l2i[lab]] = 1
        
        # 返回 特征, 多标签, 视频名
        return feat, multilabel, name

    def __len__(self):
        """
        返回数据集中样本的总数。
        """
        return len(self.videos)


def collate_fn(batch):
    """
    自定义的collate_fn函数，用于将一个batch的数据打包成合适的格式。
    主要功能是处理变长的序列数据：将所有序列填充(pad)到当前batch中最长的长度。
    """
    max_len = 0
    # 1. 找到这个batch中特征序列的最大长度
    for b in batch:
        if b[0].shape[0] > max_len:
            max_len = b[0].shape[0]

    new_batch = []
    # 2. 遍历batch中的每个样本，进行填充
    for b in batch:
        # 创建一个全零的numpy数组，作为填充后的特征容器
        f = np.zeros((max_len, b[0].shape[1], b[0].shape[2], b[0].shape[3]), np.float32)
        # 创建一个mask，用于在模型中区分哪些是真实数据，哪些是填充的0
        m = np.zeros((max_len), np.float32)
        # 标签不需要填充
        l = b[1]
        
        # 将原始数据复制到填充容器的开头
        f[:b[0].shape[0]] = b[0]
        m[:b[0].shape[0]] = 1 # mask中真实数据部分为1
        
        # 将处理后的 (特征, mask, 标签, 名字) 添加到新batch列表中
        # 注意：这里的标签l应该是torch.from_numpy(l)以保持一致性
        new_batch.append([video_to_tensor(f), torch.from_numpy(m), torch.from_numpy(l), b[2]])
        
    # 3. 使用PyTorch默认的collate_fn处理，将列表中的numpy和tensor都转换成一个大的tensor
    return default_collate(new_batch)
