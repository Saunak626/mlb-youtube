import torch
import torch.utils.data as data_utl
from torch.utils.data.dataloader import default_collate

import numpy as np
import json
import csv
import h5py

import os
import os.path

# 注意：此文件是为旧版本PyTorch编写的，并且包含语法错误。
# 注释将解释其原始意图并修复错误。

def video_to_tensor(pic):
    """
    将一个numpy数组转换为torch张量。
    输入: numpy.ndarray, 形状为 (T, H, W, C) - 时间, 高, 宽, 通道
    输出: torch.FloatTensor, 形状为 (C, T, H, W) - 通道, 时间, 高, 宽
    """
    return torch.from_numpy(pic.transpose([3,0,1,2]))


def make_dataset(split_file, split, root, num_classes=8):
    """
    构建数据集的核心函数，用于扫描JSON文件并准备数据列表。
    :param split_file: str, 指向包含视频标注的JSON文件路径。
    :param split: str, 'training' 或 'testing'，用于选择数据集的子集。
    :param root: str, 存放视频特征文件 (.npy) 的根目录。
    :param num_classes: int, 类别总数。
    :return: list, 一个包含 (视频ID, 逐帧标签, 视频时长) 元组的列表。
    """
    dataset = []
    with open(split_file, 'r') as f:
        data = json.load(f)

    # 修复：将错误的中文全角引号 ' 和 ' 替换为标准的英文单引号 '
    label2cls = {'hit':0, 'in play':1, 'strike':2,'ball':3,'hit by pitch':4, 'foul':5, 'swing':6, 'bunt':7}
    
    total_in_split = 0
    skipped_count = 0

    for vid in data.keys():
        # 1. 根据 split (training/testing) 筛选视频
        if data[vid].get('subset') != split:
            continue
        
        total_in_split += 1
        video_feature_path = os.path.join(root, vid+'.npy')

        # 2. 检查对应的特征文件是否存在，如果不存在则跳过
        if not os.path.exists(video_feature_path):
            skipped_count += 1
            continue
        
        # 3. 加载特征文件以获取总帧数
        fts = np.load(video_feature_path)
        num_feat = fts.shape[0] # 特征的长度代表了总帧数
        label = np.zeros((num_feat, num_classes), np.float32)

        # 4. 计算视频的帧率 (fps)
        # 修复：使用正确的英文引号
        duration = data[vid]['end'] - data[vid]['start']
        fps = num_feat / duration

        # 5. 为每一帧生成标签
        # 遍历该视频的所有标注
        # 修复：使用正确的英文引号
        for ann in data[vid]['annotations']:
            # 遍历所有帧
            for fr in range(0,num_feat,1):
                # 如果当前帧的时间戳位于标注的起止时间内，则打上标签
                # 修复：使用正确的英文引号
                if fr/fps > ann['segment'][0] and fr/fps < ann['segment'][1]:
                    label[fr, label2cls[ann['label']]] = 1 # 二元分类标签
        
        # 修复：使用正确的英文引号
        dataset.append((vid, label, data[vid]['duration']))
    
    # 如果所有在split中的视频都被跳过了，打印一个警告
    if total_in_split > 0 and skipped_count == total_in_split:
        print("="*80)
        print(f"!! 警告: 在 '{split}' 数据集中, JSON文件里有 {total_in_split} 个视频, 但全部被跳过。")
        print(f"!! 请检查您的特征文件路径是否正确，以及该路径下是否存在 .npy 文件。")
        print(f"!! 当前检查的路径: '{os.path.abspath(root)}'")
        print("="*80)

    return dataset

class MLB(data_utl.Dataset):
    """
    为连续视频动作检测任务定义的PyTorch数据集类。
    """
    def __init__(self, split_file, split, root, batch_size):
        self.data = make_dataset(split_file, split, root)
        self.split_file = split_file
        self.batch_size = batch_size
        self.root = root
        # 使用一个字典做内存缓存，避免重复加载大的特征文件
        self.in_mem = {}

    def __getitem__(self, index):
        """
        根据索引获取一个数据样本。
        """
        entry = self.data[index]
        # 如果特征已在内存中，直接读取
        if entry[0] in self.in_mem:
            feat = self.in_mem[entry[0]]
        else:
            # 否则从磁盘加载 .npy 文件
            feat = np.load(os.path.join(self.root, entry[0]+'.npy'))
            # 调整特征形状以适应模型输入
            feat = feat.reshape((feat.shape[0],1,1,1024))
            feat = feat.astype(np.float32)
            # self.in_mem[entry[0]] = feat # 可以取消注释以启用内存缓存
            
        label = entry[1]
        return feat, label, [entry[0], entry[2]]

    def __len__(self):
        return len(self.data)


def mt_collate_fn(batch):
    """
    自定义的collate_fn，用于打包一个批次的数据。
    主要功能是处理变长的序列数据：将所有序列填充(pad)到当前batch中最长的长度。
    """
    max_len = 0
    for b in batch:
        if b[0].shape[0] > max_len:
            max_len = b[0].shape[0]

    new_batch = []
    for b in batch:
        f = np.zeros((max_len, b[0].shape[1], b[0].shape[2], b[0].shape[3]), np.float32)
        m = np.zeros((max_len), np.float32)
        l = np.zeros((max_len, b[1].shape[1]), np.float32)
        f[:b[0].shape[0]] = b[0]
        m[:b[0].shape[0]] = 1
        l[:b[0].shape[0], :] = b[1]
        new_batch.append([video_to_tensor(f), torch.from_numpy(m), torch.from_numpy(l), b[2]])

    return default_collate(new_batch)
    
