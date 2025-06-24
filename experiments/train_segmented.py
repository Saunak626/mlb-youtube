from __future__ import division
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable # 警告: Variable 已被废弃
import numpy as np
import argparse

# 导入本地模块
import models
import segmented_dataset as dset
from apmeter import APMeter

# --- 参数解析 ---
def str2bool(v):
    """一个辅助函数，用于将字符串转换为布尔值。"""
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

parser = argparse.ArgumentParser(description='Train action recognition models on segmented videos')
parser.add_argument('--model', type=str, default='sub_event', help='Model to use (e.g., sub_event, tconv, pyramid)')
parser.add_argument('--root', type=str, default='../data/i3d_features', help='Path to the directory with I3D feature files (.npy)')
parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs')
parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
parser.add_argument('--gpu', type=str, default='0', help='GPU device ID to use')
parser.add_argument('--save-dir', type=str, default='saved_models', help='Directory to save trained models')

args = parser.parse_args()

# --- 训练主函数 ---
def train_model(model, criterion, optimizer, num_epochs=50):
    """
    模型训练和评估的主循环。
    """
    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)

        # 每个epoch都有一个训练阶段和一个验证阶段
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # 设置模型为训练模式
            else:
                model.eval()   # 设置模型为评估模式

            total_loss = 0.0
            ap_meter = APMeter() # 用于评估平均精度的工具

            # 迭代数据
            for data in dataloaders[phase]:
                # 1. 准备输入数据并转移到指定设备 (GPU)
                # 修复: 移除 Variable, 使用 .to(device)
                inputs, mask, labels, _ = data
                inputs = inputs.to(device)
                labels = labels.to(device).float()
                
                # 2. 梯度清零
                optimizer.zero_grad()

                # 3. 前向传播
                with torch.set_grad_enabled(phase == 'train'):
                    # 将特征和长度（这里用mask代替）作为元组传入
                    model_input = (inputs.squeeze(2), mask.sum(dim=1))
                    outputs = model(model_input)
                    
                    # 计算损失
                    loss = criterion(outputs, labels)

                    # 4. 反向传播和优化（仅在训练阶段）
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                
                # 统计损失和精度
                total_loss += loss.item() * inputs.size(0)
                ap_meter.add(torch.sigmoid(outputs).detach().cpu().numpy(), labels.cpu().numpy())

            epoch_loss = total_loss / dataset_sizes[phase]
            epoch_ap = ap_meter.value().mean()

            print(f'{phase} Loss: {epoch_loss:.4f} mAP: {epoch_ap:.4f}')

        # 在每个epoch后保存模型
        if not os.path.exists(args.save_dir):
            os.makedirs(args.save_dir)
        torch.save(model.state_dict(), os.path.join(args.save_dir, f'{args.model}_{epoch}.pt'))

    return model

if __name__ == '__main__':
    # --- 环境和设备设置 ---
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- 数据加载 ---
    # 定义JSON文件路径
    pos_json = '../data/mlb-youtube-segmented.json'
    neg_json = '../data/mlb-youtube-negative.json'
    
    # 创建训练和验证数据集
    image_datasets = {
        'train': dset.SegmentedPitchResultMultiLabel(pos_json, neg_json, 'training', args.root),
        'val': dset.SegmentedPitchResultMultiLabel(pos_json, neg_json, 'testing', args.root)
    }

    # 创建数据加载器 (Dataloaders)
    dataloaders = {
        x: torch.utils.data.DataLoader(image_datasets[x], batch_size=args.batch_size,
                                      shuffle=True, num_workers=4, collate_fn=dset.collate_fn)
        for x in ['train', 'val']
    }
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}

    # --- 模型、损失函数、优化器设置 ---
    # 1. 根据命令行参数构建模型
    num_classes = 8 # 8个动作类别
    model = getattr(models, args.model)(inp=1024, classes=num_classes)
    model.to(device) # 将模型转移到GPU
    
    # 2. 定义损失函数
    # BCEWithLogitsLoss 更稳定，因为它内部集成了 sigmoid
    criterion = nn.BCEWithLogitsLoss()

    # 3. 定义优化器
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # --- 开始训练 ---
    print(f"Training model: {args.model}")
    trained_model = train_model(model, criterion, optimizer, num_epochs=args.epochs)
    print("Training complete.")
