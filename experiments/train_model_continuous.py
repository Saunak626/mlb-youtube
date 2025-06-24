import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import argparse

# 导入本地模块
import mlb_continuous_models as models
import mlb_i3d_per_video as dset
from apmeter import APMeter

# --- 参数解析 ---
def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

parser = argparse.ArgumentParser(description='Train continuous action detection models')
parser.add_argument('--model', type=str, default='get_tsf_model', help='Model to use (e.g., get_tsf_model, get_baseline_model)')
parser.add_argument('--root', type=str, default='../data/i3d_features_continuous', help='Path to continuous video feature files (.npy)')
parser.add_argument('--split-file', type=str, default='../data/mlb-youtube-continuous.json', help='Path to the continuous annotation JSON file')
parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs')
parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
parser.add_argument('--batch-size', type=int, default=1, help='Batch size for continuous detection is typically 1')
parser.add_argument('--gpu', type=str, default='0', help='GPU device ID to use')
parser.add_argument('--save-dir', type=str, default='saved_models_continuous', help='Directory to save trained models')

args = parser.parse_args()

# --- 训练主函数 ---
def run(model, criterion, optimizer, dataloaders, device, num_epochs=50):
    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            ap_meter = APMeter()
            total_loss = 0.0

            for data in dataloaders[phase]:
                inputs, mask, labels, _ = data
                inputs = inputs.to(device)
                labels = labels.to(device).float()
                mask = mask.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    # 模型输入是 (B, C, T, H, W)
                    outputs = model(inputs)
                    # 输出形状是 (B, Classes, T, H, W), 需要调整以匹配标签
                    outputs = outputs.squeeze(-1).squeeze(-1).permute(0, 2, 1) # (B, T, C)
                    
                    # 为了计算损失，只考虑有标签的部分 (mask=1)
                    loss = criterion(outputs[mask.bool()], labels[mask.bool()])

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                total_loss += loss.item() * inputs.size(0)
                # 计算mAP时也只考虑有效部分
                ap_meter.add(torch.sigmoid(outputs[mask.bool()]).detach().cpu().numpy(), labels[mask.bool()].cpu().numpy())

            epoch_loss = total_loss / len(dataloaders[phase].dataset)
            epoch_map = ap_meter.value().mean()
            print(f'{phase} Loss: {epoch_loss:.4f} mAP: {epoch_map:.4f}')
        
        # 保存模型
        if not os.path.exists(args.save_dir):
            os.makedirs(args.save_dir)
        torch.save(model.state_dict(), os.path.join(args.save_dir, f'continuous_{args.model}_{epoch}.pt'))

    return model

if __name__ == '__main__':
    # --- 环境和设备设置 ---
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- 数据加载 ---
    datasets = {
        'train': dset.MLB(args.split_file, 'training', args.root, args.batch_size),
        'val': dset.MLB(args.split_file, 'testing', args.root, args.batch_size)
    }
    dataloaders = {
        x: torch.utils.data.DataLoader(datasets[x], batch_size=args.batch_size,
                                      shuffle=True, num_workers=4, collate_fn=dset.mt_collate_fn)
        for x in ['train', 'val']
    }

    # --- 模型、损失函数、优化器设置 ---
    num_classes = 8
    model = getattr(models, args.model)(classes=num_classes)
    model.to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # --- 开始训练 ---
    print(f"Training model for continuous detection: {args.model}")
    run(model, criterion, optimizer, dataloaders, device, num_epochs=args.epochs)
    print("Training complete.")
