import os
import yaml
import torch
import argparse
import numpy as np
from tqdm import tqdm
from pathlib import Path
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
from torch.amp import GradScaler, autocast
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchmetrics.image.ssim import StructuralSimilarityIndexMeasure

# 项目内模块
from models.dncnn import LightDnCNN
from scripts.data_loader import get_dataloader

class HybridLoss(nn.Module):
    """混合损失函数（L1 + SSIM）"""
    def __init__(self, alpha=0.7, device='cuda'):
        super().__init__()
        self.alpha = alpha
        self.l1_loss = nn.L1Loss()
        self.ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
        
    def forward(self, output, target):
        l1 = self.l1_loss(output, target)
        ssim = 1 - self.ssim_metric(output, target)  # SSIM越大越好，所以用1-SSIM作为损失
        return self.alpha * l1 + (1 - self.alpha) * ssim

def calculate_psnr(img1, img2):
    """计算PSNR指标"""
    mse = torch.mean((img1 - img2) ** 2)
    return 10 * torch.log10(1.0 / (mse + 1e-8))

def train(config_path, experiment_name, resume=False):
    # 自动选择最优卷积算法
    torch.backends.cudnn.benchmark = True

    # 加载配置
    with open(config_path, encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # 设备设置
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 实验目录设置
    exp_dir = Path("experiments") / f"{datetime.now().strftime('%Y%m%d')}_{experiment_name}"
    exp_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir = exp_dir / "checkpoints"
    checkpoints_dir.mkdir(exist_ok=True)
    
    # 备份配置文件
    with open(exp_dir / "config_copy.yaml", "w") as f:
        yaml.dump(config, f)

    # 初始化混合精度训练
    scaler = GradScaler('cuda', enabled=config["training"].get("use_amp", True))
    
    # 初始化模型
    model = LightDnCNN(**config["model"]["params"]).to(device)
    
    # 初始化优化器和损失函数
    optimizer = torch.optim.Adam(model.parameters(), lr=config["training"]["lr"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='min' if config["training"]["monitor"] == "val_loss" else 'max',
        factor=0.5,
        patience=5
    )
    criterion = HybridLoss(alpha=0.7, device=device)

    # 早停机制初始化
    early_stop = config["training"].get("early_stop", False)
    patience = config["training"].get("patience", 20)
    monitor_metric = config["training"].get("monitor", "ssim")
    epochs_without_improve = 0
    best_metric = float("inf") if monitor_metric == "val_loss" else -float("inf")
    
    # 断点续训逻辑
    start_epoch = 0
    if resume:
        checkpoint_path = checkpoints_dir / "latest_checkpoint.pth"
        if checkpoint_path.exists():
            checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
            model.load_state_dict(checkpoint["model_state"])
            optimizer.load_state_dict(checkpoint["optimizer_state"])
            scaler.load_state_dict(checkpoint["scaler_state"])
            start_epoch = checkpoint["epoch"] + 1
            best_metric = checkpoint["best_metric"]
            print(f"Resuming training from epoch {start_epoch}")

    # 获取数据加载器
    train_loader = get_dataloader(
        config_path=config["data_config"], 
        mode="train", 
        batch_size=config["training"].get("batch_size", 32)
    )
    val_loader = get_dataloader(
        config_path=config["data_config"],
        mode="test",
        batch_size=config["training"].get("val_batch_size", 16)
    )
    
    # 日志记录
    writer = SummaryWriter(exp_dir / "logs")
    ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)

    # 训练循环
    for epoch in range(start_epoch, config["training"]["epochs"]):
        model.train()
        epoch_loss = 0.0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config['training']['epochs']}")
        
        for noisy, clean in progress_bar:
            noisy = noisy.to(device, dtype=torch.float32)
            clean = clean.to(device, dtype=torch.float32)
            
            # 混合精度前向传播
            with autocast('cuda', enabled=config["training"].get("use_amp", True)):
                outputs = model(noisy)
                loss = criterion(outputs, clean)
            
            # 混合精度反向传播
            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            
            epoch_loss += loss.item()
            progress_bar.set_postfix({"Loss": loss.item()})
        
        # 验证阶段
        model.eval()
        val_loss = 0.0
        psnr_values = []
        ssim_values = []
        with torch.no_grad():
            for noisy, clean in val_loader:
                noisy = noisy.to(device)
                clean = clean.to(device)
                
                outputs = model(noisy)
                val_loss += criterion(outputs, clean).item()
                psnr_values.append(calculate_psnr(outputs, clean).item())
                ssim_values.append(ssim_metric(outputs, clean).item())

                # 可视化样本
                if epoch % 5 == 0 and len(psnr_values) == 1:  # 每个epoch记录第一批结果
                    viz_images = torch.cat([noisy[:3], outputs[:3], clean[:3]], dim=0)
                    grid = torchvision.utils.make_grid(viz_images, nrow=3, normalize=True)
                    writer.add_image('Validation Samples', grid, epoch)
        
        # 计算平均指标
        avg_train_loss = epoch_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        avg_psnr = np.mean(psnr_values)
        avg_ssim = np.mean(ssim_values)
        
        # 学习率调整
        current_metric = avg_val_loss if monitor_metric == "val_loss" else \
                        avg_psnr if monitor_metric == "psnr" else avg_ssim
        scheduler.step(current_metric)
        
        # 记录到TensorBoard
        writer.add_scalar("Loss/train", avg_train_loss, epoch)
        writer.add_scalar("Loss/val", avg_val_loss, epoch)
        writer.add_scalar("Metrics/PSNR", avg_psnr, epoch)
        writer.add_scalar("Metrics/SSIM", avg_ssim, epoch)
        writer.add_scalar("Learning Rate", optimizer.param_groups[0]['lr'], epoch)

        # 保存最佳模型
        if (monitor_metric == "val_loss" and current_metric < best_metric) or \
           (monitor_metric != "val_loss" and current_metric > best_metric):
            best_metric = current_metric
            epochs_without_improve = 0
            torch.save(model.state_dict(), checkpoints_dir / "best_model.pth")
        else:
            epochs_without_improve += 1
            print(f"No improvement for {epochs_without_improve} epochs")
        
        print(f"Epoch {epoch+1} | "
              f"Train Loss: {avg_train_loss:.4f} | "
              f"Val Loss: {avg_val_loss:.4f} | "
              f"PSNR: {avg_psnr:.2f} dB | "
              f"SSIM: {avg_ssim:.4f} | "
              f"LR: {optimizer.param_groups[0]['lr']:.2e}")
                
        # 早停判断
        if early_stop and epochs_without_improve >= patience:
            print(f"\nEarly stopping triggered at epoch {epoch+1}!")
            break
        
        # 保存检查点
        checkpoint = {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scaler_state": scaler.state_dict(),
            "best_metric": best_metric,
            "config": config
        }
        torch.save(checkpoint, checkpoints_dir / "latest_checkpoint.pth")
        
    writer.close()
    print("Training completed!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/train_dncnn.yaml",
                        help="Path to training configuration file")
    parser.add_argument("--name", type=str, required=True,
                        help="Experiment name for logging")
    parser.add_argument("--resume", action="store_true",
                        help="Resume training from latest checkpoint")
    args = parser.parse_args()
    
    train(args.config, args.name, args.resume)
