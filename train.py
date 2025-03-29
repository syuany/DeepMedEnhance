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

# 项目内模块
from models.dncnn import DnCNN
from models.autoencoder import MedicalAutoencoder
from models.mini_unet import MiniUNet
from models.swinir import SwinIR
from scripts.data_loader import get_dataloader

def calculate_psnr(img1, img2):
    """计算PSNR指标"""
    mse = torch.mean((img1 - img2) ** 2)
    return 10 * torch.log10(1.0 / mse)

def calculate_ssim(img1, img2, window_size=11, size_average=True):
    """计算SSIM指标（简化版）"""
    C1 = (0.01 * 1)**2
    C2 = (0.03 * 1)**2

    mu1 = F.avg_pool2d(img1, window_size, 1, 0)
    mu2 = F.avg_pool2d(img2, window_size, 1, 0)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.avg_pool2d(img1*img1, window_size, 1, 0) - mu1_sq
    sigma2_sq = F.avg_pool2d(img2*img2, window_size, 1, 0) - mu2_sq
    sigma12 = F.avg_pool2d(img1*img2, window_size, 1, 0) - mu1_mu2

    ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2)) / ((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean() if size_average else ssim_map.mean(1).mean(1).mean(1)

def train(config_path, experiment_name, resume=False):
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
    model_type = config["model"]["type"]
    model_class = {
        "DnCNN": DnCNN,
        "Autoencoder": MedicalAutoencoder,
        "MiniUNet": MiniUNet,
        'SwinIR': SwinIR
    }[model_type]
    model = model_class(**config["model"]["params"]).to(device)
    
    # 初始化优化器和损失函数
    optimizer = torch.optim.Adam(model.parameters(), lr=config["training"]["lr"])
    criterion = nn.MSELoss()

    # 早停机制初始化
    early_stop = config["training"].get("early_stop", False)
    patience = config["training"].get("patience", 10)
    monitor_metric = config["training"].get("monitor", "val_loss")  # 支持val_loss/psnr/ssim
    epochs_without_improve = 0
    best_metric = float("inf") if monitor_metric == "val_loss" else -float("inf")
    
    # 断点续训逻辑
    start_epoch = 0
    
    if resume:
        checkpoint_path = checkpoints_dir / "latest_checkpoint.pth"
        if checkpoint_path.exists():
            checkpoint = torch.load(checkpoint_path, weights_only=True)
            model.load_state_dict(checkpoint["model_state"])
            optimizer.load_state_dict(checkpoint["optimizer_state"])
            scaler.load_state_dict(checkpoint["scaler_state"])
            start_epoch = checkpoint["epoch"] + 1
            best_metric = checkpoint["best_metric"]
            print(f"Resuming training from epoch {start_epoch}")
    
    # 获取模型专用batch_size（优先使用，否则回退到数据配置）
    batch_size = config["training"].get("batch_size", None)
    
    # 获取数据加载器
    train_loader = get_dataloader(
        config_path=config["data_config"], 
        mode="train", 
        batch_size=batch_size  # 传递动态batch_size
    )
    val_loader = get_dataloader(
        config_path=config["data_config"],
        mode="test",
        batch_size=batch_size  
    )
    
    # 日志记录
    writer = SummaryWriter(exp_dir / "logs")
    
    # 训练循环
    for epoch in range(start_epoch, config["training"]["epochs"]):
        model.train()
        epoch_loss = 0.0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config['training']['epochs']}")
        
        for noisy, clean in progress_bar:
            noisy = noisy.to(device)
            clean = clean.to(device)
            
            # 混合精度前向传播
            with autocast('cuda', enabled=config["training"].get("use_amp", True)):
                outputs = model(noisy)
                loss = criterion(outputs, clean)
            
            # 混合精度反向传播
            scaler.scale(loss).backward()
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
                ssim_values.append(calculate_ssim(outputs, clean).item())
        
        # 计算平均指标
        avg_train_loss = epoch_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        avg_psnr = np.mean(psnr_values)
        avg_ssim = np.mean(ssim_values)
        
        # 记录到TensorBoard
        writer.add_scalar("Loss/train", avg_train_loss, epoch)
        writer.add_scalar("Loss/val", avg_val_loss, epoch)
        writer.add_scalar("Metrics/PSNR", avg_psnr, epoch)
        writer.add_scalar("Metrics/SSIM", avg_ssim, epoch)

        # 保存最佳模型
        current_metric = avg_val_loss if monitor_metric == "val_loss" else \
                        avg_psnr if monitor_metric == "psnr" else avg_ssim

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
              f"SSIM: {avg_ssim:.4f}")
                
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