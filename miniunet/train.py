import sys
import os
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import yaml
from datetime import datetime

from src.data_loader import COVIDDataset
from src.miniunet import MiniUNet
from src.utils import calculate_psnr, calculate_ssim, visualize_comparison


class Trainer:
    def __init__(self, config):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = MiniUNet().to(self.device)
        self.optimizer = optim.AdamW(self.model.parameters(), lr=config['lr'], weight_decay=1e-4)
        self.criterion = nn.L1Loss()
        
        # 数据加载增强配置
        self.train_loader = DataLoader(
            COVIDDataset(config['data_dir'], 'train'),
            batch_size=config['batch_size'],
            shuffle=True,
            num_workers=4,
            pin_memory=True
        )
        self.val_loader = DataLoader(
            COVIDDataset(config['data_dir'], 'val'),
            batch_size=config['batch_size'],
            num_workers=2,
            pin_memory=True
        )
        
        # 实验跟踪
        exp_name = f"exp_{datetime.now().strftime('%m%d_%H%M')}"
        self.log_dir = os.path.join(config['log_dir'], exp_name)
        os.makedirs(self.log_dir, exist_ok=True)
        self.writer = SummaryWriter(log_dir=self.log_dir)
        
        # 训练状态
        self.best_psnr = 0
        self.current_epoch = 0

        # 新增早停相关参数
        self.early_stop_patience = config.get('patience', 5)  # 默认5个epoch
        self.no_improve_count = 0
        self.best_psnr = 0
        self.early_stop = False  # 新增早停标志

    def train_epoch(self):
        self.model.train()
        total_loss = 0
        for batch_idx, samples in enumerate(self.train_loader):
            degraded = samples['degraded'].to(self.device, non_blocking=True)
            clean = samples['clean'].to(self.device, non_blocking=True)
            
            self.optimizer.zero_grad(set_to_none=True)
            outputs = self.model(degraded)
            loss = self.criterion(outputs, clean)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)  # 梯度裁剪
            self.optimizer.step()
            
            total_loss += loss.item()
            if batch_idx % 50 == 0:
                avg_loss = total_loss / (batch_idx+1)
                print(f"Epoch: {self.current_epoch} | Batch: {batch_idx}/{len(self.train_loader)} | Loss: {avg_loss:.4f}")
                self.writer.add_scalar('Train/Loss', avg_loss, self.current_epoch*len(self.train_loader)+batch_idx)
                
        return total_loss / len(self.train_loader)

    @torch.no_grad()
    def validate(self):
        self.model.eval()
        total_psnr = 0
        total_ssim = 0
        sample_outputs = []  # 保存样例用于可视化
        
        for samples in self.val_loader:
            degraded = samples['degraded'].to(self.device, non_blocking=True)
            clean = samples['clean']
            outputs = self.model(degraded).cpu().float()
            
            # 转换到[0,255]范围
            outputs_uint = ((outputs.numpy() + 1) * 127.5).astype(np.uint8)
            clean_uint = ((clean.numpy() + 1) * 127.5).astype(np.uint8)
            
            total_psnr += calculate_psnr(outputs_uint, clean_uint)
            total_ssim += calculate_ssim(outputs_uint, clean_uint)
            
            # 保存第一个batch的样例
            if not sample_outputs:
                sample_outputs.append({
                    'degraded': degraded.cpu()[0],
                    'output': outputs[0],
                    'clean': clean[0]
                })
        
        avg_psnr = total_psnr / len(self.val_loader)
        avg_ssim = total_ssim / len(self.val_loader)
        
        print(f"[Validation] PSNR: {avg_psnr:.2f} dB | SSIM: {avg_ssim:.4f}")
        self.writer.add_scalar('Val/PSNR', avg_psnr, self.current_epoch)
        self.writer.add_scalar('Val/SSIM', avg_ssim, self.current_epoch)
        
        # 可视化样例
        sample = sample_outputs[0]
        visualize_comparison(
            sample['degraded'], sample['output'], sample['clean'],
            os.path.join(self.log_dir, f"epoch_{self.current_epoch}.png")
        )
        
        return avg_psnr

    def run(self, epochs):
        for epoch in range(epochs):
            if self.early_stop:  # 新增早停检查
                print(f"Early stopping triggered at epoch {self.current_epoch}")
                break

            self.current_epoch = epoch + 1
            train_loss = self.train_epoch()
            val_psnr = self.validate()
            
            # 保存最佳模型
            if val_psnr > self.best_psnr:
                self.best_psnr = val_psnr
                self.no_improve_count = 0  # 重置计数器
                torch.save({
                    'epoch': self.current_epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'best_psnr': self.best_psnr,
                }, os.path.join(self.log_dir, "best_model.pth"))
            else:
                self.no_improve_count += 1
                print(f"No improvement for {self.no_improve_count}/{self.early_stop_patience} epochs")

            # 触发早停条件
            if self.no_improve_count >= self.early_stop_patience:
                self.early_stop = True
                
            # 保存最新模型
            torch.save({
                'epoch': self.current_epoch,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
            }, os.path.join(self.log_dir, "latest_model.pth"))

        self.writer.close()

if __name__ == "__main__":
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # 解决模块导入问题
    
    with open("configs/train_config.yaml", encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # 创建必要目录
    os.makedirs(config['log_dir'], exist_ok=True)
    os.makedirs("outputs/visuals", exist_ok=True)
    
    trainer = Trainer(config)
    trainer.run(config['epochs'])
