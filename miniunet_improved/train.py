import argparse
import os
import yaml
import torch
import torch.nn.functional as F  
from torch.utils.tensorboard import SummaryWriter
from miniunet_improved import ImprovedMiniUNet
from data_loader import COVIDDataset  
from tqdm import tqdm
from utils import calculate_psnr, calculate_ssim, visualize_comparison


class EnhancedTrainer:
    def __init__(self, config, args):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.args = args
        self.config = config

        # 模型初始化
        self.model = ImprovedMiniUNet(
            use_rcab=args.rcab,
            use_msf=args.msf
        ).to(self.device)
        
        # 优化器
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=config['lr'])
        
        # 数据加载
        self.train_loader = torch.utils.data.DataLoader(
            COVIDDataset(config['data_dir'], 'train'),
            batch_size=config['batch_size'],
            shuffle=True,
            num_workers=4
        )
        self.val_loader = torch.utils.data.DataLoader(
            COVIDDataset(config['data_dir'], 'val'),
            batch_size=config['batch_size'],
            num_workers=2
        )
        
        # 训练状态
        self.best_psnr = 0
        self.start_epoch = 0
        self.no_improve = 0
        
        # 实验管理
        self.log_dir = self._setup_logging(config)
        self.writer = SummaryWriter(self.log_dir)
        
        # 恢复训练
        if args.resume:
            self._load_checkpoint(args.resume)

    def _setup_logging(self, config):
        log_dir = os.path.join(config['log_dir'], f"rcab{int(self.args.rcab)}_msf{int(self.args.msf)}")
        os.makedirs(log_dir, exist_ok=True)
        return log_dir

    def _load_checkpoint(self, path):
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.start_epoch = checkpoint['epoch'] + 1
        self.best_psnr = checkpoint['best_psnr']
        print(f"Resuming from epoch {self.start_epoch}")

    def _save_checkpoint(self, epoch, is_best):
        state = {
            'epoch': epoch,
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'best_psnr': self.best_psnr,
            'args': vars(self.args)
        }
        torch.save(state, os.path.join(self.log_dir, 'latest.pth'))
        if is_best:
            torch.save(state, os.path.join(self.log_dir, 'best.pth'))

    def train_epoch(self, epoch):
        self.model.train()
        total_loss = 0.0
        
        # 创建带自定义描述的进度条
        progress_bar = tqdm(
            self.train_loader,
            desc=f"Epoch [{epoch+1}/{self.config['epochs']}]",
            bar_format="{l_bar}{bar:20}{r_bar}",
            dynamic_ncols=True
        )
        
        for batch_idx, samples in enumerate(progress_bar):
            inputs = samples['degraded'].to(self.device)
            targets = samples['clean'].to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = F.l1_loss(outputs, targets)
            loss.backward()
            self.optimizer.step()
            
            # 更新进度条显示
            total_loss += loss.item()
            avg_loss = total_loss / (batch_idx + 1)
            progress_bar.set_postfix({
                'Loss': f"{avg_loss:.4f}",
                'LR': f"{self.optimizer.param_groups[0]['lr']:.2e}"
            })
        
        return avg_loss

    @torch.no_grad()
    def validate(self, epoch):
        self.model.eval()
        total_psnr = 0.0
        for batch in self.val_loader:
            inputs = batch['degraded'].to(self.device)
            targets = batch['clean'].cpu().numpy()
            
            outputs = self.model(inputs).cpu().numpy()
            total_psnr += calculate_psnr(outputs, targets)
        
        avg_psnr = total_psnr / len(self.val_loader)
        self.writer.add_scalar('Val/PSNR', avg_psnr, epoch)
        return avg_psnr

    def run(self, total_epochs):
        for epoch in range(self.start_epoch, total_epochs):
            train_loss = self.train_epoch(epoch)
            val_psnr = self.validate(epoch)
                      
            self._save_checkpoint(epoch, is_best=False)
            print(f"Epoch {epoch} | Train Loss: {train_loss:.4f} | Val PSNR: {val_psnr:.2f} dB")
            self.writer.add_scalar('Train/Loss', train_loss, epoch)

            # 早停机制
            if val_psnr > self.best_psnr:
                self.best_psnr = val_psnr
                self.no_improve = 0
                self._save_checkpoint(epoch, is_best=True)
            else:
                self.no_improve += 1
                print(f'No improvement for {self.no_improve} epoch')
                if self.no_improve >= self.config['patience']:
                    print(f"Early stopping at epoch {epoch}")
                    break
            
        self.writer.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--rcab', action='store_true', help="使用RCAB模块")
    parser.add_argument('--msf', action='store_true', help="使用多尺度融合")
    parser.add_argument('--resume', type=str, help="恢复训练的检查点路径")
    args = parser.parse_args()

    with open("configs/train_config.yaml", encoding='utf-8') as f:
        config = yaml.safe_load(f)

    trainer = EnhancedTrainer(config, args)
    trainer.run(config['epochs'])
