import torch
import torch.nn as nn
import torch.nn.functional as F

class MiniUNet(nn.Module):
    def __init__(self, in_channels=1, base_channels=32):
        """
        轻量化UNet，适用于医学图像去噪
        Args:
            in_channels (int): 输入通道数（灰度图像为1）
            base_channels (int): 基础通道数（控制模型大小）
        """
        super(MiniUNet, self).__init__()
        
        # 编码器
        self.enc1 = self._block(in_channels, base_channels)      # 64x64 -> 64x64
        self.pool1 = nn.MaxPool2d(2)                             # 64x64 -> 32x32
        self.enc2 = self._block(base_channels, base_channels*2)
        self.pool2 = nn.MaxPool2d(2)                             # 32x32 -> 16x16
        self.enc3 = self._block(base_channels*2, base_channels*4)
        
        # 解码器
        self.up3 = nn.ConvTranspose2d(base_channels*4, base_channels*2, 2, stride=2)  # 16x16 ->32x32
        self.dec3 = self._block(base_channels*4, base_channels*2)  # 跳跃连接后通道数翻倍
        self.up2 = nn.ConvTranspose2d(base_channels*2, base_channels, 2, stride=2)    # 32x32->64x64
        self.dec2 = self._block(base_channels*2, base_channels)
        self.conv_out = nn.Conv2d(base_channels, in_channels, 1)  # 1x1卷积调整输出通道
        
    def _block(self, in_ch, out_ch):
        """基础卷积块：Conv -> BN -> ReLU"""
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        # 编码
        enc1 = self.enc1(x)
        enc2 = self.enc2(self.pool1(enc1))
        enc3 = self.enc3(self.pool2(enc2))
        
        # 解码 + 跳跃连接
        dec3 = self.up3(enc3)
        dec3 = torch.cat([dec3, enc2], dim=1)  # 通道维度拼接
        dec3 = self.dec3(dec3)
        
        dec2 = self.up2(dec3)
        dec2 = torch.cat([dec2, enc1], dim=1)
        dec2 = self.dec2(dec2)
        
        return self.conv_out(dec2)  # 输出与输入同尺寸

if __name__ == "__main__":
    # 测试代码
    model = MiniUNet(in_channels=1)
    x = torch.randn(2, 1, 64, 64)  # batch=2, channel=1, 64x64
    output = model(x)
    print(f"输入尺寸: {x.shape} => 输出尺寸: {output.shape}")  # 应保持相同尺寸