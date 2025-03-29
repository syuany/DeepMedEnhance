import torch
import torch.nn as nn
import torch.nn.init as init

class MedicalAutoencoder(nn.Module):
    def __init__(self, in_channels=1, base_channels=16, latent_dim=64):
        """
        轻量化自编码器（带跳跃连接）
        Args:
            in_channels (int): 输入通道数
            base_channels (int): 基础通道数（控制模型大小）
            latent_dim (int): 潜在空间特征通道数
        """
        super(MedicalAutoencoder, self).__init__()
        
        # 编码器
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, 3, stride=2, padding=1),  # 下采样1/2
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels, base_channels*2, 3, stride=2, padding=1),  # 下采样1/4
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels*2, latent_dim, 3, padding=1),  # 保持尺寸
            nn.ReLU(inplace=True)
        )
        
        # 解码器（对称结构）
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, base_channels*2, 3, stride=2, padding=1, output_padding=1),  # 上采样x2
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(base_channels*2, base_channels, 3, stride=2, padding=1, output_padding=1),  # 上采样x2
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels, in_channels, 3, padding=1)  # 最终输出层（无激活）
        )
        
        # 跳跃连接（可选）
        self.skip_conv = nn.Conv2d(base_channels*2, base_channels*2, 1)
        
        self._initialize_weights()

    def forward(self, x):
        # 编码
        x1 = self.encoder(x)
        
        # 解码（可加入跳跃连接）
        x2 = self.decoder(x1)
        
        return x2  # 输出与输入同尺寸

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    init.constant_(m.bias, 0)

if __name__ == "__main__":
    # 测试代码
    model = MedicalAutoencoder(in_channels=1)
    x = torch.randn(1, 1, 128, 128)
    output = model(x)
    print(f"输入尺寸: {x.shape}, 输出尺寸: {output.shape}")  # 应保持相同尺寸