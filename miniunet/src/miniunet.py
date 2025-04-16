import torch
import torch.nn as nn

class MiniUNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, init_features=32):
        super().__init__()
        features = init_features
        # 编码器块1
        self.encoder1 = self._block(in_channels, features, name="enc1")
        # 最大池化层，用于下采样
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        # 编码器块2
        self.encoder2 = self._block(features, features*2, name="enc2")
        # 最大池化层，用于下采样
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # 瓶颈块
        self.bottleneck = self._block(features*2, features*4, name="bottleneck")
        
        # 上采样卷积层，用于恢复空间尺寸
        self.upconv2 = nn.ConvTranspose2d(
            features*4, features*2, kernel_size=2, stride=2
        )
        # 解码器块2
        self.decoder2 = self._block(features*4, features*2, name="dec2")
        # 上采样卷积层，用于恢复空间尺寸
        self.upconv1 = nn.ConvTranspose2d(
            features*2, features, kernel_size=2, stride=2
        )
        # 解码器块1
        self.decoder1 = self._block(features*2, features, name="dec1")
        
        # 最终卷积层，用于生成输出通道
        self.conv = nn.Conv2d(
            in_channels=features, out_channels=out_channels, kernel_size=1
        )

    def _block(self, in_channels, features, name):
        # 定义一个包含两个卷积层、批量归一化和ReLU激活的块
        return nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=features,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(num_features=features),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channels=features,
                out_channels=features,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(num_features=features),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        # 应用编码器块1
        enc1 = self.encoder1(x)
        # 应用编码器块2，并在之前进行最大池化
        enc2 = self.encoder2(self.pool1(enc1))
        
        # 应用瓶颈块，并在之前进行最大池化
        bottleneck = self.bottleneck(self.pool2(enc2))
        
        # 上采样并连接编码器2的输出
        dec2 = self.upconv2(bottleneck)
        dec2 = torch.cat((dec2, enc2), dim=1)
        # 应用解码器块2
        dec2 = self.decoder2(dec2)
        # 上采样并连接编码器1的输出
        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        # 应用解码器块1
        dec1 = self.decoder1(dec1)
        # 应用最终卷积层并使用tanh激活函数
        return torch.tanh(self.conv(dec1))