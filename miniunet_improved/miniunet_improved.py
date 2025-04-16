# miniunet_improved.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class RCAB(nn.Module):
    """残差通道注意力模块（修复维度问题）"""
    def __init__(self, channels, reduction=8):
        super().__init__()
        self.ca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels//reduction, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels//reduction, channels, 1),
            nn.Sigmoid()
        )
        self.conv = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels)
        )

    def forward(self, x):
        return x + self.conv(x) * self.ca(x)

class MSFusion(nn.Module):
    """多尺度特征融合模块（修复通道维度）"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        # 通道调整层
        self.channel_adjust = nn.Conv2d(in_channels, out_channels, 1)
        
        # 多尺度卷积
        self.conv3x3 = nn.Conv2d(out_channels, out_channels//2, 3, padding=1)
        self.conv5x5 = nn.Conv2d(out_channels, out_channels//2, 5, padding=2)
        
        # 特征融合
        self.fusion = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x, skip):
        # 上采样并拼接特征
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
        x = torch.cat([x, skip], dim=1)
        
        # 通道调整
        x = self.channel_adjust(x)
        
        # 多尺度特征提取
        feat3 = self.conv3x3(x)
        feat5 = self.conv5x5(x)
        
        return self.fusion(torch.cat([feat3, feat5], dim=1))

class ImprovedMiniUNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, init_features=32, 
                 use_rcab=True, use_msf=True):
        super().__init__()
        features = init_features
        self.use_rcab = use_rcab
        self.use_msf = use_msf

        # Encoder
        self.enc1 = self._make_block(in_channels, features)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.enc2 = self._make_block(features, features*2)
        self.pool2 = nn.MaxPool2d(2, 2)
        
        # Bottleneck
        self.bottleneck = self._make_block(features*2, features*4)
        
        # Decoder with dimension fix
        if use_msf:
            self.upconv2 = MSFusion(features*4 + features*2, features*2)
            self.dec2 = self._make_block(features*2, features*2)
            self.upconv1 = MSFusion(features*2 + features, features)
            self.dec1 = self._make_block(features, features)
        else:
            self.upconv2 = nn.ConvTranspose2d(features*4, features*2, 2, 2)
            self.dec2 = self._make_block(features*2*2, features*2)  # concat后通道翻倍
            self.upconv1 = nn.ConvTranspose2d(features*2, features, 2, 2)
            self.dec1 = self._make_block(features*2, features)
        
        self.final = nn.Conv2d(features, out_channels, 1)

    def _make_block(self, in_ch, out_ch):
        layers = []
        if in_ch != out_ch:
            layers.append(nn.Conv2d(in_ch, out_ch, 3, padding=1))
        
        if self.use_rcab:
            layers += [RCAB(out_ch) for _ in range(2)]
        else:
            layers += [
                nn.Conv2d(out_ch, out_ch, 3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_ch, out_ch, 3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True)
            ]
        return nn.Sequential(*layers)

    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        
        # Bottleneck
        b = self.bottleneck(self.pool2(e2))
        
        # Decoder
        if self.use_msf:
            d2 = self.upconv2(b, e2)
        else:
            d2 = self.upconv2(b)
            d2 = torch.cat([d2, e2], 1)
        d2 = self.dec2(d2)
        
        if self.use_msf:
            d1 = self.upconv1(d2, e1)
        else:
            d1 = self.upconv1(d2)
            d1 = torch.cat([d1, e1], 1)
        d1 = self.dec1(d1)
        
        return torch.tanh(self.final(d1))
