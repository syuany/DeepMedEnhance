import torch
import torch.nn as nn
from timm.models.swin_transformer import SwinTransformerBlock

class SwinIR(nn.Module):
    def __init__(self, 
                 in_channels=1,
                 embed_dim=64,
                 depths=[6, 6, 6],
                 num_heads=[6, 6, 6],
                 window_size=8,
                 **kwargs):
        super().__init__()
        
        # 浅层特征提取
        self.conv_first = nn.Conv2d(in_channels, embed_dim, 3, padding=1)
        
        # Swin Transformer块堆叠
        self.stages = nn.ModuleList()
        for i in range(len(depths)):
            stage = nn.Sequential(
                *[SwinTransformerBlock(
                    dim=embed_dim,
                    num_heads=num_heads[i],
                    window_size=window_size,
                    shift_size=0 if (i % 2 == 0) else window_size // 2,
                ) for _ in range(depths[i])]
            )
            self.stages.append(stage)
        
        # 重建层
        self.conv_last = nn.Conv2d(embed_dim, in_channels, 3, padding=1)

    def forward(self, x):
        x = self.conv_first(x)
        for stage in self.stages:
            x = stage(x)
        x = self.conv_last(x)
        return x