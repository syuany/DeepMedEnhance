import torch
import torch.nn as nn
import torch.nn.init as init

class LightDnCNN(nn.Module):
    def __init__(self, in_channels=1, num_layers=9, num_features=32):
        """
        轻量化DnCNN 去噪模型
        Args:
            in_channels (int): 输入通道数（医学图像通常为1）
            num_layers (int): 网络层数（默认优化为9层）
            num_features (int): 中间层特征通道数（默认压缩至32）
        """
        super(LightDnCNN, self).__init__()
        layers = []
        
        # 第一层（Conv + ReLU）
        layers.append(nn.Conv2d(in_channels, num_features, kernel_size=3, padding=1))
        layers.append(nn.ReLU(inplace=True))
        
        # 中间层（Conv + BN + ReLU）
        for _ in range(num_layers - 2):
            layers.append(nn.Conv2d(num_features, num_features, kernel_size=3, padding=1))
            layers.append(nn.BatchNorm2d(num_features))
            layers.append(nn.ReLU(inplace=True))
        
        # 最后一层（Conv，无激活函数）
        layers.append(nn.Conv2d(num_features, in_channels, kernel_size=3, padding=1))
        
        self.dncnn = nn.Sequential(*layers)
        self._initialize_weights()

    def forward(self, x):
        # 残差学习：输出 = 输入 - 噪声
        return x - self.dncnn(x)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)

if __name__ == "__main__":
    # 测试轻量化模型
    model = LightDnCNN(in_channels=1)
    x = torch.randn(1, 1, 299, 299)  # 模拟COVID-19数据尺寸
    output = model(x)
    print(f"输入尺寸: {x.shape}, 输出尺寸: {output.shape}")
    print(f"总参数量: {sum(p.numel() for p in model.parameters())/1e3:.1f} K")  # 约65.5K参数
