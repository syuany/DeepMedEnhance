import numpy as np
import matplotlib.pyplot as plt
from skimage.metrics import peak_signal_noise_ratio as psnr
from torchmetrics.image import StructuralSimilarityIndexMeasure


def calculate_psnr(pred, target):
    """批量计算PSNR (支持单通道和多通道)"""
    if pred.ndim == 4:  # batch, channel, h, w
        return np.mean([psnr(t, p) for t, p in zip(target, pred)])
    elif pred.ndim == 3:  # channel, h, w
        return psnr(target, pred)
    else:
        raise ValueError("Unsupported dimension: {}".format(pred.ndim))
        
def calculate_ssim(pred, target):
    """增强的SSIM计算"""
    ssim_metric = StructuralSimilarityIndexMeasure(
        data_range=1.0,
        kernel_size=11,  # 与skimage默认参数对齐
        k1=0.01,         # 默认参数
        k2=0.03,
        sigma=1.5        # 高斯核参数
    ).to(pred.device)
    return ssim_metric(pred, target)

def visualize_comparison(degraded, output, clean, save_path, dpi=150):
    """增强可视化效果（修复保存参数）"""
    plt.figure(figsize=(15, 5), dpi=dpi)
    
    images = [
        (degraded.numpy().squeeze(), "Degraded Input"),
        (output.numpy().squeeze(), "Enhanced Output"),
        (clean.numpy().squeeze(), "Ground Truth")
    ]
    
    for i, (img, title) in enumerate(images):
        plt.subplot(1, 3, i+1)
        plt.imshow(img, cmap='gray', vmin=-1, vmax=1)
        plt.title(title, fontsize=10)
        plt.axis('off')
    
    plt.tight_layout()
    # 修改保存参数
    plt.savefig(
        save_path,
        bbox_inches='tight',  # 使用通用参数
        pad_inches=0.1,       # 添加适当边距
        dpi=dpi
    )
    plt.close()

