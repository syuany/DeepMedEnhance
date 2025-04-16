import numpy as np
import matplotlib.pyplot as plt
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

def calculate_psnr(pred, target):
    """批量计算PSNR (支持单通道和多通道)"""
    if pred.ndim == 4:  # batch, channel, h, w
        return np.mean([psnr(t, p) for t, p in zip(target, pred)])
    elif pred.ndim == 3:  # channel, h, w
        return psnr(target, pred)
    else:
        raise ValueError("Unsupported dimension: {}".format(pred.ndim))
        
def calculate_ssim(pred, target):
    """批量计算SSIM (处理灰度图)"""
    ssim_values = []
    win_size = 7  # 设置为固定值
    min_dim = min(pred.shape[-2], pred.shape[-1])  # 获取H或W的最小值
    
    # 自动调整win_size为最大可用奇数
    win_size = min(win_size, min_dim)
    win_size = win_size if win_size % 2 == 1 else win_size - 1
    
    for t, p in zip(target, pred):
        # 确保输入是二维的 (H, W)
        if t.ndim == 3:
            t = t.squeeze()  # 去除通道维度
        if p.ndim == 3:
            p = p.squeeze()
            
        # 显示检查尺寸
        if t.shape[0] < win_size or t.shape[1] < win_size:
            raise ValueError(f"图像尺寸{t.shape}小于窗口大小{win_size}")
            
        ssim_val = ssim(
            t, p,
            data_range=255,
            win_size=win_size,
            channel_axis=None  # 明确指定为灰度图
        )
        ssim_values.append(ssim_val)
        
    return np.mean(ssim_values)


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

