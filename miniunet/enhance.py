import torch
import numpy as np
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
import os
import sys
import argparse

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

from src.miniunet import MiniUNet

class MedicalEnhancer:
    def __init__(self, model_path, device='cuda'):
        """
        医学图像增强API
        参数：
            model_path: 训练好的模型路径(.pth文件)
            device: 推理设备 (cuda/cpu)
        """
        # 设备设置
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        # 加载模型
        self.model = self._load_model(model_path)
        self.model.eval()
        
        # 图像预处理流程
        self.preprocess = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])
    
    def _load_model(self, model_path):
        """加载预训练模型"""
        try:
            model = MiniUNet()
            checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
            model.load_state_dict(checkpoint['model_state_dict'])
            return model.to(self.device)
        except Exception as e:
            raise RuntimeError(f"模型加载失败: {str(e)}")
    
    def _postprocess(self, tensor):
        """将模型输出转换为可视图像"""
        return (tensor.squeeze().cpu().numpy() * 127.5 + 127.5).astype(np.uint8)
    
    def enhance(self, image_path):
        """
        端到端增强流程
        返回：
            dict: 包含输入图像和增强结果的numpy数组
        """
        try:
            # 读取与预处理
            img = Image.open(image_path).convert('L')
            tensor = self.preprocess(img).unsqueeze(0).to(self.device)
            
            # 模型推理
            with torch.no_grad():
                enhanced_tensor = self.model(tensor)
            
            # 后处理
            return {
                'input': np.array(img.resize((256, 256))),
                'output': self._postprocess(enhanced_tensor),
                'processed_input': self._postprocess(tensor)
            }
        except Exception as e:
            raise RuntimeError(f"图像增强失败: {str(e)}")
    
    def save_comparison(self, input_img, enhanced_img, save_path):
        """
        保存对比图像
        参数：
            input_img: 原始输入图像 (numpy array)
            enhanced_img: 增强结果 (numpy array)
            save_path: 图片保存路径
        """
        plt.figure(figsize=(12, 6), dpi=150)
        
        plt.subplot(121)
        plt.imshow(input_img, cmap='gray', vmin=0, vmax=255)
        plt.title('Original Input', fontsize=10)
        plt.axis('off')
        
        plt.subplot(122)
        plt.imshow(enhanced_img, cmap='gray', vmin=0, vmax=255)
        plt.title('Enhanced Output', fontsize=10)
        plt.axis('off')
        
        plt.tight_layout()
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0.1)
        plt.close()

    def _processByNum(self, args, input_dir, output_dir):
        # 处理指定的图像
        for num in args.num:
            filename = f"{args.dir}-{num}.png"
            image_path = os.path.join(input_dir, filename)
            if os.path.exists(image_path):
                try:
                    # 执行增强
                    result = enhancer.enhance(image_path)
                    
                    # 保存对比图
                    output_path = os.path.join(output_dir, f"miniunet_enhanced_{filename}")
                    enhancer.save_comparison(
                        result['processed_input'],
                        result['output'],
                        output_path
                    )
                    
                    print(f"成功处理并保存: {output_path}")
                    
                except Exception as e:
                    print(f"处理 {filename} 失败: {str(e)}")
            else:
                print(f"文件 {filename} 不存在")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some images.')
    parser.add_argument('--dir', type=str, required=True, help='Directory name (e.g., COVID, Lung_Opacity)')
    parser.add_argument('--num', type=int, nargs='+', required=True, help='Image numbers to process (e.g., 1 5 9)')
    args = parser.parse_args()

    # 使用示例：
    # python enhance.py --dir COVID --num 1 2 3

    # 初始化增强器
    enhancer = MedicalEnhancer(
        model_path="outputs/logs/best_model.pth",
        device="cuda"
    )
    
    # 输入输出配置
    base_input_dir = "E:\workspace\class_101\COVID_Enhancement\COVID-19_Radiography_Dataset"
    input_dir = os.path.join(base_input_dir, args.dir, 'images')
    output_dir = "data/enhanced"
    os.makedirs(output_dir, exist_ok=True)
    
    enhancer._processByNum(args, input_dir, output_dir)
    
    