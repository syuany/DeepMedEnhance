import os
import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
from miniunet_improved import ImprovedMiniUNet
import argparse

class ImageEnhancer:
    def __init__(self, model_path, img_size=256, device='cuda'):
        # 根据用户选择设置设备
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.img_size = img_size
        self.model = self._load_model(model_path)
        
        # 优化配置
        cv2.setNumThreads(0)  # 禁用OpenCV多线程
        torch.set_num_threads(4)  # 设置PyTorch线程数

    def _load_model(self, model_path):
        """安全加载模型"""
        try:
            checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
            args = checkpoint.get('args', {'rcab': True, 'msf': True})
            
            model = ImprovedMiniUNet(
                use_rcab=args.get('rcab', True),
                use_msf=args.get('msf', True)
            )
            model.load_state_dict(checkpoint['model'])
            return model.to(self.device).eval()
        except Exception as e:
            raise RuntimeError(f"模型加载失败: {str(e)}")

    def _preprocess(self, img):
        """标准化预处理流程"""
        # 统一使用双三次插值
        processed = cv2.resize(img, (self.img_size, self.img_size), 
                             interpolation=cv2.INTER_CUBIC)
        
        # 类型转换和归一化
        processed = processed.astype(np.float32)
        processed = (processed - 127.5) / 127.5  # [-1, 1]范围
        
        return torch.from_numpy(processed).unsqueeze(0).unsqueeze(0).to(self.device)

    def _postprocess(self, tensor):
        """标准化后处理"""
        output = tensor.squeeze().cpu().numpy()
        output = (output + 1) * 127.5  # [0, 255]范围
        return np.clip(output, 0, 255).astype(np.uint8)

    def enhance(self, img_array):
        """带异常处理的增强流程"""
        try:
            if img_array.size == 0:
                raise ValueError("空输入图像")
                
            with torch.inference_mode():
                tensor = self._preprocess(img_array)
                output = self.model(tensor)
                return self._postprocess(output)
        except Exception as e:
            raise RuntimeError(f"增强失败: {str(e)}")

    def save_comparison(self, original, enhanced, save_path):
        """
        生成专业对比图
        :param original: 原始图像数组 (H,W)
        :param enhanced: 增强结果数组 (H,W)
        :param save_path: 保存路径
        """
        plt.figure(figsize=(12, 6), dpi=150)
        
        # 原始图像子图
        plt.subplot(1, 2, 1, title='equal')
        plt.imshow(original, cmap='gray', vmin=0, vmax=255)
        plt.title("Original Image", fontsize=10)
        plt.axis('off')
        
        # 增强结果子图
        plt.subplot(1, 2, 2, title='equal')
        plt.imshow(enhanced, cmap='gray', vmin=0, vmax=255)
        plt.title("Enhanced Result", fontsize=10)
        plt.axis('off')
        
        # 保存配置
        plt.tight_layout(pad=0.5)
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        plt.close()

    def process_directory(self, input_dir, output_dir):
        """健壮的批量处理"""
        os.makedirs(output_dir, exist_ok=True)
        
        for filename in os.listdir(input_dir):
            if not filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                continue
                
            try:
                input_path = os.path.join(input_dir, filename)
                output_path = os.path.join(output_dir, f"enhanced_{filename}")
                
                # 读取校验
                orig_img = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
                if orig_img is None:
                    raise ValueError("无法读取图像文件")
                
                # 执行增强
                enhanced = self.enhance(orig_img)
                
                # 生成对比图
                resized_orig = cv2.resize(orig_img, (self.img_size, self.img_size),
                                        interpolation=cv2.INTER_CUBIC)
                comparison = np.hstack([resized_orig, enhanced])
                
                # 保存专业对比图
                self.save_comparison(resized_orig, enhanced, output_path)
                print(f"成功保存: {output_path}")
                
            except Exception as e:
                print(f"处理 {filename} 失败: {str(e)}")

    def _processByNum(self, args, input_dir, output_dir):
        # 处理指定的图像
        for num in args.num:
            filename = f"{args.dir}-{num}.png"
            image_path = os.path.join(input_dir, filename)
            if os.path.exists(image_path):
                try:
                    # 读取图像
                    orig_img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
                    if orig_img is None:
                        raise ValueError("无法读取图像文件")
                    
                    # 执行增强
                    enhanced = enhancer.enhance(orig_img)
                    
                    # 生成对比图
                    resized_orig = cv2.resize(orig_img, (enhancer.img_size, enhancer.img_size),
                                            interpolation=cv2.INTER_CUBIC)
                    
                    # 保存对比图
                    output_path = os.path.join(output_dir, f"miniunet+rcab_enhanced_{filename}")
                    enhancer.save_comparison(resized_orig, enhanced, output_path)
                    
                    print(f"成功处理并保存: {output_path}")
                    
                except Exception as e:
                    print(f"处理 {filename} 失败: {str(e)}")
            else:
                print(f"文件 {filename} 不存在")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some images.')
    parser.add_argument('--dir', type=str, required=True, help='Directory name (e.g., COVID, Lung_Opacity)')
    parser.add_argument('--num', type=int, nargs='+', required=True, help='Image numbers to process (e.g., 1 5 9)')
    parser.add_argument('--device', type=str, default='cuda', choices=['cpu', 'cuda'], help='Device to use (default: cuda)')
    args = parser.parse_args()

    # 使用示例：
    # python enhance.py --dir COVID --num 1 2 3

    # torch.backends.quantized.engine = 'qnnpack'

    # 初始化增强器
    enhancer = ImageEnhancer("outputs/logs/rcab1_msf0/best.pth", 256, args.device)
    
    # 输入输出配置
    base_input_dir = "E:\workspace\class_101\COVID_Enhancement\COVID-19_Radiography_Dataset"
    input_dir = os.path.join(base_input_dir, args.dir, 'images')
    output_dir = "data/enhanced"
    os.makedirs(output_dir, exist_ok=True)

    enhancer._processByNum(args, input_dir, output_dir)
    
    