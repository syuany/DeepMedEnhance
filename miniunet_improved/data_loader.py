import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

class COVIDDataset(Dataset):
    def __init__(self, data_dir, mode='train', img_size=256):
        self.img_size = img_size
        self.image_paths = self._prepare_data(data_dir, mode)
        
    def _prepare_data(self, root_dir, mode):
        # 收集所有类别图像路径和对应标签
        # categories = ['COVID', 'Lung_Opacity', 'Normal', 'Viral Pneumonia']
        categories = ['COVID']
        all_images = []
        labels = []
        
        cnt=0
        for idx, category in enumerate(categories):
            category_dir = os.path.join(root_dir, category, 'images')
            if not os.path.exists(category_dir):
                continue  # 跳过不存在的目录
            
            # 获取当前类别所有图像路径
            for fname in os.listdir(category_dir):
                if fname.lower().endswith(('.png', '.jpg', '.jpeg')):
                    img_path = os.path.join(category_dir, fname)
                    all_images.append(img_path)
                    labels.append(idx)  # 使用数字标签
                    cnt+=1
                    if cnt==100:
                        break
                    
        # 分层划分数据集（先分测试集，再分验证集）
        train_val, test, train_val_labels, _ = train_test_split(
            all_images, labels,
            test_size=0.2,
            stratify=labels,
            random_state=42
        )
        
        train, val = train_test_split(
            train_val,
            test_size=0.1,
            stratify=train_val_labels,
            random_state=42
        )
        
        return {
            'train': train,
            'val': val,
            'test': test
        }[mode]

    def _degrade_image(self, img):
        """模拟低质量X光图像退化（保持不变）"""
        # 添加混合噪声
        noise_var = 0.05 + np.random.rand()*0.15
        img = img + np.random.normal(0, noise_var, img.shape)
        
        # 随机模糊
        if np.random.rand() > 0.5:
            ksize = np.random.choice([3,5])
            img = cv2.GaussianBlur(img, (ksize, ksize), 0)
            
        return np.clip(img, -1, 1)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        # 读取并预处理
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (self.img_size, self.img_size))
        
        # 归一化到[-1,1]
        clean_img = (img.astype(np.float32)/127.5) - 1.0  
        degraded_img = self._degrade_image(clean_img)
        
        return {
            'degraded': torch.FloatTensor(degraded_img).unsqueeze(0),
            'clean': torch.FloatTensor(clean_img).unsqueeze(0)
        }

    def __len__(self):
        return len(self.image_paths)
