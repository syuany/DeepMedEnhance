import os
import cv2
import yaml
import random
import numpy as np
from tqdm import tqdm
from pathlib import Path
import torchvision.transforms as transforms

def add_gaussian_noise(image, noise_level=25):
    """添加高斯噪声"""
    noise = np.random.normal(0, noise_level, image.shape).astype(np.float32)
    noisy = np.clip(image + noise, 0, 255).astype(np.uint8)
    return noisy

def split_dataset(raw_dir, split_dir, test_ratio=0.2):
    """划分原始数据集并生成划分文件"""
    all_files = [f.stem for f in raw_dir.glob("*.png")]
    random.shuffle(all_files)
    
    split_idx = int(len(all_files) * (1 - test_ratio))
    train_files = all_files[:split_idx]
    test_files = all_files[split_idx:]
    
    # 保存划分文件
    split_dir.mkdir(parents=True, exist_ok=True)
    with open(split_dir / "covid_train.txt", "w") as f:
        f.write("\n".join(train_files))
    with open(split_dir / "covid_test.txt", "w") as f:
        f.write("\n".join(test_files))
    
    return train_files, test_files

def generate_patches(clean_img, patch_size, stride, transform, noise_level):
    """生成带噪声的图片块"""
    patches = []
    h, w = clean_img.shape
    for y in range(0, h-patch_size+1, stride):
        for x in range(0, w-patch_size+1, stride):
            patch = clean_img[y:y+patch_size, x:x+patch_size]
            noisy_patch = add_gaussian_noise(patch, noise_level)
            yield patch, noisy_patch, (y, x)

def process_covid_ct(config_path="configs/data_config.yaml"):
    # 加载配置文件
    with open(config_path, encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # 路径配置
    raw_dir = Path(config["data_paths"]["covid_raw"])
    split_dir = Path(config["split_paths"]["root"])
    processed_dir = Path(config["data_paths"]["covid_processed"])
    patch_size = config["preprocessing"]["patch_size"]
    stride = config["preprocessing"]["stride"]
    noise_level = config["preprocessing"]["noise_level"]
    test_ratio = config["splitting"]["test_ratio"]
    
    # 创建目录
    processed_dir.mkdir(parents=True, exist_ok=True)
    
    # 划分数据集
    train_originals, test_originals = split_dataset(raw_dir, split_dir, test_ratio)
    
    # 图像预处理流水线
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    
    # 分别处理训练集和测试集
    for dataset_type, originals in [("train", train_originals), ("test", test_originals)]:
        split_list = []
        
        for img_stem in tqdm(originals, desc=f"Processing {dataset_type} set"):
            img_file = raw_dir / f"{img_stem}.png"
            clean = cv2.imread(str(img_file), cv2.IMREAD_GRAYSCALE)
            
            # 生成并保存所有块
            for patch, noisy_patch, (y, x) in generate_patches(clean, patch_size, stride, transform, noise_level):
                # 转换为Tensor
                noisy_tensor = transform(noisy_patch)
                clean_tensor = transform(patch)
                
                # 生成唯一文件名
                patch_id = f"{img_stem}_{y}_{x}"
                save_path = processed_dir / f"{patch_id}.npz"
                np.savez(save_path, noisy=noisy_tensor.numpy(), clean=clean_tensor.numpy())
                split_list.append(f"{patch_id}.npz")
        
        # 保存块级划分文件
        with open(split_dir / f"covid_{dataset_type}.txt", "w") as f:
            f.write("\n".join(split_list))

if __name__ == "__main__":
    process_covid_ct()