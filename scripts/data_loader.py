import os
import yaml
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from torchvision import transforms  

class MedicalDataset(Dataset):
    def __init__(self, data_dir, split_file, transform=None):
        """
        Args:
            data_dir: 预处理后的数据目录
            split_file: 数据划分文件路径
            transform: 数据增强
        """
        with open(split_file, encoding='utf-8') as f:
            self.file_list = [line.strip() for line in f]
        
        self.data_paths = [Path(data_dir)/fname for fname in self.file_list]
        self.transform = transform

    def __len__(self):
        return len(self.file_list)
    
    def __getitem__(self, idx):
        data = np.load(self.data_paths[idx])
        noisy = torch.from_numpy(data["noisy"])
        clean = torch.from_numpy(data["clean"])
        
        if self.transform:
            # 合并做相同变换
            combined = torch.cat([noisy, clean], dim=0)
            combined = self.transform(combined)
            noisy, clean = torch.split(combined, 1, dim=0)
            
        return noisy, clean

def get_dataloader(config_path, mode="train", batch_size=None):
    # 加载配置
    with open(config_path, encoding='utf-8') as f:
        config = yaml.safe_load(f)

    final_batch_size = batch_size if batch_size is not None \
                      else data_config["dataloader"]["batch_size"]
    
    # 数据增强配置
    if mode == "train":
        transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(10)
        ])
    else:
        transform = None
    
    # 构建Dataset
    dataset = MedicalDataset(
        data_dir=config["data_paths"]["covid_processed"],
        split_file=Path(config["split_paths"][f"covid_{mode}"]),
        transform=transform
    )
    
    # 构建DataLoader
    loader = DataLoader(
        dataset,
        batch_size=final_batch_size,
        shuffle=(mode == "train"),
        num_workers=config["dataloader"]["num_workers"]
    )
    
    return loader