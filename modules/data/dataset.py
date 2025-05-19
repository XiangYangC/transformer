import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import albumentations as A
from albumentations.pytorch import ToTensorV2

class WindFarmDataset(Dataset):
    def __init__(self, root_dir, transform=None, image_size=640, is_training=True):
        """
        Wind-farm数据集加载器
        Args:
            root_dir (str): 数据集根目录（包含images和labels子目录）
            transform (callable, optional): 数据增强转换
            image_size (int): 图像大小
            is_training (bool): 是否为训练模式
        """
        self.root_dir = Path(root_dir)
        self.image_size = image_size
        self.is_training = is_training
        
        # 获取图像和标签文件路径
        self.images_dir = self.root_dir
        self.labels_dir = self.root_dir.parent / 'labels'  # 标签目录与图像目录平级
        
        # 支持多种图像格式
        image_formats = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
        self.image_files = []
        for fmt in image_formats:
            self.image_files.extend(list(self.images_dir.glob(fmt)))
        self.image_files = sorted(self.image_files)
        
        # 默认的数据增强
        if transform is None:
            self.transform = A.Compose([
                A.Resize(image_size, image_size),
                A.Normalize(mean=[0.485, 0.456, 0.406], 
                          std=[0.229, 0.224, 0.225]),
                ToTensorV2()
            ])
        else:
            self.transform = transform
            
    def __len__(self):
        return len(self.image_files)
        
    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        label_path = self.labels_dir / f"{img_path.stem}.txt"
        
        # 读取图像
        image = cv2.imread(str(img_path))
        if image is None:
            raise ValueError(f"无法读取图像: {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # 读取标签 - 只获取类别信息
        class_id = 0  # 默认类别
        if label_path.exists():
            with open(label_path, 'r') as f:
                line = f.readline().strip()
                if line:
                    class_id = int(float(line.split()[0]))
        
        # 应用数据增强
        if self.transform:
            transformed = self.transform(image=image)
            image = transformed['image']
            
        return image, class_id

def create_dataloader(path, image_size=640, batch_size=16, augment=True, workers=4):
    """
    创建数据加载器
    Args:
        path (str): 数据集路径
        image_size (int): 图像大小
        batch_size (int): 批次大小
        augment (bool): 是否使用数据增强
        workers (int): 数据加载线程数
    """
    if augment:
        transform = A.Compose([
            A.RandomResizedCrop(image_size, image_size, scale=(0.8, 1.0)),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            A.OneOf([
                A.MotionBlur(p=0.2),
                A.MedianBlur(blur_limit=3, p=0.1),
                A.Blur(blur_limit=3, p=0.1),
            ], p=0.2),
            A.OneOf([
                A.OpticalDistortion(p=0.3),
                A.GridDistortion(p=0.1),
            ], p=0.2),
            A.Normalize(mean=[0.485, 0.456, 0.406], 
                       std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])
    else:
        transform = A.Compose([
            A.Resize(image_size, image_size),
            A.Normalize(mean=[0.485, 0.456, 0.406], 
                       std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])
        
    dataset = WindFarmDataset(
        root_dir=path,
        transform=transform,
        image_size=image_size
    )
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True if augment else False,
        num_workers=workers,
        pin_memory=True
    ) 