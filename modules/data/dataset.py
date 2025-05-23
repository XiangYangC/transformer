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
        self.images_dir = self.root_dir / 'images' # Assuming images are in a subdirectory named 'images'
        self.labels_dir = self.root_dir / 'labels' # Assuming labels are in a subdirectory named 'labels'
        
        # 支持多种图像格式
        image_formats = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
        self.image_files = []
        for fmt in image_formats:
            self.image_files.extend(list(self.images_dir.glob(fmt)))
        self.image_files = sorted(self.image_files)
        
        # 默认的数据增强 (for object detection)
        if transform is None:
            # Define a default transform suitable for object detection with bounding boxes
            # This is a basic example, more advanced transforms can be added.
            bbox_params = A.BboxParams(format='yolo', label_fields=['class_labels'])
            
            self.transform = A.Compose([
                A.Resize(image_size, image_size),
                A.Normalize(mean=[0.485, 0.456, 0.406], 
                          std=[0.229, 0.224, 0.225]),
                ToTensorV2()
            ], bbox_params=bbox_params)
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
        
        # 读取标签 - 边界框和类别信息
        boxes = [] # List of [x_center, y_center, width, height] (normalized)
        class_labels = [] # List of class ids
        
        if label_path.exists():
            with open(label_path, 'r') as f:
                for line in f.readlines():
                    line = line.strip().split()
                    if len(line) >= 5:
                        class_id = int(float(line[0]))
                        # YOLO format: class_id center_x center_y width height (normalized)
                        bbox = [float(line[1]), float(line[2]), float(line[3]), float(line[4])]
                        boxes.append(bbox)
                        class_labels.append(class_id)
        
        # 应用数据增强 (需要同时应用到图像和边界框)
        if self.transform:
            # Albumentations expects boxes in absolute coordinates or normalized with specified format
            # If using 'yolo' format with Albumentations, it expects normalized coordinates (0-1)
            # Our loaded boxes are already normalized YOLO format, so we can pass them directly.
            transformed = self.transform(image=image, bboxes=boxes, class_labels=class_labels)
            image = transformed['image']
            boxes = transformed['bboxes']
            class_labels = transformed['class_labels']
            
        # Convert boxes and class_labels to tensors
        # The format of the output targets (boxes and labels) depends on the model's expectation.
        # For DETR-like models, targets are usually a list of dictionaries or similar structure per image.
        # Here, we'll return a simple tensor format [num_objects, 5] = [class_id, x_c, y_c, w, h]
        # Note: This format might need adjustment based on the specific Transformer/Detection Head design.
        if len(boxes) > 0:
            targets = torch.tensor(boxes, dtype=torch.float32)
            # Add class labels as the first column
            targets = torch.cat((torch.tensor(class_labels, dtype=torch.float32).unsqueeze(1), targets), dim=1)
        else:
            # Handle images with no objects
            targets = torch.empty(0, 5, dtype=torch.float32) # Empty tensor with shape [0, 5]

        # Return image and targets (boxes and labels)
        return image, targets

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
    
    # Define transforms for object detection
    bbox_params = A.BboxParams(format='yolo', label_fields=['class_labels'])
    
    if augment:
        transform = A.Compose([
            A.RandomResizedCrop(image_size, image_size, scale=(0.8, 1.0), p=0.5),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, p=0.5),
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
        ], bbox_params=bbox_params)
    else:
        transform = A.Compose([
            A.Resize(image_size, image_size),
            A.Normalize(mean=[0.485, 0.456, 0.406], 
                       std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ], bbox_params=bbox_params)
        
    dataset = WindFarmDataset(
        root_dir=path,
        transform=transform,
        image_size=image_size,
        is_training=augment # Assuming augment=True for training, False for validation
    )
    
    # Custom collate_fn to handle variable number of objects per image
    def collate_fn(batch):
        images, targets = zip(*batch) # Unzip the batch
        images = torch.stack(images, 0) # Stack images into a tensor
        
        # Process targets: need to handle variable number of boxes
        # Pad targets or use a list of tensors
        # For simplicity, let's return a list of target tensors (one per image)
        # depending on the model/loss function, this might need adjustment
        
        # Example: create a list of dictionaries, each containing 'boxes' and 'labels' for an image
        processed_targets = []
        for i, target in enumerate(targets):
            if target.size(0) > 0:
                # Assuming target is [num_objects, 5] = [class_id, x_c, y_c, w, h]
                # Need to convert from YOLO format (x_c, y_c, w, h normalized) to x_min, y_min, x_max, y_max absolute
                # This conversion depends on the image size after resizing (image_size, image_size)
                image_h, image_w = image_size, image_size # Assuming resized to image_size x image_size
                boxes = target[:, 1:] # get bbox coordinates
                class_labels = target[:, 0].long() # get class labels

                # Convert YOLO normalized to x_min, y_min, x_max, y_max absolute
                boxes[:, 0] = boxes[:, 0] * image_w # center_x_abs
                boxes[:, 1] = boxes[:, 1] * image_h # center_y_abs
                boxes[:, 2] = boxes[:, 2] * image_w # width_abs
                boxes[:, 3] = boxes[:, 3] * image_h # height_abs

                x_min = boxes[:, 0] - boxes[:, 2] / 2
                y_min = boxes[:, 1] - boxes[:, 3] / 2
                x_max = boxes[:, 0] + boxes[:, 2] / 2
                y_max = boxes[:, 1] + boxes[:, 3] / 2
                
                absolute_boxes = torch.stack([x_min, y_min, x_max, y_max], dim=1)
                
                processed_targets.append({'boxes': absolute_boxes, 'labels': class_labels})
            else:
                # Handle images with no objects
                 processed_targets.append({'boxes': torch.empty(0, 4, dtype=torch.float32), 'labels': torch.empty(0, dtype=torch.long)})


        return images, processed_targets
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True if augment else False,
        num_workers=workers,
        pin_memory=True,
        collate_fn=collate_fn # Use custom collate function
    ) 