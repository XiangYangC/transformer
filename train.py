from modules.models import WindTurbineNet
from modules.data.dataset import create_dataloader
import torch
import torch.optim as optim
import yaml
import argparse
from pathlib import Path
from tqdm import tqdm
import os
import matplotlib.pyplot as plt
from datetime import datetime
import shutil
import math

class Trainer:
    def __init__(self, cfg=None):
        """初始化训练器
        
        Args:
            cfg (dict, optional): 配置字典
        """
        self.cfg = cfg or {}
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        if torch.cuda.is_available():
            print(f"CUDA Device: {torch.cuda.get_device_name(0)}")
        
        # 创建运行目录
        self.run_dir = self._create_run_dir()
        
        # 保存配置
        if cfg:
            with open(self.run_dir / 'config.yaml', 'w') as f:
                yaml.dump(cfg, f)
        
        # 初始化模型
        self.model = None
        self.load_model()
        
        # 训练状态
        self.train_losses = []
        self.train_accs = []
        self.val_losses = []
        self.val_accs = []
        self.best_val_loss = float('inf')
        
        # 初始化混合精度训练
        self.scaler = torch.cuda.amp.GradScaler()
        
    def load_model(self, model_path=None):
        """加载或创建模型"""
        if model_path and Path(model_path).exists():
            print(f"Loading model from {model_path}")
            self.model = torch.load(model_path)
        else:
            print("Creating new WindTurbineNet model...")
            self.model = WindTurbineNet(
                num_classes=self.cfg['model']['num_classes'],
                input_size=self.cfg['model']['input_size']
            )
        self.model.to(self.device)
        
    def plot_metrics(self):
        """绘制训练指标图"""
        plt.figure(figsize=(12, 4))
        
        # 绘制损失曲线
        plt.subplot(1, 2, 1)
        plt.plot(self.train_losses, label='train_loss')
        plt.plot(self.val_losses, label='val_loss')
        plt.title('Loss Curves')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        
        # 绘制准确率曲线
        plt.subplot(1, 2, 2)
        plt.plot(self.train_accs, label='train_acc')
        plt.plot(self.val_accs, label='val_acc')
        plt.title('Accuracy Curves')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy (%)')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(self.run_dir / 'results.png')
        plt.close()
        
    def save_model(self, name):
        """保存模型"""
        save_path = self.run_dir / name
        torch.save(self.model, save_path)
        print(f"Model saved to {save_path}")
        
    def evaluate_metrics(self, dataloader):
        """评估每个类别的精度和召回率
        
        Args:
            dataloader: 数据加载器
            
        Returns:
            dict: 包含每个类别指标的字典
        """
        self.model.eval()
        num_classes = self.cfg['model']['num_classes']
        
        # 初始化混淆矩阵
        confusion_matrix = torch.zeros(num_classes, num_classes)
        
        with torch.no_grad():
            for images, targets in tqdm(dataloader, desc="评估指标"):
                images = images.to(self.device)
                targets = targets.to(self.device)
                
                outputs = self.model(images)
                _, predicted = torch.max(outputs, 1)
                
                # 更新混淆矩阵
                for t, p in zip(targets.view(-1), predicted.view(-1)):
                    confusion_matrix[t.long(), p.long()] += 1
        
        # 计算每个类别的指标
        metrics = {}
        for i in range(num_classes):
            # 真阳性(TP)：对角线上的值
            tp = confusion_matrix[i, i]
            # 假阳性(FP)：该列的其他值之和
            fp = confusion_matrix[:, i].sum() - tp
            # 假阴性(FN)：该行的其他值之和
            fn = confusion_matrix[i, :].sum() - tp
            
            # 计算精度
            precision = tp / (tp + fp) if tp + fp > 0 else torch.tensor(0.0)
            # 计算召回率
            recall = tp / (tp + fn) if tp + fn > 0 else torch.tensor(0.0)
            # 计算F1分数
            f1 = 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else torch.tensor(0.0)
            
            metrics[f'class_{i}'] = {
                'precision': float(precision),
                'recall': float(recall),
                'f1': float(f1),
                'support': int(confusion_matrix[i, :].sum().item())
            }
        
        # 计算总体指标
        metrics['overall'] = {
            'accuracy': float(confusion_matrix.diag().sum() / confusion_matrix.sum()),
            'macro_precision': float(sum(m['precision'] for m in metrics.values()) / num_classes),
            'macro_recall': float(sum(m['recall'] for m in metrics.values()) / num_classes),
            'macro_f1': float(sum(m['f1'] for m in metrics.values()) / num_classes)
        }
        
        # 打印结果
        print("\n评估指标:")
        print("-" * 50)
        print(f"总体准确率: {metrics['overall']['accuracy']:.4f}")
        print(f"宏平均精度: {metrics['overall']['macro_precision']:.4f}")
        print(f"宏平均召回率: {metrics['overall']['macro_recall']:.4f}")
        print(f"宏平均F1: {metrics['overall']['macro_f1']:.4f}")
        print("\n各类别指标:")
        print("-" * 50)
        print("类别\t精度\t召回率\tF1\t支持度")
        for i in range(num_classes):
            m = metrics[f'class_{i}']
            print(f"{i}\t{m['precision']:.4f}\t{m['recall']:.4f}\t{m['f1']:.4f}\t{int(m['support'])}")
        
        # 保存混淆矩阵图
        plt.figure(figsize=(10, 8))
        plt.imshow(confusion_matrix.cpu().numpy(), cmap='Blues')
        plt.colorbar()
        plt.xlabel('预测类别')
        plt.ylabel('真实类别')
        plt.title('混淆矩阵')
        
        # 添加数值标注
        for i in range(num_classes):
            for j in range(num_classes):
                plt.text(j, i, f'{int(confusion_matrix[i, j])}',
                        ha='center', va='center')
        
        plt.savefig(self.run_dir / 'confusion_matrix.png')
        plt.close()
        
        return metrics
        
    def train(self):
        """训练模型的主函数"""
        # 训练参数
        train_cfg = self.cfg['train']
        epochs = train_cfg['epochs']
        batch_size = train_cfg['batch']
        lr = train_cfg['lr0']
        
        print("\nTraining parameters:")
        print(f"Epochs: {epochs}")
        print(f"Batch size: {batch_size}")
        print(f"Learning rate: {lr}")
        print(f"Device: {self.device}")
        print(f"Results will be saved to: {self.run_dir}")
        
        # 创建数据加载器
        print("Creating data loaders...")
        train_loader = create_dataloader(
            path=self.cfg['data']['train'],
            image_size=self.cfg['model']['input_size'],
            batch_size=batch_size,
            augment=self.cfg['augment']['enabled'],
            workers=0
        )
        
        val_loader = create_dataloader(
            path=self.cfg['data']['val'],
            image_size=self.cfg['model']['input_size'],
            batch_size=batch_size,
            augment=False,
            workers=0
        )
        
        print(f"Train dataset size: {len(train_loader.dataset)}")
        print(f"Val dataset size: {len(val_loader.dataset)}\n")
        
        # 优化器设置
        optimizer = optim.SGD(
            self.model.parameters(),
            lr=lr,
            momentum=train_cfg['momentum'],
            weight_decay=train_cfg['weight_decay']
        )
        
        # 学习率调度器
        warmup_epochs = 3  # 预热轮数
        warmup_steps = warmup_epochs * len(train_loader)
        total_steps = epochs * len(train_loader)
        
        def lr_lambda(step):
            if step < warmup_steps:
                # 线性预热
                return float(step) / float(max(1, warmup_steps))
            else:
                # 余弦退火
                progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
                return 0.5 * (1.0 + math.cos(math.pi * progress))
                
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        
        # 训练循环
        print("Starting training...\n")
        no_improve_epochs = 0
        
        for epoch in range(epochs):
            # 训练阶段
            train_loss, train_acc = self._train_epoch(train_loader, optimizer, scheduler)
            self.train_losses.append(train_loss)
            self.train_accs.append(train_acc)
            
            # 验证阶段
            val_loss, val_acc = self._validate(val_loader)
            self.val_losses.append(val_loss)
            self.val_accs.append(val_acc)
            
            # 保存最佳模型
            if val_loss < self.best_val_loss and train_cfg['save']:
                self.best_val_loss = val_loss
                self.save_model('best.pt')
                print(f"\nNew best model saved! Val loss: {val_loss:.4f}")
                no_improve_epochs = 0
            else:
                no_improve_epochs += 1
            
            # 保存最新模型
            if train_cfg['save']:
                self.save_model('last.pt')
            
            # 定期保存检查点
            if train_cfg['save'] and (epoch + 1) % train_cfg['save_period'] == 0:
                self.save_model(f'epoch_{epoch+1}.pt')
            
            # 更新训练曲线
            self.plot_metrics()
            
            # 打印当前学习率
            current_lr = scheduler.get_last_lr()[0]
            print(f'Epoch {epoch+1}: train_loss={train_loss:.4f}, train_acc={train_acc:.2f}%, '
                  f'val_loss={val_loss:.4f}, val_acc={val_acc:.2f}%, lr={current_lr:.6f}')
            
            # 保存训练日志
            with open(self.run_dir / 'training_log.txt', 'a') as f:
                f.write(f'Epoch {epoch+1}: train_loss={train_loss:.4f}, train_acc={train_acc:.2f}%, '
                       f'val_loss={val_loss:.4f}, val_acc={val_acc:.2f}%, lr={current_lr:.6f}\n')
            
            # 早停
            if no_improve_epochs >= train_cfg['patience']:
                print(f"\nEarly stopping after {no_improve_epochs} epochs without improvement")
                break
        
        # 在训练结束后评估模型
        print("\n开始最终评估...")
        val_metrics = self.evaluate_metrics(val_loader)
        
        # 保存评估结果
        with open(self.run_dir / 'evaluation_results.yaml', 'w', encoding='utf-8') as f:
            yaml.dump(val_metrics, f, allow_unicode=True)
            
        print("\n评估结果已保存到:", self.run_dir / 'evaluation_results.yaml')
        
        # 保存最终结果
        results = {
            'best_val_loss': self.best_val_loss,
            'final_train_loss': train_loss,
            'final_train_acc': train_acc,
            'final_val_loss': val_loss,
            'final_val_acc': val_acc,
            'total_epochs': epoch + 1
        }
        
        with open(self.run_dir / 'results.yaml', 'w') as f:
            yaml.dump(results, f)
            
        print(f"\nTraining completed. Results saved to {self.run_dir}")
    
    def _train_epoch(self, dataloader, optimizer, scheduler):
        """训练一个epoch"""
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        pbar = tqdm(dataloader, desc='Training')
        for i, (images, targets) in enumerate(pbar):
            images = images.to(self.device)
            targets = targets.to(self.device)
            
            # 使用混合精度训练
            with torch.cuda.amp.autocast():
                # 前向传播
                outputs = self.model(images)
                loss = self.model.compute_loss(outputs, targets)
            
            # 反向传播
            optimizer.zero_grad()
            self.scaler.scale(loss).backward()
            
            # 梯度裁剪
            self.scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            # 优化器步进
            self.scaler.step(optimizer)
            self.scaler.update()
            scheduler.step()
            
            # 统计
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            # 更新进度条
            pbar.set_postfix({
                'loss': loss.item(),
                'acc': 100.*correct/total,
                'lr': scheduler.get_last_lr()[0]
            })
        
        return total_loss / len(dataloader), 100.*correct/total
    
    def _validate(self, dataloader):
        """验证模型"""
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for images, targets in tqdm(dataloader, desc='Validating'):
                images = images.to(self.device)
                targets = targets.to(self.device)
                
                # 使用混合精度
                with torch.cuda.amp.autocast():
                    outputs = self.model(images)
                    loss = self.model.compute_loss(outputs, targets)
                
                total_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
        
        return total_loss / len(dataloader), 100.*correct/total

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='config/wind-farm.yaml', help='配置文件路径')
    parser.add_argument('--device', type=str, help='训练设备 (e.g., "cpu", "cuda:0", "auto")')
    return parser.parse_args()

def main():
    # 解析命令行参数
    args = parse_args()
    
    # 加载配置
    cfg_path = args.cfg
    if os.path.exists(cfg_path):
        print(f"Loading configuration from {cfg_path}")
        with open(cfg_path, 'r', encoding='utf-8') as f:
            cfg = yaml.safe_load(f)
    else:
        raise FileNotFoundError(f"Configuration file {cfg_path} not found!")
    
    # 更新设备设置
    if args.device:
        cfg['train']['device'] = args.device
    
    # 初始化训练器并开始训练
    trainer = Trainer(cfg)
    trainer.load_model()
    trainer.train()

if __name__ == '__main__':
    main()
