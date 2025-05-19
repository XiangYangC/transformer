# WindTurbineNet: 风力发电机组件故障检测系统

基于深度学习的风力发电机组件故障检测系统，采用轻量级网络架构和注意力机制，实现高效准确的故障分类。

## 项目特点

- 🚀 轻量级网络设计
- 🎯 专注于7类常见故障检测
- 🔍 集成强化学习注意力机制
- 📈 实时训练监控和可视化
- 💾 完整的模型保存和加载机制

## 目录结构

```
project/
├── modules/                    # 核心模块目录
│   ├── attention/             # 注意力机制模块
│   │   ├── __init__.py
│   │   ├── spatial_attention.py
│   │   └── reinforce_attention.py
│   ├── models/                # 模型定义模块
│   │   ├── __init__.py
│   │   └── yolo_transformer.py
│   ├── data/                  # 数据处理模块
│   │   ├── __init__.py
│   │   └── dataset.py
│   └── utils/                 # 工具函数模块
├── config/                    # 配置文件目录
│   └── wind-farm.yaml        # 主配置文件
├── datasets/                  # 数据集目录
│   └── Wind-farm/
│       ├── train/
│       │   ├── images/
│       │   └── labels/
│       ├── val/
│       │   ├── images/
│       │   └── labels/
│       └── test/
│           ├── images/
│           └── labels/
├── runs/                      # 训练输出目录
│   └── train/
│       └── YYYYMMDD-HHMMSS/  # 训练结果目录
│           ├── best.pt       # 最佳模型
│           ├── last.pt       # 最新模型
│           ├── results.png   # 训练曲线图
│           ├── cfg.yaml      # 配置备份
│           └── results.yaml  # 训练结果
├── train.py                  # 训练脚本
├── requirements.txt          # 项目依赖
└── README.md                 # 项目说明
```

## 模型架构

WindTurbineNet 采用轻量级设计，主要包含以下组件：

1. **特征提取主干网络**
   - 4层卷积结构
   - 使用深度可分离卷积降低参数量
   - 多尺度特征提取

2. **注意力机制**
   - 空间注意力模块
   - 强化学习注意力模块
   - 自适应特征增强

3. **分类头**
   - 全局平均池化
   - 双层全连接网络
   - 7类故障分类输出

## 故障类别

系统可识别以下7类风力发电机组件故障：

1. burning（燃烧）
2. crack（裂纹）
3. deformity（变形）
4. dirt（污渍）
5. oil（油污）
6. peeling（剥落）
7. rusty（锈蚀）

## 快速开始

### 1. 环境配置

```bash
# 创建conda环境
conda create -n transformer python=3.8
conda activate transformer

# 安装依赖
pip install -r requirements.txt
```

### 2. 数据准备

将数据集按以下结构组织：
```
datasets/Wind-farm/
├── train/
│   ├── images/    # 训练图像
│   └── labels/    # 训练标签
├── val/
│   ├── images/    # 验证图像
│   └── labels/    # 验证标签
└── test/
    ├── images/    # 测试图像
    └── labels/    # 测试标签
```

标签格式：每行一个标签，格式为 `class_id`（0-6的整数）

### 3. 配置修改

根据需要修改 `config/wind-farm.yaml` 中的参数：

```yaml
model:
  name: "WindTurbineNet"
  num_classes: 7
  input_size: 640

train:
  epochs: 100
  batch: 32
  lr0: 0.001
  momentum: 0.9
  weight_decay: 0.0001
  patience: 20
  save: true
  save_period: 5
```

### 4. 开始训练

```bash
python train.py --cfg config/wind-farm.yaml
```

### 5. 训练监控

训练过程中会在 `runs/train/` 目录下生成实时更新的：
- 训练曲线图（损失和准确率）
- 训练日志
- 模型文件（best.pt, last.pt）
- 配置和结果文件

## 训练结果

训练完成后，在 `runs/train/YYYYMMDD-HHMMSS/` 目录下可以找到：

1. **模型文件**
   - `best.pt`: 验证集性能最好的模型
   - `last.pt`: 最后一轮训练的模型
   - `epoch_N.pt`: 定期保存的检查点

2. **训练记录**
   - `results.png`: 训练和验证的损失、准确率曲线
   - `training_log.txt`: 详细的训练日志
   - `cfg.yaml`: 训练配置备份
   - `results.yaml`: 最终训练结果统计

## 注意事项

1. **数据准备**
   - 确保图像格式支持：jpg、jpeg、png、bmp
   - 标签文件需与图像同名（扩展名为.txt）
   - 标签值必须是0-6的整数

2. **训练建议**
   - 建议使用GPU进行训练
   - 根据GPU显存大小调整batch_size
   - 可通过patience参数控制早停
   - 定期保存的检查点可用于断点续训

3. **性能优化**
   - 可通过调整学习率提升训练效果
   - 数据增强参数可根据实际需求修改
   - 注意力模块的warmup_rounds参数可调

## 引用

如果您使用了本项目，请引用：

```bibtex
@software{WindTurbineNet2024,
  title = {WindTurbineNet: 风力发电机组件故障检测系统},
  year = {2024},
  description = {基于深度学习的风力发电机组件故障检测系统}
}
``` 