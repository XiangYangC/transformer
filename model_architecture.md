# WindTurbineNet 模型架构

## 整体架构图

```mermaid
graph TD
    subgraph Input
        img[输入图像 640x640x3]
    end

    subgraph BackboneNetwork[特征提取主干网络]
        subgraph Layer1[第一层]
            conv1[标准卷积层 3x3, 32] --> bn1[BatchNorm]
            bn1 --> relu1[ReLU]
            relu1 --> pool1[MaxPool 2x2]
        end

        subgraph Layer2[第二层]
            dw_conv[深度可分离卷积]
            pool1 --> dw_conv
            dw_conv --> dw1[DWConv 3x3, 32组]
            dw1 --> pw1[PWConv 1x1, 64]
            pw1 --> bn2[BatchNorm]
            bn2 --> relu2[ReLU]
            relu2 --> pool2[MaxPool 2x2]
        end

        subgraph Layer3[第三层]
            conv3[Conv 3x3, 128]
            pool2 --> conv3
            conv3 --> bn3[BatchNorm]
            bn3 --> relu3[ReLU]
            relu3 --> pool3[MaxPool 2x2]
        end

        subgraph Layer4[第四层]
            conv4[Conv 3x3, 256]
            pool3 --> conv4
            conv4 --> bn4[BatchNorm]
            bn4 --> relu4[ReLU]
            relu4 --> pool4[MaxPool 2x2]
        end
    end

    subgraph AttentionModule[注意力模块]
        subgraph SpatialAttention[空间注意力]
            sa1[特征图权重计算]
            sa2[空间注意力图]
        end

        subgraph ReinforceAttention[强化学习注意力]
            ra1[Q,K,V计算]
            ra2[自适应权重]
            ra3[反馈更新]
        end
    end

    subgraph ClassificationHead[分类头]
        gap[全局平均池化]
        fc1[全连接层 256->128]
        drop[Dropout 0.5]
        fc2[全连接层 128->7]
    end

    img --> Layer1
    pool4 --> AttentionModule
    AttentionModule --> gap
    gap --> fc1
    fc1 --> drop
    drop --> fc2
    fc2 --> output[输出: 7类故障预测]

    style Input fill:#f9f,stroke:#333,stroke-width:2px
    style BackboneNetwork fill:#bbf,stroke:#333,stroke-width:2px
    style AttentionModule fill:#bfb,stroke:#333,stroke-width:2px
    style ClassificationHead fill:#fbf,stroke:#333,stroke-width:2px
```

## 特征提取主干网络

### 第一层：标准卷积层
- 输入：640×640×3
- 3×3 卷积，32通道
- BatchNorm + ReLU
- MaxPool 2×2 → 320×320×32

### 第二层：深度可分离卷积
- 3×3 深度卷积（32组）
- 1×1 点卷积（64通道）
- BatchNorm + ReLU
- MaxPool 2×2 → 160×160×64

### 第三层：特征提取
- 3×3 卷积，128通道
- BatchNorm + ReLU
- MaxPool 2×2 → 80×80×128

### 第四层：高级特征
- 3×3 卷积，256通道
- BatchNorm + ReLU
- MaxPool 2×2 → 40×40×256

## 注意力模块

### 空间注意力
```mermaid
graph LR
    input[输入特征] --> avg[平均池化]
    input --> max[最大池化]
    avg --> concat[特征融合]
    max --> concat
    concat --> conv[7×7卷积]
    conv --> sigmoid[Sigmoid]
    sigmoid --> output[注意力图]
```

### 强化学习注意力
```mermaid
graph TD
    input[输入特征] --> qkv[Q,K,V计算]
    qkv --> attn[注意力计算]
    attn --> weight[自适应权重]
    weight --> fusion[特征融合]
    fusion --> feedback[反馈更新]
    feedback --> |更新权重| weight
```

## 分类头

### 结构设计
```mermaid
graph LR
    input[特征图 40×40×256] --> gap[全局平均池化]
    gap --> fc1[FC 256->128]
    fc1 --> relu[ReLU]
    relu --> drop[Dropout 0.5]
    drop --> fc2[FC 128->7]
    fc2 --> output[7类输出]
```

## 数据流

### 特征图尺寸变化
```mermaid
graph LR
    input[640×640×3] --> l1[320×320×32]
    l1 --> l2[160×160×64]
    l2 --> l3[80×80×128]
    l3 --> l4[40×40×256]
    l4 --> attn[注意力增强]
    attn --> gap[1×1×256]
    gap --> fc[全连接层]
    fc --> output[7类预测]
```

## 训练流程

```mermaid
graph TD
    input[输入图像] --> aug[数据增强]
    aug --> backbone[特征提取]
    backbone --> attn[注意力模块]
    attn --> head[分类头]
    head --> loss[损失计算]
    loss --> backward[反向传播]
    backward --> update[参数更新]
    update --> feedback[注意力反馈]
    feedback --> attn
```

## 模型特点

1. **轻量级设计**
   - 使用深度可分离卷积减少参数量
   - 采用批归一化加速训练
   - 使用dropout防止过拟合

2. **注意力增强**
   - 集成强化学习注意力机制
   - 动态调整特征权重
   - 提高关键区域的识别能力

3. **多尺度特征**
   - 逐层降采样提取特征
   - 特征图尺寸变化：640->320->160->80->40
   - 感受野逐层增大

4. **故障分类**
   - 7类故障识别：
     - burning（燃烧）
     - crack（裂纹）
     - deformity（变形）
     - dirt（污渍）
     - oil（油污）
     - peeling（剥落）
     - rusty（锈蚀）

## 训练参数

```yaml
train:
  epochs: 100
  batch: 32
  lr0: 0.001
  momentum: 0.9
  weight_decay: 0.0001
  patience: 20
```

## 数据增强

```yaml
augment:
  enabled: true
  hsv_h: 0.015
  hsv_s: 0.7
  hsv_v: 0.4
  flipud: 0.5
  fliplr: 0.5
  rotate: 0.3
  scale: 0.5
  translate: 0.1
  blur: 0.3
  noise: 0.2
``` 