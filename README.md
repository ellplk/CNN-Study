# CNN-Study

一个用于学习和实践经典卷积神经网络（CNN）及多模态视觉-语言模型的 PyTorch 代码仓库。

## 项目概述

本仓库涵盖了从经典 CNN 架构（AlexNet、VGGNet、ResNet）到现代 Transformer 视觉模型（ViT）以及多模态图文检索系统（BLIP、CLIP + FAISS）的完整学习路径，所有模型均在 **CIFAR-10** 数据集上进行训练与验证。

---

## 目录结构

```
CNN-Study/
├── cnn-net/                      # 经典 CNN 与 Transformer 视觉模型实现
│   ├── AlexNet.py                # AlexNet（含特征图可视化 & Grad-CAM）
│   ├── VggNet.py                 # VGGNet（VGG11/13/16/19，含特征图保存）
│   ├── ResNet.py                 # ResNet-50 微调
│   └── VitNet.py                 # Vision Transformer（ViT-B/16，RTX 4060 优化版）
├── image-text-retrieval/         # 多模态图文检索
│   ├── blip_model.py             # BLIP 模型从零实现（视觉编码器 + 文本编解码器）
│   ├── retrieval_system.py       # 基于 CLIP + FAISS 的图文检索系统
│   ├── blip_from_scratch.ipynb   # BLIP 从零构建演示笔记本
│   ├── blip_manual.ipynb         # BLIP 手动实现笔记本
│   ├── blip_demo.ipynb           # BLIP 功能演示笔记本
│   ├── clip_faiss_retrieval.ipynb # CLIP + FAISS 检索演示笔记本
│   └── requirements.txt          # 图文检索模块依赖
└── Test/
    ├── gpu-select-test.py        # GPU 检测与选择测试脚本
    └── gpu-monitor-test.py       # GPU 显存监控测试脚本
```

---

## 模块详解

### 1. `cnn-net/` — 经典 CNN 与视觉 Transformer

#### AlexNet（`AlexNet.py`）

- **模型结构**：4 个卷积层 + 2 个全连接层，适配 CIFAR-10（32×32 输入）。
- **训练配置**：Adam 优化器，学习率 0.001，Batch Size 128，训练 20 轮。
- **可视化功能**：
  - 每 5 个 epoch 自动保存各卷积层的**特征热力图**（`feature_maps_sample_*.png`）。
  - 实现 **Grad-CAM** 梯度加权类激活图，高亮模型关注区域。
- **TensorBoard 支持**：记录训练/测试的 Loss、Accuracy 及学习率，可通过 `tensorboard --logdir=tensorboard_logs/alexnet_cifar10` 查看。
- **设备支持**：优先使用 Apple MPS（Mac），否则退回 CPU。

#### VGGNet（`VggNet.py`）

- **模型结构**：支持 VGG11 / VGG13 / **VGG16**（默认）/ VGG19 四种配置，通过 `vgg_type` 参数切换。
- **训练配置**：Adam 优化器（weight_decay=1e-4），学习率 0.0001，Batch Size 64，训练 30 轮，含 StepLR 学习率衰减。
- **可视化功能**：
  - 每 10 个 epoch 将所有卷积层的**多通道特征图**保存为 PNG 文件（含统计信息 txt）。
  - 生成 **Grad-CAM** 热力图并叠加到原图上。
- **TensorBoard 支持**：同 AlexNet，日志目录为 `tensorboard_logs/vggnet_cifar10`。
- **最佳模型保存**：自动保存测试集准确率最高的模型权重（`vggnet_best.pth`）。

#### ResNet-50（`ResNet.py`）

- **模型结构**：基于 `torchvision` 预训练 ResNet-50 进行微调：
  - 第一层卷积替换为 3×3（适配 32×32 小图）。
  - 移除 MaxPool 层（防止特征图过小）。
  - 分类头替换为 10 类输出。
- **训练配置**：Adam 优化器，学习率 0.001，Batch Size 128，训练 20 轮。
- **设备支持**：固定使用 Apple MPS（可按需修改为 `cuda` 或 `cpu`）。

#### Vision Transformer（`VitNet.py`）

- **模型结构**：基于预训练 `ViT-B/16`（ImageNet1K 权重）微调，分类头替换为 10 类输出，前向传播自动将输入上采样至 224×224。
- **RTX 4060 8GB 专项优化**：
  - 混合精度训练（`torch.amp.GradScaler`）
  - 梯度裁剪（max_norm=1.0）
  - Warmup + Cosine 学习率调度
  - AdamW 优化器 + 标签平滑（label_smoothing=0.1）
  - 多进程数据加载（`pin_memory=True`、`persistent_workers=True`）
- **训练配置**：AdamW，学习率 3e-4，Batch Size 64，训练 25 轮。
- **GPU 管理**：自动检测并选择 RTX 4060，定期清空显存缓存。

---

### 2. `image-text-retrieval/` — 多模态图文检索

#### BLIP 从零实现（`blip_model.py`）

完整手动实现 [BLIP](https://arxiv.org/abs/2201.12086)（Bootstrapping Language-Image Pre-training）模型，不依赖第三方预训练库：

| 子模块 | 说明 |
|---|---|
| `PatchEmbedding` | 将图像切分为 16×16 的 Patch 并线性投影 |
| `MultiHeadAttention` | 多头自注意力（支持自注意力与交叉注意力） |
| `TransformerBlock` | 标准 Transformer 编码器块（Pre-LN） |
| `VisionEncoder` | ViT 风格视觉编码器，输出 CLS token 与 Patch 特征 |
| `TextEmbedding` | Token + 位置嵌入 |
| `TextEncoder` | BERT 风格文本编码器 |
| `CrossAttentionBlock` | 含交叉注意力的 Transformer 解码器块 |
| `TextDecoder` | 自回归文本解码器，用于图像描述生成 |
| `BLIPModel` | 整合上述模块，支持：图像编码、文本编码、图文相似度计算、图文匹配（ITM）、图像描述生成、视觉问答（VQA） |

提供 `create_blip_base()`（768 维，12 层）和 `create_blip_small()`（512 维，8 层）两种预设配置。

#### CLIP + FAISS 图文检索系统（`retrieval_system.py`）

基于 OpenAI CLIP 与 Facebook FAISS 构建的向量检索系统：

- **检索方向**：文本→图片、图片→文本、图片→图片（以图搜图）。
- **FAISS 索引**：支持 `flat`（精确搜索）、`ivf`（倒排文件，适合大规模）、`hnsw`（图索引，高速近似）三种类型。
- **数据集支持**：内置 COCO 数据集下载器与示例随机数据集生成器。
- **索引持久化**：支持将图像/文本索引及元数据保存/加载到本地。
- **可视化**：`visualize_results()` 方法将检索结果以 matplotlib 图表展示。

---

### 3. `Test/` — 工具测试脚本

| 文件 | 说明 |
|---|---|
| `gpu-select-test.py` | 枚举所有 CUDA GPU，打印名称、显存、计算能力，并执行矩阵乘法验证 GPU 可用性 |
| `gpu-monitor-test.py` | GPU 显存监控脚本（待完善） |

---

## 环境依赖

### CNN 模型（`cnn-net/`）

```bash
pip install torch torchvision tensorboard matplotlib numpy opencv-python pillow
```

### 图文检索（`image-text-retrieval/`）

```bash
pip install -r image-text-retrieval/requirements.txt
```

主要依赖：`torch`、`torchvision`、`faiss-cpu`、`openai-clip`、`pycocotools`、`Pillow`、`matplotlib`、`tqdm`

---

## 快速开始

### 训练 AlexNet

```bash
cd cnn-net
python AlexNet.py
```

### 训练 VGGNet（默认 VGG16）

```bash
cd cnn-net
python VggNet.py
```

### 训练 ResNet-50

```bash
cd cnn-net
python ResNet.py
```

### 训练 Vision Transformer（需要 CUDA GPU）

```bash
cd cnn-net
python VitNet.py
```

### 运行图文检索演示（示例数据）

```bash
cd image-text-retrieval
python retrieval_system.py --mode sample
```

### 运行图文检索演示（COCO 数据集）

```bash
cd image-text-retrieval
python retrieval_system.py --mode coco --device cuda
```

### 查看 TensorBoard 训练曲线

```bash
tensorboard --logdir=cnn-net/tensorboard_logs
```

---

## 模型性能参考

| 模型 | 数据集 | 参数量 | 备注 |
|---|---|---|---|
| AlexNet | CIFAR-10 | ~6M | 自定义适配版 |
| VGG16 | CIFAR-10 | ~138M | 全局平均池化适配 32×32 |
| ResNet-50 | CIFAR-10 | ~25M | ImageNet 预训练微调 |
| ViT-B/16 | CIFAR-10 | ~86M | ImageNet 预训练微调，RTX 4060 优化 |
| BLIP-Base | - | ~约200M | 从零实现，未预训练 |

---

## 技术要点

- 所有 CIFAR-10 模型使用标准数据增强（随机翻转、随机裁剪）和 ImageNet/CIFAR 归一化参数。
- AlexNet / VGGNet 实现了 **特征图可视化** 和 **Grad-CAM**，便于理解模型的决策依据。
- ViT 实现了 **混合精度训练** 与 **Warmup + Cosine 学习率调度**，适合在消费级显卡上高效训练。
- BLIP 模型手动实现了完整的 **视觉-语言预训练** 架构，涵盖对比学习、图文匹配和生成式图像描述三个预训练目标所需的所有子模块。
- 图文检索系统支持 **百万级** 向量的高效近似最近邻搜索（FAISS IVF / HNSW 索引）。
