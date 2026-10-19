# AI Infra 学习记录

记录我日常学习 AI Infrastructure 的代码。


## 当前实验

| 目录 | 内容 | 学习重点 |
| --- | --- | --- |
| [transformer/](./transformer/) | 小规模中英翻译练习 | 词表、位置编码、Padding / Causal Mask、训练循环与贪心解码 |
| [cat&dog/](./cat%26dog/) | 基于预训练 ResNet18 的猫狗分类 | 数据划分、图像预处理、冻结参数、分类头训练与推理 |

Transformer 实验使用 PyTorch 内置的 `TransformerEncoderLayer` 和 `TransformerDecoderLayer`，自行组织模型、数据处理、Mask 和训练推理流程。语料是代码中内置的 16 对句子，用于理解流程；训练后的逐句测试使用同一批语料，不能作为泛化能力评估。

## 目录结构

```text
.
├── transformer/
│   ├── config.py              # 模型与训练超参数
│   ├── dataset.py             # 玩具语料、词表与 batch 整理
│   ├── model.py               # Encoder–Decoder 模型与贪心解码
│   ├── train.py               # 训练与 checkpoint 保存
│   └── predict.py             # 交互式翻译
├── cat&dog/
│   ├── download.py            # 通过 KaggleHub 下载数据集
│   ├── split_data.py          # 划分训练集和验证集
│   ├── train_cats_dogs.py     # ResNet18 分类头训练
│   └── predict.py             # 单张图片预测
├── .gitignore
└── README.md
```

## 环境准备

在仓库根目录创建虚拟环境：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

当前脚本所需的主要依赖：

```powershell
python -m pip install torch torchvision pillow tqdm kagglehub
```

仓库目前未固定 Python 和依赖版本。使用 GPU 时，需要根据本机驱动与 CUDA 环境安装匹配的 PyTorch；训练和推理脚本会在 CUDA 可用时使用 GPU，否则使用 CPU。

## 运行实验

脚本使用相对路径读写数据和权重，请进入对应实验目录后运行。以下两个示例均从仓库根目录开始。

### Transformer 翻译练习

```powershell
cd transformer
python train.py
python predict.py
```

- 数据已内置在 `dataset.py`，无需额外下载。
- 在 `config.py` 中调整超参数，默认训练 300 轮。
- 训练生成当前目录下的 `checkpoint.pt`，推理脚本读取该文件；推理时的模型结构配置应与训练时保持一致。
- 推理输入需要用空格分词，例如 `我 有 一个 苹果`，输入 `q` 退出。

### 猫狗分类

当前脚本包含本机路径，首次运行前需要调整：

| 文件 | 需要设置的内容 |
| --- | --- |
| `download.py` | 将 `KAGGLEHUB_CACHE` 改为自己的数据缓存目录 |
| `split_data.py` | 将 `src_dir` 指向下载后的 `PetImages`，将 `dst_dir` 改为 `data`（在实验目录内运行时） |
| `predict.py` | 将 `image_path` 改为待预测图片路径 |

`split_data.py` 当前的 `/cat&dog/data` 不是实验目录下的相对路径，需要先修改，再执行下面的命令。

```powershell
cd "cat&dog"
python download.py
python split_data.py
python train_cats_dogs.py
python predict.py
```

下载脚本调用 KaggleHub 登录，并下载 `shaunthesheep/microsoft-catsvsdogs-dataset`。划分脚本按每类前 1,000 张训练、随后 200 张验证进行复制，训练脚本期望的数据结构为：

```text
cat&dog/data/
├── train/
│   ├── cat/
│   └── dog/
└── val/
    ├── cat/
    └── dog/
```

首次训练需要下载 ResNet18 预训练权重，默认训练 5 轮，生成 `cats_dogs_resnet18.pth`。数据划分脚本不检查损坏图片，也不清空已有目标目录；若读取图片报错，需要检查数据后重新整理。
