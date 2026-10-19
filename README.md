# AI Infra 学习记录

记录我日常学习 AI Infra 的代码。


## 仓库内容

| 目录 | 内容 |
| --- | --- |
| [transformer/](./transformer/) | 小规模中英翻译练习 |
| [cat&dog/](./cat%26dog/) | 基于预训练 ResNet18 的猫狗分类 |

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