# Day 01 - 环境配置 + 数据下载 + 第一次训练

## 完成情况

- [x] conda 环境 minimind (python=3.10)
- [x] PyTorch 2.10.0 安装，MPS 验证通过
- [x] git clone minimind，安装项目依赖
- [x] 数据集下载完成（pretrain_hq / sft_mini_512 等）
- [ ] Pretrain 训练跑通（进行中）

## 环境记录

```bash
conda create -n minimind python=3.10
conda activate minimind
pip install torch torchvision torchaudio
pip install transformers datasets tokenizers sentencepiece accelerate tqdm wandb
```

MPS 验证：
```python
import torch
print(torch.backends.mps.is_available())  # True
```

## 数据集

从 HuggingFace 下载 `jingyaogong/minimind_dataset`，主要文件：

| 文件 | 大小 | 用途 |
|------|------|------|
| pretrain_hq.jsonl | 1.6G | Pretrain 高质量数据 |
| sft_mini_512.jsonl | 1.2G | SFT 小规模数据 |
| sft_512.jsonl | 7.5G | SFT 完整数据 |
| dpo.jsonl | 54M | DPO 偏好数据 |

## 训练命令

```bash
cd ~/projects/minimind/trainer
PYTORCH_ENABLE_MPS_FALLBACK=1 python train_pretrain.py \
  --device mps \
  --data_path ../dataset/pretrain_hq.jsonl \
  --max_seq_len 512 \
  --batch_size 8
```

## 踩坑记录

1. **tokenizer 路径问题**：脚本默认路径 `'../model'` 是相对于 `trainer/` 目录，需要从 `trainer/` 目录运行，不能从项目根目录运行。

## 今日疑问（待 Day 2 解答）

- pretrain_hq.jsonl 的数据格式是什么？每条样本长什么样？
- tokenizer 是怎么训练的？vocab size 是多少？
