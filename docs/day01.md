# Day 01 - 环境配置 + 数据下载 + 第一次训练

## 完成情况

- [x] conda 环境 minimind (python=3.10)
- [x] PyTorch 2.10.0 安装，MPS 验证通过
- [x] git clone minimind，安装项目依赖
- [x] 数据集下载完成（pretrain_hq / sft_mini_512 等）
- [x] Pretrain 训练跑通（500步，loss 7.43 → 6.45）
- [x] SFT 训练跑通
- [x] 推理对话跑通（184 tokens/s）

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
print(torch.backends.mps.is_built())      # True
```

## 数据集

从 HuggingFace 下载 `jingyaogong/minimind_dataset`，主要文件：

| 文件 | 大小 | 用途 |
|------|------|------|
| pretrain_hq.jsonl | 1.6G | Pretrain 高质量数据 |
| sft_mini_512.jsonl | 1.2G | SFT 小规模数据 |
| sft_512.jsonl | 7.5G | SFT 完整数据 |
| sft_1024.jsonl | 5.6G | SFT 长文本数据 |
| dpo.jsonl | 54M | DPO 偏好数据 |

## 训练命令

**Pretrain（从 trainer/ 目录运行）：**
```bash
cd ~/projects/minimind/trainer
PYTORCH_ENABLE_MPS_FALLBACK=1 python train_pretrain.py \
  --device mps \
  --data_path ../dataset/pretrain_hq.jsonl \
  --max_seq_len 512 \
  --batch_size 8 \
  --log_interval 50 \
  --save_interval 100
```

**SFT：**
```bash
PYTORCH_ENABLE_MPS_FALLBACK=1 python train_full_sft.py \
  --device mps \
  --data_path ../dataset/sft_mini_512.jsonl \
  --max_seq_len 512 \
  --batch_size 8 \
  --log_interval 50
```

**推理：**
```bash
cd ~/projects/minimind
python eval_llm.py --weight full_sft
```

## Pretrain loss 曲线（500步）

| Step | Loss |
|------|------|
| 50   | 7.43 |
| 100  | 7.21 |
| 150  | 7.06 |
| 200  | 6.87 |
| 250  | 6.91 |
| 300  | 7.08 |
| 350  | 6.87 |
| 400  | 6.85 |
| 450  | 6.53 |
| 500  | 6.45 |

整体下降趋势明显，验证训练流程正常。

## 推理效果（官方 full_sft_512 权重，25.83M 参数）

速度：120~200 tokens/s（MPS）

回答质量样例：
- 问候/常识类：回答流畅，逻辑正确
- 代码生成：能写出基本可运行的 Python
- 中文知识：表现良好

## 踩坑记录

1. **tokenizer 路径问题**：`train_pretrain.py` 默认 tokenizer 路径是 `'../model'`，相对于 `trainer/` 目录，必须从 `trainer/` 目录运行，不能从项目根目录运行。

2. **SFT 需要 pretrain 权重**：`train_full_sft.py` 会加载 `../out/pretrain_512.pth`，没有这个文件会报 FileNotFoundError。跑 SFT 前需先保存 pretrain 权重（加 `--save_interval 100`）。

3. **官方权重在 ModelScope**：HuggingFace 上的 `jingyaogong/minimind` 不存在，权重托管在 ModelScope：
   ```bash
   pip install modelscope
   python -c "from modelscope import snapshot_download; snapshot_download('gongjy/MiniMind2-PyTorch', local_dir='./out')"
   ```

## 今日疑问（待 Day 2 解答）

- pretrain_hq.jsonl 的数据格式是什么？每条样本长什么样？
- tokenizer 是怎么训练的？vocab size 是多少？
- 为什么 Pretrain loss 在 250-300 步有一个小反弹？
