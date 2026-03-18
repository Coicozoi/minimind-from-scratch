# MiniMind 从零复现项目

> 从零复现 MiniMind 26M 语言模型完整训练流程，用于大模型/NLP方向暑期实习简历项目。
>
> 完整记录 Pretrain → SFT → DPO → 垂直领域微调 的每一步，附每日学习笔记和实验数据。

## 项目背景

- **目标**：暑期实习，方向大模型/NLP算法
- **内容**：从零复现 MiniMind 26M 语言模型完整训练流程
- **产出**：GitHub 项目 + 知乎博客 + 简历项目描述 + 面试问答

## 开发环境

| 项目 | 版本 |
|------|------|
| 硬件 | Apple M4 MacBook Pro 16GB |
| Python | 3.10 (conda env: minimind) |
| PyTorch | 2.10.0，MPS backend |
| 系统 | macOS |

```bash
# 环境变量
export PYTORCH_ENABLE_MPS_FALLBACK=1
```

## 当前进度

- [x] 配好 conda 环境，PyTorch MPS 验证通过
- [x] git clone minimind，安装依赖
- [x] 下载数据集（pretrain_hq / sft_mini_512 等）
- [ ] Pretrain 跑通（loss 下降验证）
- [ ] SFT 跑通
- [ ] 推理对话跑通
- [ ] 读懂 Attention + RoPE 实现
- [ ] 垂直领域 SFT（医疗问答）
- [ ] 写知乎复现博客

## 目录结构

```
minimind-from-scratch/
├── README.md          # 项目总结（面试官视角）
├── docs/              # 每日学习笔记
│   ├── day01.md
│   ├── day02.md
│   └── ...
└── experiments/       # loss 曲线图、对比实验结果
```

## 学习计划

### 第一周：跑通 + 深度读懂

| Day | 主题 | 核心目标 |
|-----|------|----------|
| 1 | 环境 + 数据 + 第一次训练 | 跑通 Pretrain→SFT→推理完整流程 |
| 2 | Tokenizer + Pretrain 数据流 | 从数据到 loss 每一步都说得清 |
| 3 | Attention + RoPE | 面试能徒手写 Attention |
| 4 | Transformer 其余模块 + MoE | 整个模型结构无盲区 |
| 5 | SFT 原理 + 数据格式 | 理解 loss mask 和 instruction tuning |
| 6 | DPO 原理 | 能讲清楚偏好优化逻辑 |
| 7 | GRPO + 第一周总结 | 能讲清楚主流 RL 算法区别 |

### 第二周：改动 + 实验 + 沉淀

| Day | 主题 | 核心目标 |
|-----|------|----------|
| 8-9 | 垂直领域数据准备 | 医疗问答数据集处理 |
| 10-11 | 领域 SFT 训练 + 对比实验 | 有数据支撑的实验结论 |
| 12 | 扩展实验（选做） | 实验设计思维 |
| 13 | 写知乎复现博客 | 输出是最好的检验 |
| 14 | GitHub + 简历 + 面试准备 | 面试官可以直接看的项目 |

## 面试核心问答

1. Attention 为什么要除以 sqrt(d_k)？
2. RoPE 和绝对位置编码的区别？
3. SFT 和 Pretrain 的 loss 计算有什么不同？
4. DPO 为什么不需要 reward model？
5. GRPO 的核心创新是什么？

答案持续更新在 [docs/](docs/) 目录。

## Commit 规范

```
feat: 新功能或新实验
exp:  实验记录
docs: 笔记更新
fix:  修复问题
```

## 参考资料

- [MiniMind 原始项目](https://github.com/jingyaogong/minimind)
- 每日笔记见 [docs/](docs/) 目录
