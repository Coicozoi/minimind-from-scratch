"""
GSM8K Dataset for R1-style rule-based GRPO on minimind.
Returns {'prompt': str, 'answer': str} compatible with minimind's grpo_train_epoch.
"""
import re
import random
from torch.utils.data import Dataset
from datasets import load_dataset


R1_SYSTEM_PROMPT = """你是一个数学解题助手。对每道数学题,你必须严格按以下格式回答:

<think>
[详细的推理过程]
</think>
<answer>
[最终的数字答案,只写一个数字]
</answer>

示例:
问题:小明有3个苹果,又买了5个,总共多少?
<think>
小明一开始有3个苹果,又买了5个。3 + 5 = 8。
</think>
<answer>
8
</answer>"""


def extract_gt(answer_field: str) -> str:
    m = re.search(r"####\s*([-+]?\d[\d,]*\.?\d*)", answer_field)
    if m:
        return m.group(1).replace(",", "").strip()
    return ""


class GSM8KDataset(Dataset):
    def __init__(self, tokenizer, split="train", num_samples=1000, thinking_ratio=1.0, seed=42):
        super().__init__()
        self.tokenizer = tokenizer
        self.thinking_ratio = thinking_ratio

        ds = load_dataset("gsm8k", "main", split=split)
        ds = ds.shuffle(seed=seed)
        if num_samples and num_samples < len(ds):
            ds = ds.select(range(num_samples))
        self.samples = ds

    def __len__(self):
        return len(self.samples)

    def create_chat_prompt(self, question: str):
        conversations = [
            {"role": "system", "content": R1_SYSTEM_PROMPT},
            {"role": "user", "content": f"问题:{question}"},
        ]
        use_thinking = random.random() < self.thinking_ratio
        return self.tokenizer.apply_chat_template(
            conversations,
            tokenize=False,
            open_thinking=use_thinking,
            add_generation_prompt=True,
        )

    def __getitem__(self, index):
        sample = self.samples[index]
        prompt = self.create_chat_prompt(sample["question"])
        gt = extract_gt(sample["answer"])
        return {
            "prompt": prompt,
            "answer": gt,
        }
