"""
Plot training curves from minimind logs.

Usage:
    python plot_curves.py

Outputs PNG files to ./plots/
"""
import os
import re
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

LOG_DIR = "/usr1/home/s124mdg55_08/project_minimind/minimind/logs"
OUT_DIR = "./plots"
os.makedirs(OUT_DIR, exist_ok=True)

plt.rcParams["figure.figsize"] = (10, 5)
plt.rcParams["font.size"] = 11


# ============================================================================
# 1. Pretrain loss curve
# ============================================================================
def plot_pretrain():
    path = os.path.join(LOG_DIR, "pretrain.log")
    if not os.path.exists(path):
        print(f"[skip] {path} not found")
        return
    steps, losses = [], []
    # minimind log: "Epoch:[1/1](1234/39695) loss:X.XXX ..."
    pat = re.compile(r"\((\d+)/(\d+)\),?\s*loss:\s*([0-9.]+)")
    with open(path) as f:
        for line in f:
            m = pat.search(line)
            if m:
                steps.append(int(m.group(1)))
                losses.append(float(m.group(3)))
    if not steps:
        print(f"[skip] no data parsed from {path}")
        return
    plt.figure()
    plt.plot(steps, losses, linewidth=0.8, color="steelblue")
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.title(f"Pretrain Loss ({len(steps):,} steps)")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    out = os.path.join(OUT_DIR, "pretrain_loss.png")
    plt.savefig(out, dpi=120)
    plt.close()
    print(f"[ok] {out}  (loss {losses[0]:.2f} -> {losses[-1]:.2f})")


# ============================================================================
# 2. SFT loss curve
# ============================================================================
def plot_sft():
    path = os.path.join(LOG_DIR, "sft.log")
    if not os.path.exists(path):
        print(f"[skip] {path} not found")
        return
    steps, losses = [], []
    pat = re.compile(r"\((\d+)/(\d+)\),?\s*loss:\s*([0-9.]+)")
    with open(path) as f:
        for line in f:
            m = pat.search(line)
            if m:
                steps.append(int(m.group(1)))
                losses.append(float(m.group(3)))
    if not steps:
        print(f"[skip] no data parsed from {path}")
        return
    plt.figure()
    plt.plot(steps, losses, linewidth=0.8, color="darkorange")
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.title(f"SFT Loss ({len(steps):,} steps)")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    out = os.path.join(OUT_DIR, "sft_loss.png")
    plt.savefig(out, dpi=120)
    plt.close()
    print(f"[ok] {out}  (loss {losses[0]:.2f} -> {losses[-1]:.2f})")


# ============================================================================
# 3. DPO loss curve
# ============================================================================
def plot_dpo():
    path = os.path.join(LOG_DIR, "dpo.log")
    if not os.path.exists(path):
        print(f"[skip] {path} not found")
        return
    steps, losses = [], []
    # DPO log format may be "loss:X.XXX"
    pat = re.compile(r"\((\d+)/(\d+)\),?\s*loss:\s*([0-9.]+)")
    with open(path) as f:
        for line in f:
            m = pat.search(line)
            if m:
                steps.append(int(m.group(1)))
                losses.append(float(m.group(3)))
    if not steps:
        print(f"[skip] no data parsed from {path}")
        return
    plt.figure()
    plt.plot(steps, losses, linewidth=0.8, color="seagreen")
    plt.axhline(y=np.log(2), color="gray", linestyle="--", alpha=0.5,
                label="Theoretical start: ln(2) ≈ 0.69")
    plt.xlabel("Step")
    plt.ylabel("DPO Loss")
    plt.title(f"DPO Loss ({len(steps):,} steps)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    out = os.path.join(OUT_DIR, "dpo_loss.png")
    plt.savefig(out, dpi=120)
    plt.close()
    print(f"[ok] {out}  (loss {losses[0]:.2f} -> {losses[-1]:.2f})")


# ============================================================================
# 4. GRPO v1 (model-based) reward + KL curve
# ============================================================================
def plot_grpo_v1():
    path = os.path.join(LOG_DIR, "grpo.log")
    if not os.path.exists(path):
        print(f"[skip] {path} not found")
        return
    steps, rewards, kls, lens = [], [], [], []
    # Epoch:[1/1](1/9751), Reward: -2.3, KL_ref: 0.0, ..., Avg Response Len: 268
    pat = re.compile(
        r"\((\d+)/(\d+)\).*?Reward:\s*(-?[0-9.]+).*?KL_ref:\s*(-?[0-9.]+).*?Avg Response Len:\s*([0-9.]+)"
    )
    with open(path) as f:
        for line in f:
            m = pat.search(line)
            if m:
                steps.append(int(m.group(1)))
                rewards.append(float(m.group(3)))
                kls.append(float(m.group(4)))
                lens.append(float(m.group(5)))
    if not steps:
        print(f"[skip] no data parsed from {path}")
        return

    # Plot reward
    plt.figure()
    # moving average for smoothing
    window = max(1, len(rewards) // 200)
    smooth = np.convolve(rewards, np.ones(window)/window, mode="valid")
    plt.plot(steps, rewards, linewidth=0.3, alpha=0.3, color="crimson", label="raw")
    plt.plot(steps[window-1:], smooth, linewidth=1.5, color="crimson",
             label=f"moving avg (window={window})")
    plt.xlabel("Step")
    plt.ylabel("Reward")
    plt.title(f"GRPO v1 (model-based, InternLM2-1.8B-Reward) — Reward over {len(steps):,} steps")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    out = os.path.join(OUT_DIR, "grpo_v1_reward.png")
    plt.savefig(out, dpi=120)
    plt.close()
    print(f"[ok] {out}  (reward {rewards[0]:.2f} -> {rewards[-1]:.2f}, mean {np.mean(rewards):.2f})")

    # Plot response length (shows reward hacking via length bloat)
    plt.figure()
    smooth_len = np.convolve(lens, np.ones(window)/window, mode="valid")
    plt.plot(steps, lens, linewidth=0.3, alpha=0.3, color="darkblue", label="raw")
    plt.plot(steps[window-1:], smooth_len, linewidth=1.5, color="darkblue",
             label=f"moving avg (window={window})")
    plt.xlabel("Step")
    plt.ylabel("Average Response Length (tokens)")
    plt.title(f"GRPO v1 — Response Length Growth (reward hacking signal)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    out = os.path.join(OUT_DIR, "grpo_v1_response_len.png")
    plt.savefig(out, dpi=120)
    plt.close()
    print(f"[ok] {out}  (len {lens[0]:.0f} -> {lens[-1]:.0f})")


# ============================================================================
# 5. GRPO v2 (R1-style rule-based) reward + format + correct curves
# ============================================================================
def plot_grpo_r1():
    path = os.path.join(LOG_DIR, "r1_main.log")
    if not os.path.exists(path):
        print(f"[skip] {path} not found")
        return
    steps, rewards, fmts, corrects, kls = [], [], [], [], []
    # Epoch:[1/1](1/200), Reward: 0.339, Fmt: 1.00, Correct: 0.00, KL_ref: 0.001, ...
    pat = re.compile(
        r"\((\d+)/(\d+)\).*?Reward:\s*(-?[0-9.]+).*?Fmt:\s*([0-9.]+).*?Correct:\s*([0-9.]+).*?KL_ref:\s*(-?[0-9.]+)"
    )
    with open(path) as f:
        for line in f:
            m = pat.search(line)
            if m:
                steps.append(int(m.group(1)))
                rewards.append(float(m.group(3)))
                fmts.append(float(m.group(4)))
                corrects.append(float(m.group(5)))
                kls.append(float(m.group(6)))
    if not steps:
        print(f"[skip] no data parsed from {path}")
        return

    fig, axes = plt.subplots(1, 3, figsize=(16, 4))

    # (a) Reward
    axes[0].plot(steps, rewards, linewidth=0.8, color="crimson", alpha=0.5, label="raw")
    window = max(1, len(rewards) // 30)
    if len(rewards) > window:
        smooth = np.convolve(rewards, np.ones(window)/window, mode="valid")
        axes[0].plot(steps[window-1:], smooth, linewidth=2.0, color="crimson",
                     label=f"moving avg")
    axes[0].set_xlabel("Step")
    axes[0].set_ylabel("Reward")
    axes[0].set_title("Total Reward")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    # (b) Format rate
    axes[1].plot(steps, fmts, linewidth=1.0, color="darkgreen")
    axes[1].set_xlabel("Step")
    axes[1].set_ylabel("Format Rate")
    axes[1].set_title("Format Reward Rate (has </think>)")
    axes[1].set_ylim(-0.05, 1.1)
    axes[1].grid(alpha=0.3)

    # (c) Correct rate
    axes[2].plot(steps, corrects, linewidth=0.8, color="navy", alpha=0.5, label="raw")
    if len(corrects) > window:
        smooth = np.convolve(corrects, np.ones(window)/window, mode="valid")
        axes[2].plot(steps[window-1:], smooth, linewidth=2.0, color="navy",
                     label=f"moving avg")
    axes[2].set_xlabel("Step")
    axes[2].set_ylabel("Correct Rate")
    axes[2].set_title("Math Correctness (30M capacity ceiling)")
    axes[2].legend()
    axes[2].grid(alpha=0.3)

    plt.suptitle(f"GRPO v2 (R1-style rule-based, GSM8K) — {len(steps)} steps", y=1.02)
    plt.tight_layout()
    out = os.path.join(OUT_DIR, "grpo_r1_curves.png")
    plt.savefig(out, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"[ok] {out}  (reward {np.mean(rewards[:20]):.3f} -> {np.mean(rewards[-20:]):.3f}, "
          f"correct {np.mean(corrects[:20]):.3f} -> {np.mean(corrects[-20:]):.3f})")


# ============================================================================
# 6. GRPO v1 vs v2 side-by-side reward comparison
# ============================================================================
def plot_grpo_comparison():
    v1_path = os.path.join(LOG_DIR, "grpo.log")
    v2_path = os.path.join(LOG_DIR, "r1_main.log")

    def load_rewards(path, pat):
        steps, rewards = [], []
        if not os.path.exists(path):
            return steps, rewards
        with open(path) as f:
            for line in f:
                m = pat.search(line)
                if m:
                    steps.append(int(m.group(1)))
                    rewards.append(float(m.group(3)))
        return steps, rewards

    v1_pat = re.compile(r"\((\d+)/(\d+)\).*?Reward:\s*(-?[0-9.]+)")
    v2_pat = re.compile(r"\((\d+)/(\d+)\).*?Reward:\s*(-?[0-9.]+)")

    s1, r1 = load_rewards(v1_path, v1_pat)
    s2, r2 = load_rewards(v2_path, v2_pat)
    if not s1 or not s2:
        print(f"[skip] comparison: one of the logs missing")
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 4))

    # v1
    w1 = max(1, len(r1) // 200)
    if len(r1) > w1:
        smooth1 = np.convolve(r1, np.ones(w1)/w1, mode="valid")
        axes[0].plot(s1[w1-1:], smooth1, linewidth=1.5, color="crimson")
    axes[0].plot(s1, r1, linewidth=0.2, alpha=0.25, color="crimson")
    axes[0].set_xlabel("Step")
    axes[0].set_ylabel("Reward")
    axes[0].set_title(f"GRPO v1: Model-based Reward\n"
                     f"(InternLM2-1.8B-Reward, {len(s1):,} steps, mean={np.mean(r1):.2f})")
    axes[0].grid(alpha=0.3)
    axes[0].axhline(y=np.mean(r1), color="gray", linestyle="--", alpha=0.4)

    # v2
    w2 = max(1, len(r2) // 30)
    if len(r2) > w2:
        smooth2 = np.convolve(r2, np.ones(w2)/w2, mode="valid")
        axes[1].plot(s2[w2-1:], smooth2, linewidth=1.5, color="darkgreen")
    axes[1].plot(s2, r2, linewidth=0.5, alpha=0.4, color="darkgreen")
    axes[1].set_xlabel("Step")
    axes[1].set_ylabel("Reward")
    axes[1].set_title(f"GRPO v2: Rule-based R1-style Reward\n"
                     f"(GSM8K format+correctness, {len(s2)} steps, "
                     f"{np.mean(r2[:20]):.2f} -> {np.mean(r2[-20:]):.2f})")
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    out = os.path.join(OUT_DIR, "grpo_v1_vs_v2.png")
    plt.savefig(out, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"[ok] {out}")


if __name__ == "__main__":
    print("Plotting training curves...")
    plot_pretrain()
    plot_sft()
    plot_dpo()
    plot_grpo_v1()
    plot_grpo_r1()
    plot_grpo_comparison()
    print(f"\nAll plots saved to {OUT_DIR}/")
