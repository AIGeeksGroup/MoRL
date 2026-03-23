import types
import os
import sys

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from rl.grpo_trainer import GRPOTrainer


class DummyNet:
    def forward_decoder(self, tokens):
        if torch.is_tensor(tokens):
            n = int(tokens.numel())
        else:
            n = len(tokens)
        n = max(4, n)
        return torch.zeros(1, n, 263)


class DummyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.scalar = torch.nn.Parameter(torch.tensor(0.5))
        self.net = DummyNet()

    def generate_with_trace(self, caption, do_sample=True):
        base = len(caption.split()) + (1 if do_sample else 0)
        tokens = torch.tensor([1, 2, 3, 4, base], dtype=torch.long)
        return {
            "caption": caption,
            "reasoning": "step-by-step plan",
            "answer": "generated motion",
            "motion_tokens": tokens,
        }

    def generate_caption_with_trace(self, motion_tokens, do_sample=True):
        n = int(motion_tokens.numel()) if torch.is_tensor(motion_tokens) else len(motion_tokens)
        return {
            "reasoning": "observe then summarize",
            "answer": f"caption with {n} tokens",
            "motion_tokens": motion_tokens,
        }

    def generate_com(self, caption=None, motion_tokens=None, task="t2m", k=4, t=1, reward_fn=None, return_candidates=False):
        if task == "t2m":
            samples = [self.generate_with_trace(caption, do_sample=True) for _ in range(k)]
        else:
            samples = [self.generate_caption_with_trace(motion_tokens, do_sample=True) for _ in range(k)]
        if return_candidates:
            return samples
        return samples[0]

    def compute_sequence_nll(self, captions, motions, reasoning=None, sample_weights=None):
        total = self.scalar * 0.0
        for caption, motion in zip(captions, motions):
            clen = len(caption.split()) if isinstance(caption, str) else 1
            mlen = int(motion.numel()) if torch.is_tensor(motion) else len(motion)
            total = total + self.scalar * float(clen + mlen) / 100.0
        return total


class DummyRewardComponent:
    pass


class DummyReward:
    def __init__(self):
        self.components = [DummyRewardComponent()]
        self.weights = [1.0]

    def score_group(self, samples):
        totals = []
        details = []
        key = "DummyRewardComponent"
        comp_scores = {key: []}
        for sample in samples:
            ans = sample.get("answer") or ""
            score = float(len(ans.split()))
            totals.append(score)
            details.append({key: score})
            comp_scores[key].append(score)
        return {
            "totals": totals,
            "details": details,
            "component_means": {key: float(sum(totals) / max(1, len(totals)))},
            "component_stds": {key: float(torch.tensor(totals).std(unbiased=False).item()) if len(totals) > 0 else 0.0},
            "component_cache_stats": {},
            "component_scores": comp_scores,
        }

    def normalize_group(self, rewards, eps=1e-6):
        arr = torch.tensor(rewards, dtype=torch.float32)
        mean = float(arr.mean().item())
        std = float(arr.std(unbiased=False).item())
        norm = ((arr - mean) / (std + eps)).tolist()
        return norm, {"mean": mean, "std": std}

    def normalize_components(self, component_scores, eps=1e-6):
        out = {}
        stats = {}
        for k, vals in component_scores.items():
            arr = torch.tensor(vals, dtype=torch.float32)
            mean = float(arr.mean().item())
            std = float(arr.std(unbiased=False).item())
            out[k] = ((arr - mean) / (std + eps)).tolist()
            stats[k] = {"mean": mean, "std": std}
        return out, stats


def build_args():
    args = types.SimpleNamespace()
    args.rl_group_size = 4
    args.com_refine_steps = 1
    args.rl_kl_coef = 0.01
    args.rl_component_norm = True
    return args


def main():
    model = DummyModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    trainer = GRPOTrainer(
        model=model,
        optimizer=optimizer,
        reward_fn=DummyReward(),
        args=build_args(),
        device="cpu",
    )

    t2m_examples = [{"caption": "a person walks and turns"}, {"caption": "a person jumps"}]
    m2t_examples = [
        {"caption": "walk forward", "motion_tokens": torch.tensor([1, 2, 3, 4, 5], dtype=torch.long)},
        {"caption": "jump once", "motion_tokens": torch.tensor([6, 7, 8], dtype=torch.long)},
    ]

    out_t2m = trainer.train_batch(t2m_examples, task="t2m")
    out_m2t = trainer.train_batch(m2t_examples, task="m2t")

    print("t2m_loss", round(out_t2m["loss"], 6), "t2m_reward", round(out_t2m["reward"], 6))
    print("m2t_loss", round(out_m2t["loss"], 6), "m2t_reward", round(out_m2t["reward"], 6))


if __name__ == "__main__":
    main()





