import json
import torch
from torch.utils import data


class MotionCoTDataset(data.Dataset):
    def __init__(self, jsonl_path, task=None):
        self.samples = []
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                sample = json.loads(line)
                sample_task = sample.get("task")
                if task is not None and sample_task is not None and sample_task != task:
                    continue
                if "caption" not in sample:
                    raise ValueError("Each CoT sample must contain 'caption'.")
                if "motion_tokens" not in sample:
                    raise ValueError("Each CoT sample must contain 'motion_tokens'.")
                motion_tokens = sample["motion_tokens"]
                if isinstance(motion_tokens, str):
                    motion_tokens = [int(x) for x in motion_tokens.split() if x.strip()]
                if not isinstance(motion_tokens, list) or len(motion_tokens) == 0:
                    raise ValueError("'motion_tokens' must be a non-empty list or space-separated string.")
                sample["motion_tokens"] = motion_tokens
                sample["reasoning"] = sample.get("reasoning")
                sample["reward"] = sample.get("reward")
                sample["advantage"] = sample.get("advantage")
                sample["sample_weight"] = sample.get("sample_weight")
                self.samples.append(sample)

        if len(self.samples) == 0:
            raise ValueError("No valid CoT samples loaded.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = dict(self.samples[index])
        caption = sample.get("caption")
        motion_tokens = sample.get("motion_tokens")
        reasoning = sample.get("reasoning")
        task = sample.get("task")
        reward = sample.get("reward")
        advantage = sample.get("advantage")
        sample_weight = sample.get("sample_weight")

        if caption is None or motion_tokens is None:
            raise ValueError("Invalid CoT sample encountered.")

        return {
            "caption": caption,
            "motion_tokens": torch.tensor(motion_tokens, dtype=torch.long),
            "reasoning": reasoning,
            "task": task,
            "reward": reward,
            "advantage": advantage,
            "sample_weight": sample_weight,
        }


def cot_collate_fn(batch):
    captions = [x["caption"] for x in batch]
    motion_tokens = [x["motion_tokens"] for x in batch]
    reasonings = [x.get("reasoning") for x in batch]
    tasks = [x.get("task") for x in batch]
    rewards = [x.get("reward") for x in batch]
    advantages = [x.get("advantage") for x in batch]
    sample_weights = [x.get("sample_weight") for x in batch]
    return {
        "caption": captions,
        "motion_tokens": motion_tokens,
        "reasoning": reasonings,
        "task": tasks,
        "reward": rewards,
        "advantage": advantages,
        "sample_weight": sample_weights,
    }


def DATALoader(jsonl_path, batch_size, task=None, num_workers=0, shuffle=True):
    dataset = MotionCoTDataset(jsonl_path, task=task)
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=cot_collate_fn,
        drop_last=True,
    )

