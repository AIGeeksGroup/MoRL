import numpy as np


def group_normalized_advantages(rewards, eps=1e-6):
    rewards = np.array(rewards, dtype=np.float32)
    mean = rewards.mean()
    std = rewards.std()
    return ((rewards - mean) / (std + eps)).tolist()

