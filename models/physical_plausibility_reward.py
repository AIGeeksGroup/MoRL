import torch

from rewards.base import RewardComponent


class PhysicalPlausibilityReward(RewardComponent):
    def __init__(
        self,
        joint_weight=0.6,
        vel_weight=0.25,
        acc_weight=0.15,
        joint_limit=4.0,
        vel_limit=0.25,
        acc_limit=0.45,
    ):
        self.joint_weight = float(joint_weight)
        self.vel_weight = float(vel_weight)
        self.acc_weight = float(acc_weight)
        self.joint_limit = float(joint_limit)
        self.vel_limit = float(vel_limit)
        self.acc_limit = float(acc_limit)

    def __call__(self, sample):
        motion = sample.get("motion")
        if motion is None:
            return 0.0

        if hasattr(motion, "detach"):
            m = motion.detach()
        else:
            m = motion

        if getattr(m, "ndim", 0) < 3:
            return 0.0

        joint_excess = torch.relu(m.abs() - self.joint_limit)
        joint_penalty = torch.tanh(joint_excess.mean()).item()

        if m.shape[1] > 1:
            vel = torch.abs(m[:, 1:, :] - m[:, :-1, :])
            vel_excess = torch.relu(vel - self.vel_limit)
            vel_penalty = torch.tanh(vel_excess.mean()).item()
        else:
            vel_penalty = 0.0

        if m.shape[1] > 2:
            acc = torch.abs(m[:, 2:, :] - 2.0 * m[:, 1:-1, :] + m[:, :-2, :])
            acc_excess = torch.relu(acc - self.acc_limit)
            acc_penalty = torch.tanh(acc_excess.mean()).item()
        else:
            acc_penalty = 0.0

        reward = -(
            self.joint_weight * joint_penalty
            + self.vel_weight * vel_penalty
            + self.acc_weight * acc_penalty
        )
        return float(max(-1.0, min(1.0, reward)))


