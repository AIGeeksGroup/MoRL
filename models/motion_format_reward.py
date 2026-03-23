import torch

from rewards.base import RewardComponent


class MotionFormatReward(RewardComponent):
    """
    Validates the structural quality of generated motion tokens.

    Scoring rules (additive, total in [-1, 1]):
      - No tokens at all            →  empty_penalty  (-1.0)
      - Token count out of range    →  range_penalty   (-0.5)
      - Any token index out-of-vocab→  vocab_penalty   (-0.3) per violation (capped)
      - All checks passed           →  valid_bonus     (+1.0)

    HumanML3D defaults:
      min_tokens = 4   (≈ 40 frames / downsample-factor 4 / 2 ≈ 5, be generous)
      max_tokens = 52  (≈ 196 frames / 4 = 49, +small margin)
      nb_code    = 1024  (VQ-VAE codebook size)
    """

    def __init__(
        self,
        empty_penalty: float = -1.0,
        range_penalty: float = -0.5,
        vocab_penalty: float = -0.3,
        valid_bonus: float = 1.0,
        min_tokens: int = 4,
        max_tokens: int = 52,
        nb_code: int = 1024,
    ):
        self.empty_penalty = float(empty_penalty)
        self.range_penalty = float(range_penalty)
        self.vocab_penalty = float(vocab_penalty)
        self.valid_bonus   = float(valid_bonus)
        self.min_tokens    = int(min_tokens)
        self.max_tokens    = int(max_tokens)
        self.nb_code       = int(nb_code)

    def __call__(self, sample):
        motion_tokens = sample.get("motion_tokens")

        # ---- 1. Must exist and be non-empty ----
        if motion_tokens is None:
            return self.empty_penalty

        if torch.is_tensor(motion_tokens):
            tokens_list = motion_tokens.detach().cpu().tolist()
        else:
            tokens_list = list(motion_tokens)

        n = len(tokens_list)
        if n == 0:
            return self.empty_penalty

        # ---- 2. Token count must be within plausible range ----
        if not (self.min_tokens <= n <= self.max_tokens):
            return self.range_penalty

        # ---- 3. Every token index must be in [0, nb_code) ----
        bad = sum(1 for t in tokens_list if not (0 <= int(t) < self.nb_code))
        if bad > 0:
            # Penalty proportional to fraction of bad tokens, capped at vocab_penalty
            frac = bad / n
            return float(max(self.vocab_penalty, self.vocab_penalty * frac))

        # ---- 4. All checks passed ----
        return self.valid_bonus

