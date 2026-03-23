import os

import numpy as np
import torch
import torch.nn.functional as F

from rewards.base import RewardComponent
from models.evaluator_wrapper import EvaluatorModelWrapper
from options.get_eval_option import get_opt
from utils.word_vectorizer import WordVectorizer


class TextMotionConsistencyReward(RewardComponent):
    def __init__(
        self,
        dataset_opt_path='checkpoints/t2m/Comp_v6_KLD005/opt.txt',
        glove_path='./glove',
        device='cpu',
        max_text_len=20,
        fallback=0.0,
    ):
        self.dataset_opt_path = str(dataset_opt_path)
        self.glove_path = str(glove_path)
        self.device = str(device)
        self.max_text_len = int(max_text_len)
        self.fallback = float(fallback)

        self._wrapper = None
        self._w_vectorizer = None
        self._text_cache = {}

    def _lazy_init(self):
        if self._wrapper is not None and self._w_vectorizer is not None:
            return

        if not os.path.exists(self.dataset_opt_path):
            raise FileNotFoundError(self.dataset_opt_path)

        self._w_vectorizer = WordVectorizer(self.glove_path, 'our_vab')
        opt = get_opt(self.dataset_opt_path, self.device)
        self._wrapper = EvaluatorModelWrapper(opt)

    def _to_motion_tensor(self, motion):
        if motion is None:
            return None
        if not torch.is_tensor(motion):
            motion = torch.tensor(motion)
        motion = motion.detach().to(self.device).float()
        if motion.ndim == 2:
            motion = motion.unsqueeze(0)
        if motion.ndim != 3 or motion.shape[1] < 2:
            return None
        return motion

    def _build_text_tensors(self, caption):
        if caption in self._text_cache:
            return self._text_cache[caption]

        words = [w.lower().strip(',.!?;:"\'()[]{}') for w in caption.split()]
        words = [w for w in words if len(w) > 0]
        tokens = [f'{w}/OTHER' for w in words[:self.max_text_len]]
        tokens = ['sos/OTHER'] + tokens + ['eos/OTHER']
        sent_len = len(tokens)
        pad_len = self.max_text_len + 2 - sent_len
        if pad_len > 0:
            tokens = tokens + ['unk/OTHER'] * pad_len

        word_embeddings = []
        pos_one_hots = []
        for token in tokens:
            emb, pos = self._w_vectorizer[token]
            word_embeddings.append(emb[None, :])
            pos_one_hots.append(pos[None, :])

        word_embeddings = np.concatenate(word_embeddings, axis=0)
        pos_one_hots = np.concatenate(pos_one_hots, axis=0)

        out = (
            torch.from_numpy(word_embeddings).unsqueeze(0).float().to(self.device),
            torch.from_numpy(pos_one_hots).unsqueeze(0).float().to(self.device),
            torch.tensor([sent_len], dtype=torch.long, device=self.device),
        )
        if len(self._text_cache) >= 4096:
            self._text_cache.clear()
        self._text_cache[caption] = out
        return out

    def _encode_motion(self, motion):
        m_lens = torch.tensor([motion.shape[1]], dtype=torch.long, device=self.device)
        return self._wrapper.get_motion_embeddings(motion, m_lens)

    def __call__(self, sample):
        caption = sample.get('caption')
        motion = self._to_motion_tensor(sample.get('motion'))
        if not caption or motion is None:
            return self.fallback

        try:
            self._lazy_init()
            word_embeddings, pos_one_hots, sent_len = self._build_text_tensors(caption)
            text_emb = self._wrapper.text_encoder(word_embeddings, pos_one_hots, sent_len)
            motion_emb = self._encode_motion(motion)
            sim = F.cosine_similarity(text_emb, motion_emb, dim=-1).mean().item()
            return float(max(-1.0, min(1.0, sim)))
        except Exception:
            return self.fallback


