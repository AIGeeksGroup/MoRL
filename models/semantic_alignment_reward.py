import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

from rewards.base import RewardComponent


class SemanticAlignmentReward(RewardComponent):
    def __init__(self, model_name='microsoft/deberta-v3-base', device='cpu', default=0.0, max_batch_size=16):
        self.model_name = model_name
        self.device = device
        self.default = float(default)
        self.max_batch_size = int(max_batch_size)
        self._tokenizer = None
        self._model = None
        self._cache = {}
        self._cache_hits = 0
        self._cache_misses = 0

    def _lazy_init(self):
        if self._tokenizer is not None and self._model is not None:
            return
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self._model = AutoModel.from_pretrained(self.model_name)
        self._model.to(self.device)
        self._model.eval()

    def _encode(self, text):
        if text in self._cache:
            self._cache_hits += 1
            return self._cache[text]
        self._cache_misses += 1
        self._lazy_init()
        tokens = self._tokenizer(
            text,
            return_tensors='pt',
            truncation=True,
            max_length=128
        )
        tokens = {k: v.to(self.device) for k, v in tokens.items()}
        with torch.no_grad():
            hidden = self._model(**tokens).last_hidden_state
        mask = tokens['attention_mask'].unsqueeze(-1)
        pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)
        if len(self._cache) >= 4096:
            self._cache.clear()
        self._cache[text] = pooled
        return pooled

    def _encode_batch(self, texts):
        self._lazy_init()
        result = [None for _ in texts]
        missing = []

        for idx, text in enumerate(texts):
            if text in self._cache:
                self._cache_hits += 1
                result[idx] = self._cache[text]
            else:
                self._cache_misses += 1
                missing.append((idx, text))

        for start in range(0, len(missing), self.max_batch_size):
            chunk = missing[start:start + self.max_batch_size]
            chunk_texts = [x[1] for x in chunk]
            tokens = self._tokenizer(
                chunk_texts,
                return_tensors='pt',
                truncation=True,
                padding=True,
                max_length=128
            )
            tokens = {k: v.to(self.device) for k, v in tokens.items()}
            with torch.no_grad():
                hidden = self._model(**tokens).last_hidden_state
            mask = tokens['attention_mask'].unsqueeze(-1)
            pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)

            for j, (idx, text) in enumerate(chunk):
                vec = pooled[j:j + 1]
                result[idx] = vec
                if len(self._cache) >= 4096:
                    self._cache.clear()
                self._cache[text] = vec

        return result

    def __call__(self, sample):
        caption = sample.get("caption")
        answer = sample.get("answer")
        if not caption or not answer:
            return self.default
        try:
            c = self._encode(caption)
            a = self._encode(answer)
            sim = F.cosine_similarity(c, a, dim=-1).mean().item()
            return float(max(-1.0, min(1.0, sim)))
        except Exception:
            return self.default

    def score_batch(self, samples):
        captions = []
        answers = []
        valid = []
        scores = [self.default for _ in samples]

        for i, sample in enumerate(samples):
            caption = sample.get('caption')
            answer = sample.get('answer')
            if caption and answer:
                valid.append(i)
                captions.append(caption)
                answers.append(answer)

        if len(valid) == 0:
            return scores

        try:
            cap_vecs = self._encode_batch(captions)
            ans_vecs = self._encode_batch(answers)
            for out_idx, cv, av in zip(valid, cap_vecs, ans_vecs):
                sim = F.cosine_similarity(cv, av, dim=-1).mean().item()
                scores[out_idx] = float(max(-1.0, min(1.0, sim)))
        except Exception:
            return scores

        return scores

    def get_cache_stats(self):
        total = self._cache_hits + self._cache_misses
        hit_rate = float(self._cache_hits / total) if total > 0 else 0.0
        return {
            'hits': int(self._cache_hits),
            'misses': int(self._cache_misses),
            'hit_rate': hit_rate,
            'size': int(len(self._cache)),
        }

    def reset_cache(self, clear_values=False):
        self._cache_hits = 0
        self._cache_misses = 0
        if clear_values:
            self._cache.clear()






