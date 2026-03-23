import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from rewards.base import RewardComponent


class NLICoherenceReward(RewardComponent):
    def __init__(self, model_name='microsoft/deberta-v3-large-mnli', device='cpu', fallback=0.0, max_batch_size=16):
        self.model_name = model_name
        self.device = device
        self.fallback = float(fallback)
        self.max_batch_size = int(max_batch_size)
        self._tokenizer = None
        self._model = None
        self._entail_idx = 2
        self._cache = {}
        self._cache_hits = 0
        self._cache_misses = 0

    def _lazy_init(self):
        if self._tokenizer is not None and self._model is not None:
            return
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self._model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
        self._model.to(self.device)
        self._model.eval()

        id2label = getattr(self._model.config, 'id2label', None)
        if isinstance(id2label, dict):
            for idx, name in id2label.items():
                if 'entail' in str(name).lower():
                    self._entail_idx = int(idx)
                    break

    def __call__(self, sample):
        reasoning = sample.get("reasoning")
        answer = sample.get("answer")
        if not reasoning or not answer:
            return self.fallback
        key = (reasoning, answer)
        if key in self._cache:
            self._cache_hits += 1
            return self._cache[key]
        self._cache_misses += 1
        try:
            self._lazy_init()
            tokens = self._tokenizer(
                reasoning,
                answer,
                return_tensors='pt',
                truncation=True,
                max_length=256
            )
            tokens = {k: v.to(self.device) for k, v in tokens.items()}
            with torch.no_grad():
                logits = self._model(**tokens).logits
                probs = torch.softmax(logits, dim=-1)
            score = probs[0, self._entail_idx].item()
            score = float(score)
            if len(self._cache) >= 4096:
                self._cache.clear()
            self._cache[key] = score
            return score
        except Exception:
            return self.fallback

    def score_batch(self, samples):
        scores = [self.fallback for _ in samples]
        valid = []
        missing = []

        for i, sample in enumerate(samples):
            reasoning = sample.get('reasoning')
            answer = sample.get('answer')
            if not reasoning or not answer:
                continue
            key = (reasoning, answer)
            if key in self._cache:
                self._cache_hits += 1
                scores[i] = self._cache[key]
            else:
                self._cache_misses += 1
                valid.append(i)
                missing.append((reasoning, answer))

        if len(missing) == 0:
            return scores

        try:
            self._lazy_init()
            offset = 0
            for start in range(0, len(missing), self.max_batch_size):
                chunk = missing[start:start + self.max_batch_size]
                prem = [x[0] for x in chunk]
                hyp = [x[1] for x in chunk]

                tokens = self._tokenizer(
                    prem,
                    hyp,
                    return_tensors='pt',
                    truncation=True,
                    padding=True,
                    max_length=256
                )
                tokens = {k: v.to(self.device) for k, v in tokens.items()}

                with torch.no_grad():
                    logits = self._model(**tokens).logits
                    probs = torch.softmax(logits, dim=-1)

                for j in range(len(chunk)):
                    score = float(probs[j, self._entail_idx].item())
                    out_idx = valid[offset + j]
                    pair = chunk[j]
                    scores[out_idx] = score
                    if len(self._cache) >= 4096:
                        self._cache.clear()
                    self._cache[pair] = score

                offset += len(chunk)
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






