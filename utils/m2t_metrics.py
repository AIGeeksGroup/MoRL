import math
import re
from collections import Counter, defaultdict

import numpy as np


def _tokenize(text):
    text = text.lower().strip()
    return re.findall(r"[a-z0-9']+", text)


def _ngrams(tokens, n):
    if len(tokens) < n:
        return []
    return [tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]


def _closest_ref_len(cand_len, ref_lens):
    return min(ref_lens, key=lambda x: (abs(x - cand_len), x))


def corpus_bleu(preds, refs, n=4):
    clipped = [0.0] * n
    total = [0.0] * n
    c_len = 0
    r_len = 0

    for pred, ref_list in zip(preds, refs):
        cand_toks = _tokenize(pred)
        ref_toks_list = [_tokenize(r) for r in ref_list]
        c_len += len(cand_toks)
        r_len += _closest_ref_len(len(cand_toks), [len(x) for x in ref_toks_list])

        for i in range(1, n + 1):
            cand_ng = Counter(_ngrams(cand_toks, i))
            total[i - 1] += max(1.0, float(sum(cand_ng.values())))

            ref_max = Counter()
            for rt in ref_toks_list:
                ref_max |= Counter(_ngrams(rt, i))
            overlap = sum((cand_ng & ref_max).values())
            clipped[i - 1] += float(overlap)

    precisions = [clipped[i] / max(1e-12, total[i]) for i in range(n)]
    bp = 1.0
    if c_len > 0 and c_len < r_len:
        bp = math.exp(1.0 - float(r_len) / float(c_len))

    if any(p <= 0 for p in precisions):
        bleu_n = 0.0
    else:
        bleu_n = bp * math.exp(sum(math.log(p) for p in precisions) / n)

    bleu_1 = bp * precisions[0]
    return float(bleu_1 * 100.0), float(bleu_n * 100.0)


def _lcs_len(a, b):
    if len(a) == 0 or len(b) == 0:
        return 0
    dp = [[0] * (len(b) + 1) for _ in range(len(a) + 1)]
    for i in range(1, len(a) + 1):
        for j in range(1, len(b) + 1):
            if a[i - 1] == b[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    return dp[-1][-1]


def rouge_l(preds, refs):
    scores = []
    for pred, ref_list in zip(preds, refs):
        cand = _tokenize(pred)
        best = 0.0
        for r in ref_list:
            rt = _tokenize(r)
            l = _lcs_len(cand, rt)
            if l == 0:
                cur = 0.0
            else:
                prec = l / max(1, len(cand))
                rec = l / max(1, len(rt))
                cur = (2 * prec * rec) / max(1e-12, (prec + rec))
            best = max(best, cur)
        scores.append(best)
    return float(np.mean(scores) * 100.0)


def cider_lite(preds, refs, n=4):
    # Build document frequency on reference ngrams.
    df = [defaultdict(int) for _ in range(n)]
    num_docs = 0
    for ref_list in refs:
        num_docs += 1
        seen = [set() for _ in range(n)]
        for r in ref_list:
            toks = _tokenize(r)
            for i in range(1, n + 1):
                seen[i - 1].update(_ngrams(toks, i))
        for i in range(n):
            for ng in seen[i]:
                df[i][ng] += 1

    def tfidf_vec(tokens, order):
        ngs = Counter(_ngrams(tokens, order))
        vec = {}
        for ng, tf in ngs.items():
            idf = math.log((num_docs + 1.0) / (1.0 + df[order - 1].get(ng, 0)))
            vec[ng] = tf * idf
        norm = math.sqrt(sum(v * v for v in vec.values()))
        if norm > 0:
            for k in list(vec.keys()):
                vec[k] /= norm
        return vec

    def cosine(a, b):
        if len(a) == 0 or len(b) == 0:
            return 0.0
        keys = set(a.keys()) & set(b.keys())
        return float(sum(a[k] * b[k] for k in keys))

    sample_scores = []
    for pred, ref_list in zip(preds, refs):
        cand_toks = _tokenize(pred)
        order_scores = []
        for i in range(1, n + 1):
            cand_vec = tfidf_vec(cand_toks, i)
            sims = []
            for r in ref_list:
                ref_vec = tfidf_vec(_tokenize(r), i)
                sims.append(cosine(cand_vec, ref_vec))
            order_scores.append(float(np.mean(sims) if len(sims) > 0 else 0.0))
        sample_scores.append(float(np.mean(order_scores)))
    return float(np.mean(sample_scores) * 10.0)


def bertscore_f1(preds, refs):
    try:
        from bert_score import score as bert_score
        # bert-score supports list[list[str]] references.
        _, _, f1 = bert_score(preds, refs, lang='en', verbose=False)
        return float(f1.mean().item() * 100.0)
    except Exception:
        return 0.0


def compute_m2t_metrics(preds, refs):
    bleu1, bleu4 = corpus_bleu(preds, refs, n=4)
    rouge = rouge_l(preds, refs)
    cider = cider_lite(preds, refs, n=4)
    try:
        from pycocoevalcap.cider.cider import Cider
        gts = {i: list(refs[i]) for i in range(len(refs))}
        res = {i: [preds[i]] for i in range(len(preds))}
        cider, _ = Cider().compute_score(gts, res)
    except Exception:
        pass
    bert = bertscore_f1(preds, refs)
    return {
        'BLEU@1': bleu1,
        'BLEU@4': bleu4,
        'ROUGE-L': rouge,
        'CIDEr': cider,
        'BERTScore': bert,
    }

