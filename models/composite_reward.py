import numpy as np


class CompositeReward:
    def __init__(self, components, weights=None):
        self.components = components
        if weights is None:
            weights = [1.0 for _ in components]
        self.weights = [float(w) for w in weights]

    def __call__(self, sample):
        total = 0.0
        detail = {}
        for comp, weight in zip(self.components, self.weights):
            name = comp.__class__.__name__
            value = float(comp(sample))
            detail[name] = value
            total += weight * value
        return total, detail

    def score_group(self, samples):
        if len(samples) == 0:
            return {
                'totals': [],
                'details': [],
                'component_means': {},
                'component_stds': {},
                'component_cache_stats': {},
            }

        component_scores = {}
        component_cache_stats = {}
        for comp in self.components:
            name = comp.__class__.__name__
            if hasattr(comp, 'score_batch'):
                vals = comp.score_batch(samples)
            else:
                vals = [comp(sample) for sample in samples]
            component_scores[name] = [float(v) for v in vals]
            if hasattr(comp, 'get_cache_stats'):
                component_cache_stats[name] = comp.get_cache_stats()

        totals = []
        details = []
        for i in range(len(samples)):
            total = 0.0
            detail = {}
            for comp, weight in zip(self.components, self.weights):
                name = comp.__class__.__name__
                value = component_scores[name][i]
                detail[name] = value
                total += float(weight) * float(value)
            totals.append(float(total))
            details.append(detail)

        component_means = {k: float(np.mean(v)) for k, v in component_scores.items()}
        component_stds = {k: float(np.std(v)) for k, v in component_scores.items()}

        return {
            'totals': totals,
            'details': details,
            'component_means': component_means,
            'component_stds': component_stds,
            'component_cache_stats': component_cache_stats,
            'component_scores': component_scores,
        }

    def normalize_group(self, rewards, eps=1e-6):
        arr = np.array(rewards, dtype=np.float32)
        mean = float(arr.mean())
        std = float(arr.std())
        norm = ((arr - mean) / (std + eps)).tolist()
        return norm, {'mean': mean, 'std': std}

    def normalize_components(self, component_scores, eps=1e-6):
        out = {}
        stats = {}
        for k, v in component_scores.items():
            arr = np.array(v, dtype=np.float32)
            mean = float(arr.mean())
            std = float(arr.std())
            out[k] = ((arr - mean) / (std + eps)).tolist()
            stats[k] = {'mean': mean, 'std': std}
        return out, stats

    def get_component_cache_stats(self):
        out = {}
        for comp in self.components:
            if hasattr(comp, 'get_cache_stats'):
                out[comp.__class__.__name__] = comp.get_cache_stats()
        return out

    def reset_component_caches(self, clear_values=False):
        for comp in self.components:
            if hasattr(comp, 'reset_cache'):
                comp.reset_cache(clear_values=clear_values)





