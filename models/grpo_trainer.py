import torch
import copy

from rl.advantage import group_normalized_advantages
from rl.rollout import rollout_group


class GRPOTrainer:
    def __init__(self, model, optimizer, reward_fn, args, device):
        self.model = model
        self.optimizer = optimizer
        self.reward_fn = reward_fn
        self.args = args
        self.device = device
        self.reference_model = copy.deepcopy(model)
        self.reference_model.eval()
        for p in self.reference_model.parameters():
            p.requires_grad = False

    # ------------------------------------------------------------------
    # Helper: compute (sum_logprob, num_tokens) for one candidate.
    # ------------------------------------------------------------------
    def _get_logprob(self, model, task, example, cand):
        """
        Returns (sum_logprob: Tensor, num_tokens: int).
        Delegates to model.compute_per_token_logprob which keeps grad.
        """
        if task == 't2m':
            return model.compute_per_token_logprob(
                [example['caption']],
                [cand['motion_tokens']],
                reasoning=[cand.get('reasoning')],
                task=task,
            )
        else:  # m2t
            return model.compute_per_token_logprob(
                [cand.get('answer', '')],
                [example['motion_tokens']],
                reasoning=[cand.get('reasoning')],
                task=task,
            )

    # ------------------------------------------------------------------
    # Main training step
    # ------------------------------------------------------------------
    def train_batch(self, examples, task='t2m'):
        all_losses = []
        all_rewards = []
        reward_means = []
        reward_stds = []
        component_mean_acc = {}
        component_std_acc = {}
        component_count = {}
        latest_cache_stats = {}

        # Hyper-parameters for GRPO
        clip_eps = float(getattr(self.args, 'rl_clip_eps', 0.2))
        kl_coef  = float(getattr(self.args, 'rl_kl_coef',  0.02))

        for example in examples:

            # ---- 1. Rollout: generate a group of G candidates ----
            candidates = rollout_group(
                self.model,
                example,
                group_size=int(self.args.rl_group_size),
                com_steps=int(self.args.com_refine_steps),
                task=task,
            )

            # ---- 2. Score rewards ----
            group_out = (
                self.reward_fn.score_group(candidates)
                if hasattr(self.reward_fn, 'score_group')
                else None
            )
            if group_out is not None:
                rewards = [float(x) for x in group_out['totals']]
                for k, v in group_out.get('component_means', {}).items():
                    component_mean_acc[k] = component_mean_acc.get(k, 0.0) + float(v)
                    component_count[k] = component_count.get(k, 0) + 1
                for k, v in group_out.get('component_stds', {}).items():
                    component_std_acc[k] = component_std_acc.get(k, 0.0) + float(v)
                for k, v in group_out.get('component_cache_stats', {}).items():
                    latest_cache_stats[k] = {
                        'hits':     int(v.get('hits', 0)),
                        'misses':   int(v.get('misses', 0)),
                        'hit_rate': float(v.get('hit_rate', 0.0)),
                        'size':     float(v.get('size', 0.0)),
                    }
            else:
                rewards = [float(self.reward_fn(item)[0]) for item in candidates]

            # ---- 3. Compute group-normalised advantages ----
            if hasattr(self.reward_fn, 'normalize_group'):
                if bool(getattr(self.args, 'rl_component_norm', True)) and group_out is not None:
                    comp_scores = group_out.get('component_scores', {})
                    comp_norm, _ = self.reward_fn.normalize_components(comp_scores)
                    advantages = []
                    for i in range(len(rewards)):
                        adv = sum(
                            float(w) * float(comp_norm[comp.__class__.__name__][i])
                            for comp, w in zip(self.reward_fn.components, self.reward_fn.weights)
                        )
                        advantages.append(adv)
                    stats = {
                        'mean': float(sum(rewards) / max(1, len(rewards))),
                        'std':  float(
                            torch.tensor(rewards, dtype=torch.float32)
                            .std(unbiased=False).item()
                        ) if len(rewards) > 0 else 0.0,
                    }
                else:
                    advantages, stats = self.reward_fn.normalize_group(rewards)
                reward_means.append(float(stats['mean']))
                reward_stds.append(float(stats['std']))
            else:
                advantages = group_normalized_advantages(rewards)

            # ---- 4. GRPO clipped policy-gradient update (group-level) ----
            #
            # Algorithm (per candidate i):
            #   log_ratio_i = mean_token_logp(policy) - mean_token_logp(ref)
            #   ratio_i     = exp(log_ratio_i)                         # ≈ π_θ / π_ref
            #   pg_loss_i   = -min(ratio_i * A_i,
            #                      clip(ratio_i, 1-ε, 1+ε) * A_i)
            #   kl_i        = ratio_i - log_ratio_i - 1                # Schulman unbiased, ≥ 0
            #   loss_i      = pg_loss_i + β * kl_i
            #
            # Gradients are **accumulated** across all candidates; a single
            # optimizer.step() is called per group for correct GRPO semantics.

            n_cands = max(1, len(candidates))
            self.optimizer.zero_grad()   # ← ONE zero_grad per group
            group_loss_accum = 0.0
            valid_count = 0

            for cand, adv in zip(candidates, advantages):

                # Policy log-prob (gradient flows through this)
                policy_sum_lp, num_tokens = self._get_logprob(
                    self.model, task, example, cand
                )
                if num_tokens == 0:
                    continue

                # Reference log-prob (no gradient)
                with torch.no_grad():
                    ref_sum_lp, _ = self._get_logprob(
                        self.reference_model, task, example, cand
                    )
                    ref_sum_lp = ref_sum_lp.detach()

                # Per-token mean log-prob for numerically stable ratio.
                # ratio ≈ exp(mean_token_log π_θ - mean_token_log π_ref)
                policy_mean_lp = policy_sum_lp / num_tokens
                ref_mean_lp    = ref_sum_lp    / num_tokens

                log_ratio = policy_mean_lp - ref_mean_lp
                ratio     = torch.exp(log_ratio.clamp(-20.0, 20.0))

                adv_val = torch.tensor(
                    float(adv), device=self.device, dtype=ratio.dtype
                )

                # Clipped surrogate objective (PPO-style)
                surr1    = ratio * adv_val
                surr2    = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * adv_val
                pg_loss  = -torch.min(surr1, surr2)

                # KL penalty: r - log(r) - 1  (≥ 0, no grad needed for KL itself)
                kl_pen   = (ratio.detach() - log_ratio.detach() - 1.0).clamp(min=0.0)

                # Normalise by group size so the effective LR is independent of G
                cand_loss = (pg_loss + kl_coef * kl_pen) / n_cands
                cand_loss.backward()   # accumulate gradients

                group_loss_accum += float(cand_loss.detach().cpu().item())
                valid_count += 1

            if valid_count > 0:
                self.optimizer.step()   # ← ONE step per group

            all_losses.append(group_loss_accum)
            all_rewards.extend(rewards)

        # ---- Return metrics ----
        if len(all_losses) == 0:
            return {
                'loss': 0.0, 'reward': 0.0,
                'reward_mean': 0.0, 'reward_std': 0.0,
                'component_means': {}, 'component_stds': {},
                'component_cache_stats': {},
            }

        component_means = {
            k: float(v / max(1, component_count.get(k, 1)))
            for k, v in component_mean_acc.items()
        }
        component_stds = {
            k: float(v / max(1, component_count.get(k, 1)))
            for k, v in component_std_acc.items()
        }

        return {
            'loss':             float(sum(all_losses) / len(all_losses)),
            'reward':           float(sum(all_rewards) / len(all_rewards)),
            'reward_mean':      float(sum(reward_means) / len(reward_means)) if reward_means else 0.0,
            'reward_std':       float(sum(reward_stds)  / len(reward_stds))  if reward_stds  else 0.0,
            'component_means':  component_means,
            'component_stds':   component_stds,
            'component_cache_stats': latest_cache_stats,
        }

