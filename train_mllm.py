import torch
import numpy as np
import os
import logging
import json
import sys

from dataset import dataset_TM_eval
from dataset import dataset_cot
from utils.evaluation import evaluation_test
from utils.word_vectorizer import WordVectorizer
from models.evaluator_wrapper import EvaluatorModelWrapper
from options.get_eval_option import get_opt
from models.mllm import MotionLLM
from options.option_train import get_args_parser
from rewards import (
    build_task_reward,
    build_reward_config_from_args,
)
from rl import GRPOTrainer

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def get_logger(out_dir):
    logger = logging.getLogger('Exp')
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")

    file_path = os.path.join(out_dir, "run.log")
    file_hdlr = logging.FileHandler(file_path)
    file_hdlr.setFormatter(formatter)

    strm_hdlr = logging.StreamHandler(sys.stdout)
    strm_hdlr.setFormatter(formatter)

    logger.addHandler(file_hdlr)
    logger.addHandler(strm_hdlr)
    return logger


def encode_motion_tokens(model, motion, m_length, device):
    """
    Convert motion to motion tokens using VQ-VAE
    """
    motion_tokens = []

    for i in range(motion.shape[0]):

        tokens = model.net.encode(
            motion[i:i + 1, :m_length[i], :].to(device)
        ).squeeze(0)

        for j in range(tokens.shape[0]):
            tokens[j] = model.motion_token_indices[tokens[j]]

        motion_tokens.append(tokens)

    return motion_tokens


def unpack_train_batch(batch, args, model):
    if isinstance(batch, dict):
        captions = batch["caption"]
        motion_tokens = [x.to(args.device) for x in batch["motion_tokens"]]
        reasoning = batch.get("reasoning")
        sample_weights = None
        if getattr(args, "use_sample_weight", False):
            sample_weights = build_sample_weights(batch, args)
        if not getattr(args, "use_reasoning", False):
            reasoning = None
        return captions, motion_tokens, reasoning, sample_weights

    reasoning = None
    if len(batch) == 8:
        (
            _,
            _,
            caption,
            _,
            motion,
            m_length,
            _,
            _
        ) = batch
    elif len(batch) == 9:
        (
            _,
            _,
            caption,
            _,
            motion,
            m_length,
            _,
            _,
            reasoning
        ) = batch
    else:
        raise ValueError(f"Unexpected batch format with {len(batch)} fields")

    motion_tokens = encode_motion_tokens(model, motion, m_length, args.device)
    if not getattr(args, "use_reasoning", False):
        reasoning = None
    return caption, motion_tokens, reasoning, None


def build_sample_weights(batch, args):
    explicit = batch.get("sample_weight")
    rewards = batch.get("reward")
    advantages = batch.get("advantage")

    values = []
    source = explicit if explicit is not None else rewards
    if source is None:
        source = advantages

    if source is None:
        return None

    for x in source:
        if x is None:
            values.append(1.0)
        else:
            values.append(float(x))

    weights = np.array(values, dtype=np.float32)

    clip_val = float(getattr(args, "reward_clip", 0.0))
    if clip_val > 0:
        weights = np.clip(weights, -clip_val, clip_val)

    if getattr(args, "normalize_reward", False):
        mean = float(weights.mean())
        std = float(weights.std())
        if std > 1e-6:
            weights = (weights - mean) / std

    min_w = float(getattr(args, "min_sample_weight", 0.05))
    weights = np.maximum(weights, min_w)
    return weights.tolist()


def train_rlvr_stage(args, model, train_loader, optimizer, logger):
    reward_device = str(args.reward_device) if getattr(args, 'reward_device', None) else str(args.device)
    training_task = str(getattr(args, 'training_task', 't2m'))
    reward_cfg = build_reward_config_from_args(args, reward_device=reward_device)
    reward = build_task_reward(training_task, reward_cfg)

    trainer = GRPOTrainer(
        model=model,
        optimizer=optimizer,
        reward_fn=reward,
        args=args,
        device=args.device
    )

    prev_cache_snapshot = {}

    metrics_jsonl_path = args.rl_metrics_jsonl
    if metrics_jsonl_path is not None:
        metrics_jsonl_path = str(metrics_jsonl_path)
        metrics_parent = os.path.dirname(metrics_jsonl_path)
        if metrics_parent:
            os.makedirs(metrics_parent, exist_ok=True)

    for epoch in range(int(args.rl_epochs)):
        if getattr(args, 'reward_reset_cache_each_epoch', False):
            reward.reset_component_caches(clear_values=bool(getattr(args, 'reward_clear_cache_values', False)))

        metrics = []
        for batch in train_loader:
            captions, motion_tokens, _, _ = unpack_train_batch(batch, args, model)
            if training_task == 'm2t':
                examples = [
                    {
                        'caption': caption,
                        'motion_tokens': tokens,
                    }
                    for caption, tokens in zip(captions, motion_tokens)
                ]
            else:
                examples = [{'caption': caption} for caption in captions]

            out = trainer.train_batch(examples, task=training_task)
            metrics.append(out)

        if len(metrics) > 0:
            loss_val = float(np.mean([m['loss'] for m in metrics]))
            reward_val = float(np.mean([m['reward'] for m in metrics]))
            reward_mean = float(np.mean([m['reward_mean'] for m in metrics]))
            reward_std = float(np.mean([m['reward_std'] for m in metrics]))
            comp_pool = {}
            comp_std_pool = {}
            for m in metrics:
                for k, v in m.get('component_means', {}).items():
                    comp_pool.setdefault(k, []).append(float(v))
                for k, v in m.get('component_stds', {}).items():
                    comp_std_pool.setdefault(k, []).append(float(v))
            comp_means = {k: float(np.mean(v)) for k, v in comp_pool.items()}
            comp_stds = {k: float(np.mean(v)) for k, v in comp_std_pool.items()}
            comp_cache = reward.get_component_cache_stats() if hasattr(reward, 'get_component_cache_stats') else {}

            comp_cache_delta = {}
            for k, cur in comp_cache.items():
                prev = prev_cache_snapshot.get(k, {'hits': 0, 'misses': 0})
                delta_hits = int(cur.get('hits', 0)) - int(prev.get('hits', 0))
                delta_misses = int(cur.get('misses', 0)) - int(prev.get('misses', 0))
                delta_total = delta_hits + delta_misses
                comp_cache_delta[k] = {
                    'hits': delta_hits,
                    'misses': delta_misses,
                    'hit_rate': float(delta_hits / delta_total) if delta_total > 0 else 0.0,
                    'size': int(cur.get('size', 0)),
                }
            prev_cache_snapshot = {k: dict(v) for k, v in comp_cache.items()}
        else:
            loss_val = 0.0
            reward_val = 0.0
            reward_mean = 0.0
            reward_std = 0.0
            comp_means = {}
            comp_stds = {}
            comp_cache = {}
            comp_cache_delta = {}

        logger.info(
            f'RL Epoch {epoch}, '
            f'PolicyLoss: {loss_val}, '
            f'Reward: {reward_val}, '
            f'RewardMean: {reward_mean}, '
            f'RewardStd: {reward_std}'
        )

        if len(comp_means) > 0:
            logger.info(f'RL Epoch {epoch}, ComponentMeans: {json.dumps(comp_means)}')
        if len(comp_stds) > 0:
            logger.info(f'RL Epoch {epoch}, ComponentStds: {json.dumps(comp_stds)}')
        if len(comp_cache) > 0:
            logger.info(f'RL Epoch {epoch}, ComponentCache: {json.dumps(comp_cache)}')
        if len(comp_cache_delta) > 0:
            logger.info(f'RL Epoch {epoch}, ComponentCacheDelta: {json.dumps(comp_cache_delta)}')

        if metrics_jsonl_path is not None:
            row = {
                'epoch': int(epoch),
                'policy_loss': float(loss_val),
                'reward': float(reward_val),
                'reward_mean': float(reward_mean),
                'reward_std': float(reward_std),
                'component_means': comp_means,
                'component_stds': comp_stds,
                'component_cache': comp_cache,
                'component_cache_delta': comp_cache_delta,
            }
            with open(metrics_jsonl_path, 'a', encoding='utf-8') as f:
                f.write(json.dumps(row, ensure_ascii=True) + '\n')

        model.save_model(
            os.path.join(args.out_dir, f'motionllm_rlvr_epoch_{epoch}.pth')
        )


if __name__ == "__main__":

    args = get_args_parser()

    model = MotionLLM(args)
    model.train()

    # logging
    args.out_dir = os.path.join(args.out_dir, f'{args.exp_name}')
    os.makedirs(args.out_dir, exist_ok=True)

    logger = get_logger(args.out_dir)
    logger.info(json.dumps(vars(args), indent=4, sort_keys=True))

    use_cot_loader = args.cot_train_jsonl is not None

    if use_cot_loader:
        train_loader = dataset_cot.DATALoader(
            args.cot_train_jsonl,
            args.batch_size,
            task=args.cot_task_filter,
            num_workers=args.cot_num_workers,
            shuffle=True
        )
        val_loader = None
        eval_wrapper = None
    else:
        w_vectorizer = WordVectorizer('./glove', 'our_vab')
        args.dataname = 't2m'

        dataset_opt_path = 'checkpoints/t2m/Comp_v6_KLD005/opt.txt'
        wrapper_opt = get_opt(dataset_opt_path, args.device)
        eval_wrapper = EvaluatorModelWrapper(wrapper_opt)

        val_loader = dataset_TM_eval.DATALoader(
            args.dataname,
            "val",
            32,
            w_vectorizer,
            unit_length=2 ** args.down_t
        )

        train_loader = dataset_TM_eval.DATALoader(
            args.dataname,
            "train",
            args.batch_size,
            w_vectorizer,
            unit_length=2 ** args.down_t
        )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.learning_rate
    )

    if args.train_stage == 'rlvr':
        model.training_task = args.training_task
        if args.rl_reference_ckpt is not None:
            model.load_model(args.rl_reference_ckpt)
        train_rlvr_stage(args, model, train_loader, optimizer, logger)
        sys.exit(0)

    if args.training_task == 't2m':

        model.training_task = 't2m'
        best_fid = 1000

        for epoch in range(args.epochs_t2m):

            batch_losses = []
            batch_accs = []
            batch_sample_weight = []

            for batch in train_loader:
                caption, motion_tokens, reasoning, sample_weights = unpack_train_batch(batch, args, model)

                optimizer.zero_grad()

                loss, gen_acc, output, labels = model.forward(
                    caption,
                    motion_tokens,
                    reasoning=reasoning,
                    sample_weights=sample_weights
                )

                loss.backward()

                optimizer.step()

                batch_losses.append(loss.item())
                batch_accs.append(gen_acc)
                if sample_weights is not None:
                    batch_sample_weight.extend(sample_weights)

            if len(batch_sample_weight) > 0:
                logger.info(
                    f'Epoch {epoch}, '
                    f'Loss: {np.mean(batch_losses)}, '
                    f'Accuracy: {np.mean(batch_accs)}, '
                    f'SampleWeightMean: {np.mean(batch_sample_weight)}'
                )
            else:
                logger.info(
                    f'Epoch {epoch}, '
                    f'Loss: {np.mean(batch_losses)}, '
                    f'Accuracy: {np.mean(batch_accs)}'
                )

            model.save_model(
                os.path.join(
                    args.out_dir,
                    f'motionllm_t2m_latest.pth'
                )
            )

            if (
                not use_cot_loader
                and val_loader is not None
                and
                epoch > args.epochs_start_val
                and epoch % args.epochs_val_interval == 0
            ):

                model.eval()

                fid, div, top1, top2, top3, matching, multi = evaluation_test(
                    args.out_dir,
                    val_loader,
                    model,
                    eval_wrapper=eval_wrapper,
                    draw=False,
                    savenpy=False
                )

                model.train()

                logger.info(
                    f'Epoch [{epoch}/{args.epochs_t2m}], '
                    f'FID: {fid}, '
                    f'Div: {div}, '
                    f'Top1: {top1}, '
                    f'Top2: {top2}, '
                    f'Top3: {top3}, '
                    f'Matching: {matching}, '
                    f'Multi: {multi}'
                )

                if fid < best_fid:

                    best_fid = fid

                    model.save_model(
                        os.path.join(
                            args.out_dir,
                            f'motionllm_t2m_best.pth'
                        )
                    )

                    logger.info(f'Best FID: {best_fid}')

    elif args.training_task == 'm2t':

        model.load_model(
            os.path.join(
                args.out_dir,
                f'motionllm_t2m_best.pth'
            )
        )

        model.training_task = 'm2t'

        for epoch in range(args.epochs_m2t):

            batch_losses = []
            batch_accs = []
            batch_sample_weight = []

            for batch in train_loader:
                caption, motion_tokens, reasoning, sample_weights = unpack_train_batch(batch, args, model)

                optimizer.zero_grad()

                loss, gen_acc, output, labels = model.forward(
                    caption,
                    motion_tokens,
                    reasoning=reasoning,
                    sample_weights=sample_weights
                )

                loss.backward()

                optimizer.step()

                batch_losses.append(loss.item())
                batch_accs.append(gen_acc)
                if sample_weights is not None:
                    batch_sample_weight.extend(sample_weights)

            if len(batch_sample_weight) > 0:
                logger.info(
                    f'Epoch {epoch}, '
                    f'Loss: {np.mean(batch_losses)}, '
                    f'Accuracy: {np.mean(batch_accs)}, '
                    f'SampleWeightMean: {np.mean(batch_sample_weight)}'
                )
            else:
                logger.info(
                    f'Epoch {epoch}, '
                    f'Loss: {np.mean(batch_losses)}, '
                    f'Accuracy: {np.mean(batch_accs)}'
                )

            model.save_model(
                os.path.join(
                    args.out_dir,
                    f'motionllm.pth'
                )
            )