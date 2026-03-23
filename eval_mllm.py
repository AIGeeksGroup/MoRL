from options.option_llm import get_args_parser
from models.mllm import MotionLLM
import torch
from utils.evaluation import evaluation_test
from dataset import dataset_TM_eval
from dataset import dataset_m2t_eval
from utils.word_vectorizer import WordVectorizer
from models.evaluator_wrapper import EvaluatorModelWrapper
from options.get_eval_option import get_opt
from utils.m2t_metrics import compute_m2t_metrics
from rewards import build_task_reward, build_reward_config_from_args
import numpy as np
import os

os.environ['TOKENIZERS_PARALLELISM'] = 'false'


def eval_t2m():
    args = get_args_parser()

    args.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    model = MotionLLM(args)
    model.load_model(args.eval_ckpt)
    model = model.to(args.device)
    model.eval()

    if getattr(args, 'use_com', False):
        reward_cfg = build_reward_config_from_args(args)
        com_reward = build_task_reward('t2m', reward_cfg)

        def _generate_with_com(caption):
            out = model.generate_com(
                caption=caption,
                task='t2m',
                k=int(args.com_candidates),
                t=int(args.com_refine_steps),
                reward_fn=com_reward,
            )
            return out['motion_tokens']
        model.generate = _generate_with_com

    glove_path = os.path.abspath('../LLM-MotionGen/glove')
    dataset_opt_path = os.path.abspath('checkpoints/t2m/Comp_v6_KLD005/opt.txt')

    w_vectorizer = WordVectorizer(glove_path, 'our_vab')

    args.dataname = 't2m'
    wrapper_opt = get_opt(dataset_opt_path, args.device)
    eval_wrapper = EvaluatorModelWrapper(wrapper_opt)

    if not hasattr(args, 'down_t'):
        raise ValueError("args.down_t is required but not found")

    test_loader = dataset_TM_eval.DATALoader(
        args.dataname,
        "test",
        32,
        w_vectorizer,
        unit_length=2 ** args.down_t
    )

    fid, div, top1, top2, top3, matching, multi = [], [], [], [], [], [], []

    repeat_time = 20

    with torch.no_grad():
        for _ in range(repeat_time):
            best_fid, best_div, best_top1, best_top2, best_top3, best_matching, best_multi = evaluation_test(
                args.out_dir,
                test_loader,
                model,
                eval_wrapper=eval_wrapper,
                draw=False,
                savenpy=False
            )

            fid.append(best_fid)
            div.append(best_div)
            top1.append(best_top1)
            top2.append(best_top2)
            top3.append(best_top3)
            matching.append(best_matching)
            multi.append(best_multi)

    print('final result:')
    print('fid: ', sum(fid) / repeat_time)
    print('div: ', sum(div) / repeat_time)
    print('top1: ', sum(top1) / repeat_time)
    print('top2: ', sum(top2) / repeat_time)
    print('top3: ', sum(top3) / repeat_time)
    print('matching: ', sum(matching) / repeat_time)
    print('multi: ', sum(multi) / repeat_time)

    fid = np.array(fid)
    div = np.array(div)
    top1 = np.array(top1)
    top2 = np.array(top2)
    top3 = np.array(top3)
    matching = np.array(matching)
    multi = np.array(multi)

    msg_final = (
        f"FID. {np.mean(fid):.3f}, conf. {np.std(fid) * 1.96 / np.sqrt(repeat_time):.3f}, "
        f"Diversity. {np.mean(div):.3f}, conf. {np.std(div) * 1.96 / np.sqrt(repeat_time):.3f}, "
        f"TOP1. {np.mean(top1):.3f}, conf. {np.std(top1) * 1.96 / np.sqrt(repeat_time):.3f}, "
        f"TOP2. {np.mean(top2):.3f}, conf. {np.std(top2) * 1.96 / np.sqrt(repeat_time):.3f}, "
        f"TOP3. {np.mean(top3):.3f}, conf. {np.std(top3) * 1.96 / np.sqrt(repeat_time):.3f}, "
        f"Matching. {np.mean(matching):.3f}, conf. {np.std(matching) * 1.96 / np.sqrt(repeat_time):.3f}, "
        f"Multi. {np.mean(multi):.3f}, conf. {np.std(multi) * 1.96 / np.sqrt(repeat_time):.3f}"
    )

    print(msg_final)


def eval_m2t():
    args = get_args_parser()
    args.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    model = MotionLLM(args)
    model.load_model(args.eval_ckpt)
    model = model.to(args.device)
    model.eval()

    com_reward = None
    if getattr(args, 'use_com', False):
        reward_cfg = build_reward_config_from_args(args)
        com_reward = build_task_reward('m2t', reward_cfg)

    loader = dataset_m2t_eval.DATALoader(
        dataset_name=str(getattr(args, 'dataname', 't2m')),
        split=str(getattr(args, 'm2t_eval_split', 'test')),
        batch_size=int(getattr(args, 'm2t_eval_batch_size', 8)),
        num_workers=int(getattr(args, 'm2t_eval_num_workers', 0)),
        shuffle=False,
    )

    preds = []
    refs = []
    for batch in loader:
        motions, m_lengths, batch_refs, _ = batch
        for motion, m_len, one_refs in zip(motions, m_lengths, batch_refs):
            motion_np = motion[:int(m_len)].numpy()
            if getattr(args, 'use_com', False):
                tokens = model.net.encode(
                    torch.from_numpy(model.normalize(motion_np)).float().to(args.device).unsqueeze(0)
                ).squeeze(0)
                tokens = tokens + model.motion_token_start

                ref_caption = one_refs[0] if len(one_refs) > 0 else ''

                out = model.generate_com(
                    caption=ref_caption,
                    motion_tokens=tokens,
                    task='m2t',
                    k=int(args.com_candidates),
                    t=int(args.com_refine_steps),
                    reward_fn=com_reward,
                )
                pred = (out.get('answer') or '').strip()
            else:
                pred = str(model.caption(motion_np)).strip()

            preds.append(pred)
            refs.append(list(one_refs))

    metrics = compute_m2t_metrics(preds, refs)
    print('final m2t result:')
    for k in ['BLEU@1', 'BLEU@4', 'ROUGE-L', 'CIDEr', 'BERTScore']:
        print(f'{k}: {metrics[k]:.4f}')


if __name__ == "__main__":
    args = get_args_parser()
    task = str(getattr(args, 'eval_task', 't2m'))
    if task == 'm2t':
        eval_m2t()
    else:
        eval_t2m()
