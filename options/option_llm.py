import argparse


def get_args_parser():
    parser = argparse.ArgumentParser(
        description='MORL Motion LLM Training',
        add_help=True,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # ---------------- device ----------------
    parser.add_argument('--device', type=str, default='cuda:0')

    # ---------------- LLM ----------------
    parser.add_argument('--llm-backbone', type=str, default='Qwen/Qwen3-4B-Instruct')

    parser.add_argument('--lora-r-t2m', type=int, default=64)
    parser.add_argument('--lora-alpha-t2m', type=int, default=64)

    parser.add_argument('--lora-r-m2t', type=int, default=32)
    parser.add_argument('--lora-alpha-m2t', type=int, default=32)

    parser.add_argument('--lora-dropout', type=float, default=0.1)

    # ---------------- MORL (NEW) ----------------
    parser.add_argument(
        '--use-reasoning',
        action='store_true',
        help='train with chain-of-thought reasoning supervision'
    )

    parser.add_argument(
        '--reasoning-weight',
        type=float,
        default=1.0,
        help='loss weight for reasoning tokens'
    )

    parser.add_argument(
        '--motion-weight',
        type=float,
        default=1.0,
        help='loss weight for motion tokens'
    )

    parser.add_argument(
        '--max-text-len',
        type=int,
        default=256
    )

    parser.add_argument(
        '--max-motion-len',
        type=int,
        default=200
    )

    parser.add_argument('--use-com', action='store_true')
    parser.add_argument('--com-candidates', type=int, default=8)
    parser.add_argument('--com-refine-steps', type=int, default=2)
    parser.add_argument('--model-ckpt', type=str, default='ckpt/motionllm.pth')
    parser.add_argument('--save-dir', type=str, default='./demo')
    parser.add_argument('--eval-ckpt', type=str, default='ckpt/motionllm.pth')
    parser.add_argument('--eval-task', type=str, default='t2m', choices=['t2m', 'm2t'])
    parser.add_argument('--m2t-eval-split', type=str, default='test')
    parser.add_argument('--m2t-eval-batch-size', type=int, default=8)
    parser.add_argument('--m2t-eval-num-workers', type=int, default=0)

    parser.add_argument('--reward-format-weight', type=float, default=0.2)
    parser.add_argument('--reward-sem-weight', type=float, default=0.5)
    parser.add_argument('--reward-coh-weight', type=float, default=0.8)
    parser.add_argument('--reward-phys-weight', type=float, default=0.5)
    parser.add_argument('--reward-align-weight', type=float, default=0.5)
    parser.add_argument('--semantic-model-name', type=str, default='microsoft/deberta-v3-base')
    parser.add_argument('--nli-model-name', type=str, default='microsoft/deberta-v3-large-mnli')
    parser.add_argument('--reward-device', type=str, default=None)
    parser.add_argument('--reward-max-batch-size', type=int, default=16)
    parser.add_argument('--reward-dataset-opt-path', type=str, default='checkpoints/t2m/Comp_v6_KLD005/opt.txt')
    parser.add_argument('--reward-glove-path', type=str, default='./glove')

    # ---------------- dataloader ----------------
    parser.add_argument('--dataname', type=str, default='kit')

    parser.add_argument('--batch-size', type=int, default=128)

    parser.add_argument('--window-size', type=int, default=64)

    # ---------------- optimization ----------------
    parser.add_argument('--total-iter', type=int, default=200000)

    parser.add_argument('--warm-up-iter', type=int, default=1000)

    parser.add_argument('--lr', type=float, default=2e-4)

    parser.add_argument(
        '--lr-scheduler',
        default=[50000, 400000],
        nargs="+",
        type=int
    )

    parser.add_argument('--gamma', type=float, default=0.05)

    parser.add_argument('--weight-decay', type=float, default=0.01)

    parser.add_argument("--commit", type=float, default=0.02)

    parser.add_argument('--loss-vel', type=float, default=0.1)

    parser.add_argument('--recons-loss', type=str, default='l2')

    # ---------------- VQ-VAE ----------------
    parser.add_argument("--code-dim", type=int, default=512)

    parser.add_argument("--nb-code", type=int, default=512)

    parser.add_argument("--mu", type=float, default=0.99)

    parser.add_argument("--down-t", type=int, default=2)

    parser.add_argument("--stride-t", type=int, default=2)

    parser.add_argument("--width", type=int, default=512)

    parser.add_argument("--depth", type=int, default=3)

    parser.add_argument("--dilation-growth-rate", type=int, default=3)

    parser.add_argument("--output-emb-width", type=int, default=512)

    parser.add_argument(
        '--vq-act',
        type=str,
        default='relu',
        choices=['relu', 'silu', 'gelu']
    )

    parser.add_argument('--vq-norm', type=str, default=None)

    # ---------------- quantizer ----------------
    parser.add_argument(
        "--quantizer",
        type=str,
        default='ema_reset',
        choices=['ema', 'orig', 'ema_reset', 'reset']
    )

    parser.add_argument('--beta', type=float, default=1.0)

    # ---------------- resume ----------------
    parser.add_argument("--resume-pth", type=str, default=None)

    parser.add_argument("--resume-gpt", type=str, default=None)

    # ---------------- output ----------------
    parser.add_argument('--out-dir', type=str, default='experiments')

    parser.add_argument('--results-dir', type=str, default='visual_results/')

    parser.add_argument('--visual-name', type=str, default='baseline')

    parser.add_argument('--exp-name', type=str, default='exp_debug')

    # ---------------- logging ----------------
    parser.add_argument('--print-iter', type=int, default=200)

    parser.add_argument('--eval-iter', type=int, default=1000)

    parser.add_argument('--seed', type=int, default=123)

    # ---------------- visualization ----------------
    parser.add_argument('--vis-gt', action='store_true')

    parser.add_argument('--nb-vis', type=int, default=20)

    return parser.parse_args()