import argparse

def get_args_parser():
    parser = argparse.ArgumentParser(description='MORL MotionLLM training',
                                     add_help=True,
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    ## device
    parser.add_argument('--device', type=str, default='cuda:0', help='device')

    ## MotionLLM training
    parser.add_argument('--learning-rate', type=float, default=1e-5, help='learning rate')
    parser.add_argument('--batch-size', type=int, default=6, help='batch size')
    parser.add_argument('--epochs-t2m', type=int, default=500, help='number of epochs for t2m')
    parser.add_argument('--epochs-m2t', type=int, default=10, help='number of epochs for m2t')
    parser.add_argument('--training-task', type=str, default='t2m', help='training task, t2m or m2t')
    parser.add_argument('--epochs-start-val', type=int, default=70, help='number of epochs to start validation')
    parser.add_argument('--epochs-val-interval', type=int, default=3, help='number of epochs between validation')
    parser.add_argument('--train-stage', type=str, default='sft', choices=['sft', 'rlvr'], help='training stage selector')
    parser.add_argument('--use-reasoning', action='store_true', help='enable chain-of-thought reasoning supervision')
    parser.add_argument('--reasoning-weight', type=float, default=1.0, help='loss weight for reasoning tokens')
    parser.add_argument('--motion-weight', type=float, default=1.0, help='loss weight for motion tokens')
    parser.add_argument('--use-sample-weight', action='store_true', help='enable per-sample loss weighting')
    parser.add_argument('--reward-clip', type=float, default=5.0, help='clip value for reward-based sample weights')
    parser.add_argument('--normalize-reward', action='store_true', help='normalize sample rewards within a batch')
    parser.add_argument('--min-sample-weight', type=float, default=0.05, help='minimum sample weight after transforms')
    parser.add_argument('--rl-epochs', type=int, default=3, help='epochs for RLVR stage')
    parser.add_argument('--rl-group-size', type=int, default=8, help='group size for GRPO style rollouts')
    parser.add_argument('--rl-kl-coef', type=float, default=0.02, help='KL regularization coefficient (β in r - log(r) - 1 penalty)')
    parser.add_argument('--rl-clip-eps', type=float, default=0.2, help='PPO-style clipping epsilon for GRPO importance ratio (ε in clip(r, 1-ε, 1+ε))')
    parser.set_defaults(rl_component_norm=True)
    parser.add_argument('--disable-rl-component-norm', dest='rl_component_norm', action='store_false', help='disable component-wise reward normalization and use total reward normalization only')
    parser.add_argument('--rl-reference-ckpt', type=str, default=None, help='checkpoint loaded before RLVR stage')
    parser.add_argument('--com-candidates', type=int, default=8, help='number of candidates for Chain-of-Motion decoding')
    parser.add_argument('--com-refine-steps', type=int, default=2, help='number of Chain-of-Motion refinement steps')
    parser.add_argument('--reward-format-weight', type=float, default=0.2, help='weight for format validity reward')
    parser.add_argument('--reward-sem-weight', type=float, default=0.5, help='weight for semantic alignment reward')
    parser.add_argument('--reward-coh-weight', type=float, default=0.8, help='weight for reasoning coherence reward')
    parser.add_argument('--reward-phys-weight', type=float, default=0.5, help='weight for physical plausibility reward')
    parser.add_argument('--reward-align-weight', type=float, default=0.5, help='weight for text-motion consistency reward')
    parser.add_argument('--semantic-model-name', type=str, default='microsoft/deberta-v3-base', help='encoder model for semantic alignment reward')
    parser.add_argument('--nli-model-name', type=str, default='microsoft/deberta-v3-large-mnli', help='NLI model for reasoning coherence reward')
    parser.add_argument('--reward-device', type=str, default=None, help='device for reward models, defaults to --device when not set')
    parser.add_argument('--reward-max-batch-size', type=int, default=16, help='max batch size for reward model forward passes')
    parser.add_argument('--reward-dataset-opt-path', type=str, default='checkpoints/t2m/Comp_v6_KLD005/opt.txt', help='dataset opt path for text-motion embedding reward')
    parser.add_argument('--reward-glove-path', type=str, default='./glove', help='glove root path for text-motion embedding reward')
    parser.add_argument('--reward-reset-cache-each-epoch', action='store_true', help='reset reward cache counters at each RL epoch start')
    parser.add_argument('--reward-clear-cache-values', action='store_true', help='clear cached reward values when resetting cache counters')
    parser.add_argument('--rl-metrics-jsonl', type=str, default=None, help='optional path to write per-epoch RL metrics as jsonl')
    parser.add_argument('--max-text-len', type=int, default=256)
    parser.add_argument('--max-motion-len', type=int, default=200)
    ## LLM
    parser.add_argument('--llm-backbone', type=str, default='Qwen/Qwen3-4B-Instruct', help='name of huggingface model backbone')
    parser.add_argument('--lora-r-t2m', type=int, default=64, help='lora_r for t2m')
    parser.add_argument('--lora-alpha-t2m', type=int, default=64, help='lora_alpha for t2m')
    parser.add_argument('--lora-r-m2t', type=int, default=32, help='lora_r for m2t')
    parser.add_argument('--lora-alpha-m2t', type=int, default=32, help='lora_alpha for m2t')
    parser.add_argument('--lora-dropout', type=float, default=0.1, help='lora_dropout')

    ## dataloader
    parser.add_argument('--dataname', type=str, default='t2m', help='dataset directory')
    parser.add_argument('--cot-train-jsonl', type=str, default=None, help='path to CoT train jsonl with caption/reasoning/motion_tokens')
    parser.add_argument('--cot-task-filter', type=str, default=None, help='optional task filter for CoT jsonl: t2m or m2t')
    parser.add_argument('--cot-num-workers', type=int, default=0)

    ## vqvae arch
    parser.add_argument("--code-dim", type=int, default=512, help="embedding dimension")
    parser.add_argument("--nb-code", type=int, default=1024, help="nb of embedding (codebook size; must match the VQ-VAE checkpoint, e.g. VQVAEV3_CB1024)")
    parser.add_argument("--mu", type=float, default=0.99, help="exponential moving average to update the codebook")
    parser.add_argument("--down-t", type=int, default=2, help="downsampling rate")
    parser.add_argument("--stride-t", type=int, default=2, help="stride size")
    parser.add_argument("--width", type=int, default=512, help="width of the network")
    parser.add_argument("--depth", type=int, default=3, help="depth of the network")
    parser.add_argument("--dilation-growth-rate", type=int, default=3, help="dilation growth rate")
    parser.add_argument("--output-emb-width", type=int, default=512, help="output embedding width")
    parser.add_argument('--vq-act', type=str, default='relu', choices = ['relu', 'silu', 'gelu'], help='dataset directory')
    parser.add_argument('--vq-norm', type=str, default=None, help='dataset directory')

    ## quantizer
    parser.add_argument("--quantizer", type=str, default='ema_reset', choices = ['ema', 'orig', 'ema_reset', 'reset'], help="eps for optimal transport")
    parser.add_argument('--beta', type=float, default=1.0, help='commitment loss in standard VQ')


    ## output directory
    parser.add_argument('--out-dir', type=str, default='experiments', help='output directory')
    parser.add_argument('--exp-name', type=str, default='test', help='name of the experiment, will create a file inside out-dir')

    ## other
    parser.add_argument('--print-iter', default=200, type=int, help='print frequency')
    parser.add_argument('--eval-iter', default=1000, type=int, help='evaluation frequency')
    parser.add_argument('--seed', default=123, type=int, help='seed for initializing training.')


    return parser.parse_args()