from rewards.composite_reward import CompositeReward
from rewards.motion_format_reward import MotionFormatReward
from rewards.nli_coherence_reward import NLICoherenceReward
from rewards.physical_plausibility_reward import PhysicalPlausibilityReward
from rewards.semantic_alignment_reward import SemanticAlignmentReward
from rewards.text_motion_consistency_reward import TextMotionConsistencyReward


def build_reward_config_from_args(args, reward_device=None):
    device = reward_device if reward_device is not None else getattr(args, 'reward_device', None)
    if device is None:
        device = getattr(args, 'device', 'cpu')
    return {
        'reward_device': str(device),
        'semantic_model_name': str(getattr(args, 'semantic_model_name', 'microsoft/deberta-v3-base')),
        'nli_model_name': str(getattr(args, 'nli_model_name', 'microsoft/deberta-v3-large-mnli')),
        'reward_max_batch_size': int(getattr(args, 'reward_max_batch_size', 16)),
        'reward_dataset_opt_path': str(getattr(args, 'reward_dataset_opt_path', 'checkpoints/t2m/Comp_v6_KLD005/opt.txt')),
        'reward_glove_path': str(getattr(args, 'reward_glove_path', './glove')),
        'reward_format_weight': float(getattr(args, 'reward_format_weight', 0.2)),
        'reward_sem_weight': float(getattr(args, 'reward_sem_weight', 0.5)),
        'reward_coh_weight': float(getattr(args, 'reward_coh_weight', 0.8)),
        'reward_phys_weight': float(getattr(args, 'reward_phys_weight', 0.5)),
        'reward_align_weight': float(getattr(args, 'reward_align_weight', 0.5)),
    }


def build_task_reward(task, config):
    task = str(task)
    if task == 'm2t':
        return CompositeReward(
            components=[
                SemanticAlignmentReward(
                    model_name=config['semantic_model_name'],
                    device=config['reward_device'],
                    max_batch_size=config['reward_max_batch_size'],
                ),
                NLICoherenceReward(
                    model_name=config['nli_model_name'],
                    device=config['reward_device'],
                    max_batch_size=config['reward_max_batch_size'],
                ),
            ],
            weights=[
                config['reward_sem_weight'],
                config['reward_coh_weight'],
            ],
        )

    if task == 't2m':
        return CompositeReward(
            components=[
                MotionFormatReward(),
                PhysicalPlausibilityReward(),
                TextMotionConsistencyReward(
                    device=config['reward_device'],
                    dataset_opt_path=config['reward_dataset_opt_path'],
                    glove_path=config['reward_glove_path'],
                ),
            ],
            weights=[
                config['reward_format_weight'],
                config['reward_phys_weight'],
                config['reward_align_weight'],
            ],
        )

    raise ValueError(f'Unsupported reward task: {task}')

