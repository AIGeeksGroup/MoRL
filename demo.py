from openai import AzureOpenAI
from models.motion_agent import MotionAgent
from models.mllm import MotionLLM
from options.option_llm import get_args_parser
from utils.motion_utils import recover_from_ric, plot_3d_motion
from utils.paramUtil import t2m_kinematic_chain
from rewards import build_task_reward, build_reward_config_from_args
import torch

def motion_agent_demo():
    # Initialize the client
    client = AzureOpenAI(
        api_key="********", # your api key
        api_version="2024-10-21",
        azure_endpoint="********" # your azure endpoint
    )

    args = get_args_parser()
    args.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    motion_agent = MotionAgent(args, client)
    motion_agent.chat()

def motionllm_demo():
    args = get_args_parser()
    args.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    model = MotionLLM(args)
    model.load_model(args.model_ckpt)
    model.llm.eval()
    model.llm.to(args.device)
    
    caption = 'A man is doing cartwheels.'
    if getattr(args, 'use_com', False):
        reward_cfg = build_reward_config_from_args(args)
        com_reward = build_task_reward('t2m', reward_cfg)
        out = model.generate_com(
            caption=caption,
            task='t2m',
            k=int(args.com_candidates),
            t=int(args.com_refine_steps),
            reward_fn=com_reward,
        )
        motion = out['motion_tokens']
    else:
        motion = model.generate(caption)

    motion = model.net.forward_decoder(motion)
    motion = model.denormalize(motion.detach().cpu().numpy())
    motion = recover_from_ric(torch.from_numpy(motion).float().to(args.device), 22)
    print(motion.shape)
    plot_3d_motion(f"motionllm_demo.mp4", t2m_kinematic_chain, motion.squeeze().detach().cpu().numpy(), title=caption, fps=20, radius=4)

if __name__ == "__main__":
    motion_agent_demo()
