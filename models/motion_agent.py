from models.mllm import MotionLLM
import torch
from utils.motion_utils import recover_from_ric, plot_3d_motion
from utils.paramUtil import t2m_kinematic_chain
from rewards import build_task_reward, build_reward_config_from_args
import time
import os
import numpy as np


class MotionAgent:
    def __init__(self, args, client):
        self.args = args
        self.device = args.device

        self.model = MotionLLM(self.args)
        self.model.load_model(args.model_ckpt)
        self.model.eval()
        self.model.to(self.device)

        self.client = client
        self.save_dir = args.save_dir
        os.makedirs(self.save_dir, exist_ok=True)

        self.context = []
        self.motion_history = {}
        self.com_reward = None
        if getattr(self.args, 'use_com', False):
            reward_cfg = build_reward_config_from_args(self.args)
            self.com_reward = build_task_reward('t2m', reward_cfg)

        print("Loading example prompt from example_prompt.txt")
        self.prompt = open("example_prompt.txt", "r").read()

        self.context.append({"role": "system", "content": self.prompt})

    def parse_response(self, text):
        """
        Parse <think> and <answer> from the LLM output
        """
        reasoning = None
        answer = None

        if "<think>" in text and "</think>" in text:
            reasoning = text.split("<think>")[1].split("</think>")[0]

        if "<answer>" in text and "</answer>" in text:
            answer = text.split("<answer>")[1].split("</answer>")[0]

        return reasoning, answer

    def process_motion_dialogue(self, message):

        motion_input = None

        if 'npy' in message:
            motion_file = message.split(' ')[-1]
            assert motion_file.endswith('.npy')
            message = message.replace(motion_file, '<motion_file>')
            motion_input = np.load(motion_file)

        self.context.append({"role": "user", "content": message})

        response = self.client.chat.completions.create(
            model="gpt-4-turbo",
            messages=self.context
        )

        assistant_response = response.choices[0].message.content
        self.context.append({"role": "assistant", "content": assistant_response})

        reasoning, answer = self.parse_response(assistant_response)

        if reasoning is not None:
            print("\n[Reasoning]")
            print(reasoning)

        if answer is None:
            return

        # ---------- Motion Generation ----------
        if "generate_motion" in answer:

            descriptions = answer.split("generate_motion(")[1:]
            motion_tokens_all = []

            for desc in descriptions:
                desc = desc.split(")")[0].strip("'\"")

                if desc not in self.motion_history:
                    if getattr(self.args, 'use_com', False):
                        com_out = self.model.generate_com(
                            caption=desc,
                            task='t2m',
                            k=int(getattr(self.args, 'com_candidates', 8)),
                            t=int(getattr(self.args, 'com_refine_steps', 2)),
                            reward_fn=self.com_reward,
                        )
                        motion_tokens = com_out['motion_tokens']
                    else:
                        motion_tokens = self.model.generate(desc)
                    self.motion_history[desc] = motion_tokens
                else:
                    motion_tokens = self.motion_history[desc]

                motion_tokens_all.append(motion_tokens)

            motion_tokens = torch.cat(motion_tokens_all)

            motion = self.model.net.forward_decoder(motion_tokens)

            motion = self.model.denormalize(motion.detach().cpu().numpy())

            motion = recover_from_ric(
                torch.from_numpy(motion).float().to(self.device),
                22
            )

            filename = f"{self.save_dir}/motion_{int(time.time())}.mp4"

            print("Plotting motion...")
            plot_3d_motion(
                filename,
                t2m_kinematic_chain,
                motion.squeeze().detach().cpu().numpy(),
                title=message,
                fps=20,
                radius=4
            )

            np.save(
                f"{self.save_dir}/motion_{int(time.time())}.npy",
                motion.squeeze().detach().cpu().numpy()
            )

            print(f"Motion saved to {filename}")

        # ---------- Motion Caption ----------
        elif "caption_motion" in answer:

            if motion_input is None:
                print("No motion input provided.")
                return

            caption = self.model.caption(motion_input)

            new_message = f"MotionLLM caption: {caption}"

            print("\n[Motion Caption]")
            print(caption)

            self.context.append({"role": "assistant", "content": new_message})

        else:
            print(answer)

    def clean(self):
        self.context = []
        self.motion_history = {}
        self.context.append({"role": "system", "content": self.prompt})
        print("Cleaned context.")

    def chat(self):

        while True:

            message = input("User: ")

            if message == "exit":
                break

            if message == "clean":
                self.clean()
                continue

            try:
                self.process_motion_dialogue(message)

            except Exception as e:
                print(f"Error: {e}")