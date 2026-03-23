import json
import os
import tempfile
import sys
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from dataset import dataset_cot
from models.training_utils import process_batch


class DummyTokenizer:
    def __init__(self):
        self.vocab = {}
        self.next_id = 10
        self.bos_token_id = 1
        self.pad_token_id = 0

    def _tokenize(self, text):
        out = []
        for token in text.replace("\n", " ").split(" "):
            if token == "":
                continue
            if token not in self.vocab:
                self.vocab[token] = self.next_id
                self.next_id += 1
            out.append(self.vocab[token])
        return out

    def __call__(self, text, add_special_tokens=False):
        class Output:
            pass
        output = Output()
        output.input_ids = self._tokenize(text)
        return output

    def decode(self, x):
        if isinstance(x, torch.Tensor):
            x = x.tolist()
        return " ".join(str(i) for i in x)


def main():
    path = os.path.join(tempfile.gettempdir(), "morl_weighted_sft_smoke.jsonl")
    with open(path, "w", encoding="utf-8") as f:
        f.write(json.dumps({
            "task": "t2m",
            "caption": "a person walks",
            "reasoning": "first move left leg then right leg",
            "motion_tokens": [1, 2, 3, 4],
            "reward": 0.5
        }) + "\n")
        f.write(json.dumps({
            "task": "t2m",
            "caption": "a person jumps",
            "reasoning": "bend and push up",
            "motion_tokens": "5 6 7",
            "sample_weight": 1.6
        }) + "\n")

    loader = dataset_cot.DATALoader(path, batch_size=2, task="t2m", num_workers=0, shuffle=False)
    batch = next(iter(loader))

    tokenizer = DummyTokenizer()
    inputs, targets, attn, regions = process_batch(
        tokenizer=tokenizer,
        batch_of_captions=batch["caption"],
        max_tgt_len=128,
        batch_of_motions=batch["motion_tokens"],
        training_task="t2m",
        batch_of_reasonings=batch["reasoning"]
    )

    print("batch_size", len(batch["caption"]))
    print("has_reward", batch["reward"])
    print("has_sample_weight", batch["sample_weight"])
    print("shapes", tuple(inputs.shape), tuple(targets.shape), tuple(attn.shape), tuple(regions.shape))
    print("region_max", int(regions.max().item()))

    os.remove(path)


if __name__ == "__main__":
    main()


