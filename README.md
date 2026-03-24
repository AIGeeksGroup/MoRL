# MoRL: Reinforced Reasoning for Unified Motion Understanding and Generation
>
> Hongpeng Wang*, Zeyu Zhang*<sup>†</sup>, Wenhao Li, Hao Tang<sup>‡</sup>
>
> *Equal contribution. <sup>†</sup>Project lead. <sup>‡</sup>Corresponding author.
>
> ### [Paper](https://arxiv.org/abs/2602.14534) | [Code](https://github.com/AIGeeksGroup/MoRL) | [Website](https://aigeeksgroup.github.io/MoRL) | [Data](https://huggingface.co/datasets/AIGeeksGroup/MoUnd-MoGen-CoT-140K)


## Qualitative Videos

<table>
  <tr>
    <td align="center"><img src="./assets/A_person_backflips_three_times_in_a_row..gif" width="240" alt="A person backflips three times in a row"/></td>
    <td align="center"><img src="./assets/A_person_is_practicing_karate_moves_across_the_floor..gif" width="240" alt="A person is practicing karate moves across the floor"/></td>
    <td align="center"><img src="./assets/A_person_looks_to_the_left_then_kicks_something_with_their_right_foot..gif" width="240" alt="A person looks to the left then kicks something with their right foot"/></td>
    <td align="center"><img src="./assets/A_person_walks_along_a_curved_path_to_the_right..gif" width="240" alt="A person walks along a curved path to the right"/></td>
  </tr>
  <tr>
    <td align="center"><img src="./assets/A_person_walks_forward_slightly_shifting_to_the_right..gif" width="240" alt="A person walks forward slightly shifting to the right"/></td>
    <td align="center"><img src="./assets/A_person_walks_forward_with_a_side-to-side_sway..gif" width="240" alt="A person walks forward with a side-to-side sway"/></td>
    <td align="center"><img src="./assets/A_person_walks_up_stairs..gif" width="240" alt="A person walks up stairs"/></td>
    <td align="center"><img src="./assets/Walking_slowly_along_the_path_shaped_like_an_infinity_symbol..gif" width="240" alt="Walking slowly along the path shaped like an infinity symbol"/></td>
  </tr>
</table>


## Intro

MoRL is a unified multimodal motion model designed to advance both human motion understanding and generation. Unlike prior approaches that treat user queries as a whole and lack explicit reasoning or planning, MoRL leverages a hierarchical post-training pipeline combining supervised fine-tuning (SFT) and reinforcement learning with verifiable rewards (RLVR). Our task-specific reward design is dual-headed: for motion understanding, we introduce semantic alignment and a novel reasoning coherence reward to enforce logically consistent reasoning traces; for motion generation, we combine text–motion consistency with a physical plausibility reward to ensure biomechanical validity and perceptual realism. To further enhance inference, we propose Chain-of-Motion (CoM), a test-time reasoning strategy that enables step-by-step planning and reflection. CoM improves both the robustness of reasoning-based motion understanding and the quality of motion generation through iterative selection and correction. This principle also guides the construction of two large-scale synthetic Chain-of-Thinking (CoT) datasets: MoUnd-CoT-140K and MoGen-CoT-140K, which align motion sequences with reasoning traces and concise action descriptions. Extensive experiments on HumanML3D and KIT-ML demonstrate that MoRL achieves significant gains over state-of-the-art baselines in both logical reasoning and perceptual realism. Our code, data, and models are open-sourced to facilitate further research in unified motion-language modeling.

<p align="center">
  <img src="./assets/img.png" width="480" alt="Overview of MoRL"/>
</p>

<b>Overview of MoRL.</b> MoRL unifies motion understanding and generation under a reinforcement learning paradigm. Motion and text inputs are tokenized into a shared representation space. A hierarchical post-training pipeline first applies SFT on large-scale synthetic CoT datasets to align motion sequences with reasoning traces and concise descriptions, then employs RLVR to refine outputs, enhancing semantic alignment, reasoning coherence, physical plausibility, and text–motion consistency. At inference, the Chain-of-Motion (CoM) decoding strategy enables step-by-step reasoning and reflection, improving both motion understanding and perceptually realistic motion generation.

<p align="center">
  <img src="./assets/img_1.png" width="480" alt="Motion CoT data engine"/>
</p>
<b>Motion CoT data engine.</b> Built on MotionHubV2, one branch (MoUnd-CoT-140K) uses motion sequences and captions with Gemini to construct reasoning chains for understanding, while the other (MoGen-CoT-140K) builds reasoning chains for generation.

[//]: # (## News)

[//]: # ()
[//]: # (- **2026/03/24**: Repository README updated for MoRL reproduction workflow.)

[//]: # (- More updates &#40;datasets/checkpoints/scripts&#41; will be added soon.)

## TODO List

- [x] Upload our paper to arXiv and build project pages.
- [x] Upload the code.
- [x] Release curated MoUnd-CoT / MoGen-CoT data. (see [MoUnd-MoGen-CoT-140K](https://huggingface.co/datasets/AIGeeksGroup/MoUnd-MoGen-CoT-140K))
- [x] Release training checkpoints.

## Quick Start

### Environment Setup

Install dependencies:

```bash
pip install -r requirements.txt
```

### Prepare Basic Resources

This repo provides helper scripts under `prepare/`:
- `prepare/download_extractor.sh`
- `prepare/download_glove.sh`

Run with bash:

```bash
bash prepare/download_extractor.sh
bash prepare/download_ckpt.sh
```

If you are on native Windows PowerShell, you can manually download and unzip these assets following the URLs in the scripts.

## Data Preparation


### Download and Prepare Motion Datasets

You can download the pre-processed motion datasets from our [HuggingFace page](https://huggingface.co/datasets/AIGeeksGroup/MoUnd-MoGen-CoT-140K).
For custom data or full AMASS/kitml/HumanML3D, please follow the instructions in `dataset/` and `prepare/` folders.





## Training

### A) SFT Stage

#### SFT for t2m

```bash
python train_mllm.py \
  --train-stage sft \
  --training-task t2m \
  --cot-train-jsonl path/to/cot_train.jsonl \
  --use-reasoning \
  --exp-name morl_sft_t2m
```

#### SFT for m2t

```bash
python train_mllm.py \
  --train-stage sft \
  --training-task m2t \
  --cot-train-jsonl path/to/cot_train.jsonl \
  --cot-task-filter m2t \
  --use-reasoning \
  --exp-name morl_sft_m2t
```

### B) RLVR Stage (GRPO)

```bash
python train_mllm.py \
  --train-stage rlvr \
  --training-task t2m \
  --cot-train-jsonl path/to/cot_train.jsonl \
  --rl-reference-ckpt experiments/morl_sft_t2m/motionllm_t2m_best.pth \
  --rl-epochs 3 \
  --rl-group-size 8 \
  --exp-name morl_rlvr_t2m
```

For m2t RLVR, set `--training-task m2t` and optionally `--cot-task-filter m2t`.

## Evaluation / Inference

### Evaluate t2m

```bash
python eval_mllm.py \
  --eval-task t2m \
  --eval-ckpt experiments/morl_rlvr_t2m/motionllm_rlvr_epoch_2.pth
```

### Evaluate m2t

```bash
python eval_mllm.py \
  --eval-task m2t \
  --eval-ckpt experiments/morl_rlvr_m2t/motionllm_rlvr_epoch_2.pth
```

### Enable CoM decoding

```bash
python eval_mllm.py \
  --eval-task t2m \
  --eval-ckpt experiments/morl_rlvr_t2m/motionllm_rlvr_epoch_2.pth \
  --use-com \
  --com-candidates 8 \
  --com-refine-steps 2
```

## Citation

If you find this project useful, please consider citing:

```bibtex
@article{wang2026morl,
  title={MoRL: Reinforced Reasoning for Unified Motion Understanding and Generation},
  author={Wang, Hongpeng and Zhang, Zeyu and Li, Wenhao and Tang, Hao},
  journal={arXiv preprint arXiv:2602.14534},
  year={2026}
}
```

## Acknowledgement

We thank the open-source communities behind Motion-Agent, MotionGPT, Qwen, and related motion-language benchmarks for their foundational contributions.
