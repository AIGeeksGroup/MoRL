from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model
import torch.nn as nn
import torch
import torch.nn.functional as F
from models.training_utils import *
import numpy as np
import models.vqvae as vqvae


class MotionLLM(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.args = args
        self.tokenizer = AutoTokenizer.from_pretrained(self.args.llm_backbone)
        self.llm = AutoModelForCausalLM.from_pretrained(self.args.llm_backbone)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.nb_text_tokens = len(self.tokenizer)

        self.mean = np.load('checkpoints/t2m/VQVAEV3_CB1024_CMT_H1024_NRES3/meta/mean.npy')
        self.std = np.load('checkpoints/t2m/VQVAEV3_CB1024_CMT_H1024_NRES3/meta/std.npy')

        self.device = args.device
        self.training_task = None

        self.lora_config_t2m = LoraConfig(
            r=self.args.lora_r_t2m,
            lora_alpha=self.args.lora_alpha_t2m,
            target_modules=['o_proj', 'q_proj', 'up_proj', 'v_proj', 'k_proj', 'down_proj', 'gate_proj'],
            lora_dropout=self.args.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM"
        )

        self.lora_config_m2t = LoraConfig(
            r=self.args.lora_r_m2t,
            lora_alpha=self.args.lora_alpha_m2t,
            target_modules=['o_proj', 'q_proj', 'up_proj', 'v_proj', 'k_proj', 'down_proj', 'gate_proj'],
            lora_dropout=self.args.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM"
        )

        self.llm = get_peft_model(self.llm, self.lora_config_t2m, adapter_name='t2m')
        self.llm.add_adapter('m2t', self.lora_config_m2t)

        self.args.nb_joints = 22
        self.args.dataname = 't2m'
        self.args.vq_path = "ckpt/vqvae.pth"

        self.net = vqvae.HumanVQVAE(
            self.args,
            self.args.nb_code,
            self.args.code_dim,
            self.args.output_emb_width,
            self.args.down_t,
            self.args.stride_t,
            self.args.width,
            self.args.depth,
            self.args.dilation_growth_rate,
            self.args.vq_act,
            self.args.vq_norm
        )

        print('loading vqvae from {}'.format(self.args.vq_path))
        ckpt = torch.load(self.args.vq_path, map_location='cpu')
        self.net.load_state_dict(ckpt['net'], strict=True)
        self.net.eval()
        self.net.to(self.device)

        # add tokens
        self.tokenizer.add_tokens([
            '<Motion>', '</Motion>',
            '<think>', '</think>',
            '<answer>', '</answer>'
        ])

        self.motion_token_start = len(self.tokenizer)
        self.motion_token_indices = np.arange(self.args.nb_code)
        self.motion_token_indices = self.motion_token_start + self.motion_token_indices

        for i in range(self.args.nb_code):
            self.tokenizer.add_tokens([f'<Motion_{i}>'])

        self.motion_token_end = self.motion_token_start + self.args.nb_code
        self.motion_tag_start_id = self.tokenizer.convert_tokens_to_ids('<Motion>')
        self.motion_tag_end_id = self.tokenizer.convert_tokens_to_ids('</Motion>')

        self.llm.resize_token_embeddings(len(self.tokenizer))
        self.llm.to(self.device)
        self.llm.eval()

    def forward(self, caption, motion, reasoning=None, sample_weights=None):

        if self.training_task == 't2m':
            self.llm.set_adapter('t2m')
        elif self.training_task == 'm2t':
            self.llm.set_adapter('m2t')

        inputs_ids, targets, attention_mask, region_ids = process_batch(
            tokenizer=self.tokenizer,
            batch_of_captions=caption,
            max_tgt_len=getattr(self.args, 'max_motion_len', 200),
            batch_of_motions=motion,
            batch_of_reasonings=reasoning,
            training_task=self.training_task
        )

        inputs_ids = inputs_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)
        targets = targets.to(self.device)
        region_ids = region_ids.to(self.device)

        outputs = self.llm(
            input_ids=inputs_ids,
            attention_mask=attention_mask,
            return_dict=True,
            output_hidden_states=True,
        )

        loss = self.compute_sequence_nll_from_tensors(
            logits=outputs.logits,
            targets=targets,
            region_ids=region_ids,
            sample_weights=sample_weights,
        )

        chosen_tokens = torch.max(outputs.logits, dim=-1)[1][:, 1:-1]
        labels = targets[:, 2:]

        gen_acc = (chosen_tokens.reshape(-1) == labels.reshape(-1)).to(torch.long)
        valid_mask = (labels != -100).reshape(-1)
        valid_tokens = gen_acc & valid_mask

        gen_acc = valid_tokens.sum().item() / (valid_mask.sum().item() + 1.0)

        return loss, gen_acc, chosen_tokens, labels

    def compute_sequence_nll(self, caption, motion, reasoning=None, sample_weights=None):
        inputs_ids, targets, attention_mask, region_ids = process_batch(
            tokenizer=self.tokenizer,
            batch_of_captions=caption,
            max_tgt_len=getattr(self.args, 'max_motion_len', 200),
            batch_of_motions=motion,
            batch_of_reasonings=reasoning,
            training_task=self.training_task
        )

        inputs_ids = inputs_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)
        targets = targets.to(self.device)
        region_ids = region_ids.to(self.device)

        outputs = self.llm(
            input_ids=inputs_ids,
            attention_mask=attention_mask,
            return_dict=True,
            output_hidden_states=False,
        )

        return self.compute_sequence_nll_from_tensors(
            logits=outputs.logits,
            targets=targets,
            region_ids=region_ids,
            sample_weights=sample_weights,
        )

    def compute_sequence_nll_from_tensors(self, logits, targets, region_ids, sample_weights=None):
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = targets[:, 1:].contiguous()
        shift_regions = region_ids[:, 1:].contiguous()

        token_loss = F.cross_entropy(
            shift_logits.reshape(-1, shift_logits.size(-1)),
            shift_labels.reshape(-1),
            reduction='none',
            ignore_index=-100
        ).view(shift_labels.size(0), shift_labels.size(1))

        token_weights = torch.ones_like(token_loss)
        token_weights = torch.where(
            shift_regions == 1,
            torch.full_like(token_weights, float(getattr(self.args, 'reasoning_weight', 1.0))),
            token_weights
        )
        token_weights = torch.where(
            shift_regions == 2,
            torch.full_like(token_weights, float(getattr(self.args, 'motion_weight', 1.0))),
            token_weights
        )

        valid_mask = shift_labels.ne(-100).float()
        weighted_loss = token_loss * token_weights * valid_mask

        if sample_weights is not None:
            sw = torch.tensor(sample_weights, dtype=weighted_loss.dtype, device=self.device).view(-1, 1)
            weighted_loss = weighted_loss * sw

        return weighted_loss.sum() / valid_mask.sum().clamp_min(1.0)

    def compute_per_token_logprob(self, caption, motion, reasoning=None, task=None):
        """
        Compute the **sum** of log-probabilities over all valid output tokens.
        Returns (sum_logprob: Tensor [scalar, grad-enabled], num_tokens: int).

        Used by GRPOTrainer to compute per-token-normalised importance ratios:
            log_ratio = (sum_logprob_policy - sum_logprob_ref) / num_tokens
            ratio      = exp(log_ratio)
        """
        if task is None:
            task = self.training_task

        # Ensure the correct LoRA adapter is active before the forward pass.
        if task == 't2m':
            self.llm.set_adapter('t2m')
        elif task == 'm2t':
            self.llm.set_adapter('m2t')

        inputs_ids, targets, attention_mask, _ = process_batch(
            tokenizer=self.tokenizer,
            batch_of_captions=caption,
            max_tgt_len=getattr(self.args, 'max_motion_len', 200),
            batch_of_motions=motion,
            batch_of_reasonings=reasoning,
            training_task=task,
        )

        inputs_ids = inputs_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)
        targets = targets.to(self.device)

        outputs = self.llm(
            input_ids=inputs_ids,
            attention_mask=attention_mask,
            return_dict=True,
        )

        shift_logits = outputs.logits[:, :-1, :].contiguous()   # [B, T-1, V]
        shift_labels = targets[:, 1:].contiguous()               # [B, T-1]
        valid_mask = shift_labels.ne(-100)                       # [B, T-1]

        log_probs = F.log_softmax(shift_logits, dim=-1)          # [B, T-1, V]

        # Replace -100 padding with 0 so gather doesn't fail; mask them out afterwards.
        safe_labels = shift_labels.masked_fill(~valid_mask, 0)
        token_logprobs = log_probs.gather(
            -1, safe_labels.unsqueeze(-1)
        ).squeeze(-1)                                            # [B, T-1]
        token_logprobs = token_logprobs * valid_mask.float()

        sum_logprob = token_logprobs.sum()                       # scalar, has grad
        num_tokens = int(valid_mask.sum().item())
        return sum_logprob, num_tokens

    def generate(self, caption):

        self.llm.set_adapter('t2m')
        self.llm.eval()

        prompt = (
            "Below is an instruction that describes a task.\n\n"
            "### Instruction:\n"
            "Generate reasoning and motion for the following description.\n\n"
            "### Input:\n"
            + caption +
            "\n\n<think>"
        )

        input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)

        outputs = self.llm.generate(
            input_ids,
            max_length=200,
            num_beams=2,
            early_stopping=True,
            return_dict_in_generate=True
        )

        gen_ids = outputs.sequences[0, input_ids.shape[1]:]
        return self._extract_motion_tokens(gen_ids)

    def generate_with_trace(self, caption, max_length=240, do_sample=True, temperature=0.9):
        self.llm.set_adapter('t2m')
        self.llm.eval()

        prompt = (
            "Below is an instruction that describes a task.\n\n"
            "### Instruction:\n"
            "Generate reasoning and motion for the following description.\n\n"
            "### Input:\n"
            + caption +
            "\n\n<think>"
        )

        input_ids = self.tokenizer.encode(prompt, return_tensors='pt').to(self.device)
        outputs = self.llm.generate(
            input_ids,
            max_length=max_length,
            do_sample=do_sample,
            temperature=temperature,
            num_beams=1,
            return_dict_in_generate=True
        )

        seq = outputs.sequences[0, input_ids.shape[1]:]
        text = self.tokenizer.decode(seq, skip_special_tokens=False)
        reasoning = self._extract_tag_content(text, 'think')
        answer = self._extract_tag_content(text, 'answer')
        motion_tokens = self._extract_motion_tokens(seq)

        return {
            'caption': caption,
            'text': text,
            'reasoning': reasoning,
            'answer': answer,
            'motion_tokens': motion_tokens
        }

    def generate_caption_with_trace(self, motion_tokens, max_length=240, do_sample=True, temperature=0.9, reflection_context=None):
        self.llm.set_adapter('m2t')
        self.llm.eval()

        if torch.is_tensor(motion_tokens):
            motion_text = self.tokenizer.decode(motion_tokens.detach().cpu().tolist())
        else:
            motion_text = self.tokenizer.decode(motion_tokens)
        prompt = (
            "Below is an instruction that describes a task.\n\n"
            "### Instruction:\n"
            "Generate reasoning and caption for the following motion.\n\n"
            "### Input:\n"
            "<Motion>" + motion_text + "</Motion>\n\n<think>"
        )
        if reflection_context:
            prompt = (
                "Below is an instruction that describes a task.\n\n"
                "### Instruction:\n"
                "Refine your previous reasoning and produce a better caption for the same motion.\n\n"
                "### Input:\n"
                "<Motion>" + motion_text + "</Motion>\n\n"
                "### Previous reasoning:\n"
                + str(reflection_context)
                + "\n\n<think>"
            )

        input_ids = self.tokenizer.encode(prompt, return_tensors='pt').to(self.device)
        outputs = self.llm.generate(
            input_ids,
            max_length=max_length,
            do_sample=do_sample,
            temperature=temperature,
            num_beams=1,
            return_dict_in_generate=True
        )

        seq = outputs.sequences[0, input_ids.shape[1]:]
        text = self.tokenizer.decode(seq, skip_special_tokens=False)
        reasoning = self._extract_tag_content(text, 'think')
        answer = self._extract_tag_content(text, 'answer')

        return {
            'text': text,
            'reasoning': reasoning,
            'answer': answer,
            'motion_tokens': motion_tokens,
        }

    def _default_com_score(self, candidate, task='t2m'):
        # Compatibility-only fallback. Main CoM path should use reward_fn.
        score = 0.0
        if task == 't2m':
            motion_tokens = candidate.get('motion_tokens')
            if motion_tokens is not None and hasattr(motion_tokens, 'numel'):
                score += min(1.0, float(motion_tokens.numel()) / 120.0)
            motion = candidate.get('motion')
            if motion is not None and getattr(motion, 'ndim', 0) >= 3 and motion.shape[1] > 2:
                vel = torch.abs(motion[:, 1:, :] - motion[:, :-1, :]).mean().item()
                acc = torch.abs(motion[:, 2:, :] - 2 * motion[:, 1:-1, :] + motion[:, :-2, :]).mean().item()
                score += float(max(0.0, 1.0 - min(1.0, abs(vel - 0.12) / 0.2)))
                score += float(max(0.0, 1.0 - min(1.0, acc / 0.35)))
        else:
            answer = candidate.get('answer', '') or ''
            score += min(1.0, len(answer.split()) / 12.0)
            if candidate.get('reasoning'):
                score += 0.2
        return float(score)

    def generate_com(self, caption=None, motion_tokens=None, task='t2m', k=8, t=2, reward_fn=None, return_candidates=False, allow_heuristic=False):
        best = None
        best_score = None
        current_caption = caption
        current_motion_tokens = motion_tokens
        current_reflection = None
        all_candidates = []

        if task == 't2m' and not isinstance(current_caption, str):
            raise ValueError('caption is required for t2m CoM')
        if task == 'm2t' and current_motion_tokens is None:
            raise ValueError('motion_tokens is required for m2t CoM')

        if reward_fn is None and not return_candidates and not bool(allow_heuristic):
            raise ValueError('generate_com requires reward_fn for candidate ranking unless allow_heuristic=True')

        for _ in range(max(1, int(t))):
            if task == 't2m':
                candidates = [
                    self.generate_with_trace(current_caption, do_sample=True)
                    for _ in range(max(1, int(k)))
                ]
            else:
                candidates = [
                    self.generate_caption_with_trace(
                        current_motion_tokens,
                        do_sample=True,
                        reflection_context=current_reflection,
                    )
                    for _ in range(max(1, int(k)))
                ]

            for cand in candidates:
                if task == 't2m':
                    cand['caption'] = current_caption
                    tokens = cand.get('motion_tokens')
                    if tokens is not None:
                        try:
                            cand['motion'] = self.net.forward_decoder(tokens)
                        except Exception:
                            cand['motion'] = None
                else:
                    if current_caption is not None:
                        cand['caption'] = current_caption
                    cand['motion_tokens'] = current_motion_tokens
                all_candidates.append(cand)

            if reward_fn is not None or bool(allow_heuristic):
                for cand in candidates:
                    if reward_fn is not None:
                        out = reward_fn(cand)
                        score = float(out[0] if isinstance(out, tuple) else out)
                    else:
                        score = self._default_com_score(cand, task=task)
                    if best_score is None or score > best_score:
                        best = cand
                        best_score = score
            elif len(candidates) > 0 and best is None:
                best = candidates[0]

            if task == 't2m' and best is not None and best.get('reasoning'):
                # Reflection prompt: keep the source caption and append best prior reasoning.
                current_caption = (
                    caption
                    + "\n\nRefine the motion plan using this prior reasoning:\n"
                    + str(best.get('reasoning'))
                    + "\nReturn improved <think> and <answer><Motion>...</Motion></answer>."
                )
            elif task == 'm2t' and best is not None and best.get('reasoning'):
                current_reflection = str(best.get('reasoning'))

        if return_candidates:
            if len(all_candidates) > 0:
                return all_candidates
            if task == 't2m':
                return [self.generate_with_trace(caption, do_sample=True)]
            return [self.generate_caption_with_trace(motion_tokens, do_sample=True)]

        if best is not None:
            return best
        if task == 't2m':
            return self.generate_with_trace(caption, do_sample=False)
        return self.generate_caption_with_trace(motion_tokens, do_sample=False, reflection_context=current_reflection)

    def generate_reasoning(self, caption, task='t2m', max_length=128):
        """Generate a reasoning trace for a caption or batch of captions."""
        if isinstance(caption, (list, tuple)):
            return [self.generate_reasoning(one, task=task, max_length=max_length) for one in caption]

        self.llm.eval()
        self.llm.set_adapter('t2m' if task == 't2m' else 'm2t')

        prompt = (
            "Below is an instruction that describes a task.\n\n"
            "### Instruction:\n"
            "Reason step-by-step for the following description.\n\n"
            "### Input:\n"
            + caption +
            "\n\n<think>"
        )

        input_ids = self.tokenizer.encode(prompt, return_tensors='pt').to(self.device)
        outputs = self.llm.generate(
            input_ids,
            max_length=min(max_length + input_ids.shape[1], 512),
            num_beams=1,
            do_sample=False
        )

        pred = outputs[0, input_ids.shape[1]:]
        pred_text = self.tokenizer.decode(pred, skip_special_tokens=False)
        return self._extract_tag_content(pred_text, 'think')

    def caption(self, motion):

        self.llm.set_adapter('m2t')
        self.llm.eval()

        motion = self.normalize(motion)
        motion = torch.from_numpy(motion).float().to(self.device).unsqueeze(0)

        motion_tokens = self.net.encode(motion).squeeze(0)
        motion_tokens = motion_tokens + self.motion_token_start

        prompt = (
            "Below is an instruction that describes a task.\n\n"
            "### Instruction:\n"
            "Generate reasoning and caption for the following motion.\n\n"
            "### Input:\n"
            "<Motion>" + self.tokenizer.decode(motion_tokens) + "</Motion>\n\n<think>"
        )

        input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)

        pred = self.llm.generate(
            input_ids,
            max_length=200,
            num_beams=2
        )

        pred = pred[0, len(input_ids[0]):]
        pred = self.tokenizer.decode(pred, skip_special_tokens=False)

        answer = self._extract_tag_content(pred, 'answer')
        if answer == pred.strip():
            caption = pred.split('<eos>')[0].strip()
        else:
            caption = answer.strip()

        return caption

    def _extract_tag_content(self, text, tag_name):
        start_tag = f'<{tag_name}>'
        end_tag = f'</{tag_name}>'
        if start_tag in text and end_tag in text:
            return text.split(start_tag, 1)[1].split(end_tag, 1)[0].strip()
        return text.strip()

    def _extract_motion_tokens(self, generated_ids):
        motion_tokens = []
        inside_motion = False

        for token_id in generated_ids.tolist():
            if token_id == self.motion_tag_start_id:
                inside_motion = True
                continue

            if token_id == self.motion_tag_end_id and inside_motion:
                break

            if self.motion_token_start <= token_id < self.motion_token_end:
                inside_motion = True
                motion_tokens.append(token_id - self.motion_token_start)

        if len(motion_tokens) == 0:
            return torch.zeros(1, dtype=torch.long, device=self.device)

        return torch.tensor(motion_tokens, dtype=torch.long, device=self.device)

    def save_model(self, path):

        save_dict = {}

        for name, param in self.llm.named_parameters():
            if 'lora' in name:
                save_dict[name] = param

        embeddings = self.llm.get_input_embeddings().weight[self.nb_text_tokens:]
        save_dict['embeddings'] = embeddings

        lm_head = self.llm.lm_head.weight[self.nb_text_tokens:]
        save_dict['lm_head'] = lm_head

        torch.save(save_dict, path)

    def load_model(self, path):

        print(f"Loading model from {path}")

        save_dict = torch.load(path, map_location=self.device)

        for name, param in self.llm.named_parameters():
            if name in save_dict:
                param.data = save_dict[name]

        self.llm.get_input_embeddings().weight.data[self.nb_text_tokens:] = save_dict['embeddings']
        self.llm.lm_head.weight.data[self.nb_text_tokens:] = save_dict['lm_head']

    def denormalize(self, motion):
        return self.mean + motion * self.std

    def normalize(self, motion):
        return (motion - self.mean) / self.std