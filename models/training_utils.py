import torch
from torch.nn.utils import rnn


# -------- Text → Motion with reasoning --------
def build_one_instance_t2m(tokenizer, caption, motion, reasoning=None):

    input_ids, target_ids, region_ids = [], [], []

    bos = tokenizer.bos_token_id
    input_ids.append(bos)
    target_ids.append(-100)
    region_ids.append(0)

    prompt = (
        "Below is an instruction that describes a task.\n\n"
        "### Instruction:\n"
        "Generate reasoning and a motion that matches the following description.\n\n"
        "### Input:\n"
        f"{caption}\n\n"
        "<think>"
    )

    prompt_ids = tokenizer(prompt, add_special_tokens=False).input_ids
    input_ids += prompt_ids
    target_ids += [-100] * len(prompt_ids)
    region_ids += [0] * len(prompt_ids)

    # -------- reasoning supervision --------
    if reasoning is not None:
        reasoning_text = reasoning + "</think><answer><Motion>"
        reasoning_ids = tokenizer(reasoning_text, add_special_tokens=False).input_ids

        input_ids += reasoning_ids
        target_ids += reasoning_ids
        region_ids += [1] * len(reasoning_ids)
    else:
        tag_ids = tokenizer("</think><answer><Motion>", add_special_tokens=False).input_ids
        input_ids += tag_ids
        target_ids += tag_ids
        region_ids += [1] * len(tag_ids)

    # -------- motion tokens --------
    motion = motion.tolist()
    motion_end = tokenizer("</Motion><eos>", add_special_tokens=False).input_ids

    input_ids += motion + motion_end
    target_ids += motion + motion_end
    region_ids += [2] * (len(motion) + len(motion_end))

    return input_ids, target_ids, region_ids


# -------- Motion → Text with reasoning --------
def build_one_instance_m2t(tokenizer, caption, motion, reasoning=None):

    input_ids, target_ids, region_ids = [], [], []

    bos = tokenizer.bos_token_id
    input_ids.append(bos)
    target_ids.append(-100)
    region_ids.append(0)

    motion_tokens = tokenizer.decode(motion)

    prompt = (
        "Below is an instruction that describes a task.\n\n"
        "### Instruction:\n"
        "Generate reasoning and a caption for the following motion.\n\n"
        "### Input:\n"
        f"<Motion>{motion_tokens}</Motion>\n\n"
        "<think>"
    )

    prompt_ids = tokenizer(prompt, add_special_tokens=False).input_ids
    input_ids += prompt_ids
    target_ids += [-100] * len(prompt_ids)
    region_ids += [0] * len(prompt_ids)

    # reasoning
    if reasoning is not None:
        reasoning_text = reasoning + "</think><answer>"
        reasoning_ids = tokenizer(reasoning_text, add_special_tokens=False).input_ids

        input_ids += reasoning_ids
        target_ids += reasoning_ids
        region_ids += [1] * len(reasoning_ids)
    else:
        tag_ids = tokenizer("</think><answer>", add_special_tokens=False).input_ids
        input_ids += tag_ids
        target_ids += tag_ids
        region_ids += [1] * len(tag_ids)

    caption_ids = tokenizer(caption, add_special_tokens=False).input_ids
    eos_ids = tokenizer("<eos>", add_special_tokens=False).input_ids

    input_ids += caption_ids + eos_ids
    target_ids += caption_ids + eos_ids
    region_ids += [2] * (len(caption_ids) + len(eos_ids))

    return input_ids, target_ids, region_ids


# -------- batch processing --------
def process_batch(tokenizer, batch_of_captions, max_tgt_len, batch_of_motions,
                  training_task, batch_of_reasonings=None, reasoning=None):

    batch_input_ids = []
    batch_target_ids = []
    batch_region_ids = []

    # Backward compatibility: some callers still pass `reasoning=`.
    if batch_of_reasonings is None and reasoning is not None:
        batch_of_reasonings = reasoning

    if batch_of_reasonings is None:
        batch_of_reasonings = [None] * len(batch_of_captions)
    elif isinstance(batch_of_reasonings, str):
        batch_of_reasonings = [batch_of_reasonings] * len(batch_of_captions)

    for caption, motion, reasoning in zip(batch_of_captions, batch_of_motions, batch_of_reasonings):
        if training_task == 't2m':
            built_instance = build_one_instance_t2m(tokenizer, caption, motion, reasoning)
        elif training_task == 'm2t':
            built_instance = build_one_instance_m2t(tokenizer, caption, motion, reasoning)
        else:
            raise ValueError(f"Unsupported training task: {training_task}")

        batch_input_ids.append(torch.LongTensor(built_instance[0]))
        batch_target_ids.append(torch.LongTensor(built_instance[1]))
        batch_region_ids.append(torch.LongTensor(built_instance[2]))

    input_ids = rnn.pad_sequence(
        batch_input_ids,
        batch_first=True,
        padding_value=tokenizer.pad_token_id
    )

    target_ids = rnn.pad_sequence(
        batch_target_ids,
        batch_first=True,
        padding_value=-100
    )

    region_ids = rnn.pad_sequence(
        batch_region_ids,
        batch_first=True,
        padding_value=0
    )

    input_ids = input_ids[:, :max_tgt_len]
    target_ids = target_ids[:, :max_tgt_len]
    region_ids = region_ids[:, :max_tgt_len]

    attention_mask = input_ids.ne(tokenizer.pad_token_id)

    return input_ids, target_ids, attention_mask.long(), region_ids
