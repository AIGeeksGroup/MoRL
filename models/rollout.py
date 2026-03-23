def rollout_group(model, example, group_size, com_steps=0, task='t2m'):
    if task == 't2m':
        caption = example['caption']
        if com_steps > 0 and hasattr(model, 'generate_com'):
            samples = model.generate_com(
                caption=caption,
                task='t2m',
                k=group_size,
                t=com_steps,
                return_candidates=True,
            )
        else:
            samples = [model.generate_with_trace(caption) for _ in range(group_size)]
    elif task == 'm2t':
        motion_tokens = example['motion_tokens']
        caption = example.get('caption')
        if com_steps > 0 and hasattr(model, 'generate_com'):
            samples = model.generate_com(
                caption=caption,
                motion_tokens=motion_tokens,
                task='m2t',
                k=group_size,
                t=com_steps,
                return_candidates=True,
            )
        else:
            samples = [model.generate_caption_with_trace(motion_tokens) for _ in range(group_size)]
    else:
        raise ValueError(f'Unsupported rollout task: {task}')

    for out in samples:
        out['caption'] = example.get('caption')
        tokens = out.get("motion_tokens")
        if tokens is not None:
            try:
                motion = model.net.forward_decoder(tokens)
            except Exception:
                motion = None
            out["motion"] = motion
    return samples



