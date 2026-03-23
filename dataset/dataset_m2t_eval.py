import codecs as cs
from os.path import join as pjoin

import numpy as np
import torch
from torch.utils import data
from tqdm import tqdm


def collate_fn(batch):
    batch.sort(key=lambda x: x[1], reverse=True)
    motions = [torch.from_numpy(x[0]).float() if isinstance(x[0], np.ndarray) else x[0].float() for x in batch]
    lengths = torch.tensor([int(x[1]) for x in batch], dtype=torch.long)
    refs = [list(x[2]) for x in batch]
    names = [x[3] for x in batch]
    max_len = max(int(m.shape[0]) for m in motions)
    dim = int(motions[0].shape[1])
    padded = torch.zeros(len(motions), max_len, dim, dtype=torch.float32)
    for i, m in enumerate(motions):
        padded[i, :m.shape[0]] = m
    return padded, lengths, refs, names


class Motion2TextEvalDataset(data.Dataset):
    def __init__(self, dataset_name='t2m', split='test'):
        self.dataset_name = dataset_name
        self.split = split

        if dataset_name == 't2m':
            self.data_root = '../HumanML3D/HumanML3D'
            self.motion_dir = pjoin(self.data_root, 'new_joint_vecs')
            self.text_dir = pjoin(self.data_root, 'texts')
            self.max_motion_length = 196
        elif dataset_name == 'kit':
            self.data_root = '../KIT-ML'
            self.motion_dir = pjoin(self.data_root, 'new_joint_vecs')
            self.text_dir = pjoin(self.data_root, 'texts')
            self.max_motion_length = 196
        else:
            raise ValueError(f'Unsupported dataset: {dataset_name}')

        split_file = pjoin(self.data_root, f'{split}.txt')
        id_list = []
        with cs.open(split_file, 'r') as f:
            for line in f.readlines():
                id_list.append(line.strip())

        self.samples = []
        min_motion_len = 40 if self.dataset_name == 't2m' else 24
        for name in tqdm(id_list, desc=f'Loading {dataset_name}-{split} for m2t eval'):
            try:
                motion = np.load(pjoin(self.motion_dir, name + '.npy'))
                if len(motion) < min_motion_len or len(motion) > self.max_motion_length:
                    continue

                refs = []
                with cs.open(pjoin(self.text_dir, name + '.txt')) as f:
                    for line in f.readlines():
                        line_split = line.strip().split('#')
                        if len(line_split) < 1:
                            continue
                        caption = line_split[0].strip()
                        if len(caption) > 0:
                            refs.append(caption)

                refs = sorted(list(set(refs)))
                if len(refs) == 0:
                    continue

                self.samples.append({
                    'name': name,
                    'motion': motion,
                    'length': len(motion),
                    'refs': refs,
                })
            except Exception:
                continue

        if len(self.samples) == 0:
            raise ValueError('No valid m2t eval samples found.')

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        motion = item['motion']
        m_length = item['length']
        refs = item['refs']
        return motion, m_length, refs, item['name']


def DATALoader(dataset_name='t2m', split='test', batch_size=16, num_workers=0, shuffle=False):
    dataset = Motion2TextEvalDataset(dataset_name=dataset_name, split=split)
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
        drop_last=False,
    )


