import csv
import os
import pickle

import numpy as np

from datasets.audioset import get_audioset
from datasets.dataset_base_classes import DatasetBaseClass
from utils.directories import get_dataset_dir

SPLITS = ['train', 'val', 'test']


def get_audiocaps(split):
    ds = AudioCapsDataset(split)
    return ds


class AudioCapsDataset(DatasetBaseClass):

    def __init__(self, split, folder_name='audiocaps', compress=True, mp3=False):
        super().__init__()

        root_dir = os.path.join(get_dataset_dir(), folder_name)
        # check parameters
        assert os.path.exists(root_dir), f'Parameter \'audio_caps_root\' is invalid. {root_dir} does not exist.'
        assert split in SPLITS, f'Parameter \'split\' must be in {SPLITS}.'

        self.audio_caps_root = root_dir
        self.split = split
        self.compress = compress

        if split == 'validation':  # rename validation split
            split = 'val'

        # read ytids and captions from csv
        with open(os.path.join(self.audio_caps_root, 'dataset', f'{split}.csv'), 'r', encoding='utf-8') as f:
            reader = csv.reader(f, delimiter=',')
            lines = list(reader)[1:]
        audiocap_ids, self.ytids, _, self.captions = list(map(list, zip(*lines)))
        # sort captions by ytid
        self.ytids, audiocap_ids, self.captions = list(zip(*sorted(zip(self.ytids, audiocap_ids, self.captions))))

        # get paths and prediction targets
        self.audioset = get_audioset('train').set_quick(True)
        idx = dict(zip(self.audioset.ytids, range(0, len(self.audioset.ytids))))

        # get caption sbert embeddings
        captions_sbert = f'captions_sbert_{split}.pkl'
        with open(os.path.join(root_dir, captions_sbert), "rb") as stream:
            captions_embed = pickle.load(stream)

        self.paths, self.targets, self.keywords, self.captions_embed = [], [], [], []
        for ytid, aid, caption in zip(self.ytids, audiocap_ids, self.captions):
            i = idx.get('Y' + ytid)
            if i is None:
                continue
            self.paths.append(self.audioset[i]['path'][:-3] + 'mp3' if mp3 else self.audioset[i]['path'])
            self.keywords.append(";".join([
                self.audioset.ix_to_lb[i] for i in np.where(self.audioset[i]['target'])[0]
            ]))
            self.targets.append(caption)
            # self.targets.append(self.audioset[i]['target'])
            self.captions_embed.append(captions_embed[aid])

        self.captions = self.targets

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        a = self.__get_audio__(index)
        a['keywords'] = self.keywords[index]
        a['caption'] = self.targets[index]
        a['caption_embed'] = self.captions_embed[index]
        a['idx'] = index + 1000000
        a['caption_hard'] = ''
        a['html'] = ''
        a['xhid'] = ''
        return a

    def __get_audio_paths__(self):
        return self.paths

    def __str__(self):
        return f'AudioCaps_{self.split}'
