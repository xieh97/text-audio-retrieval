import csv
import json
import os
import pickle

from datasets.dataset_base_classes import DatasetBaseClass
from utils.directories import get_dataset_dir


def get_wavecaps(folder_name='wavcaps', exclude_clothov2=True):
    path = os.path.join(get_dataset_dir(), folder_name)
    wc = WaveCaps()

    if exclude_clothov2:
        with open(os.path.join(path, 'dcase2024_task6_excluded_freesound_ids.csv'), 'r') as f:
            ids = set([r[0] for r in csv.reader(f)][1:])
            before_len = len(wc)
            print("WavCaps before filtering ClothoV2:", len(wc))
            wc = wc.get_subset(
                lambda s: not (s['path'].split(os.sep)[-2] == 'FreeSound' and s['path'].split(os.sep)[-1].split('.')[
                    0] in ids)
            )
            after_len = len(wc)
            print("WavCaps after filtering ClothoV2:", len(wc))
            assert after_len < before_len
    return wc


def get_audioset_subset(wave_caps_root):
    with open(os.path.join(wave_caps_root, 'json_files', 'AudioSet_SL', 'as_final.json'), 'r') as f:
        files = json.load(f)['data']

    missing = {'Y06-g5jz-OGc.wav', 'Y3sSblRfEG2o.wav', 'YFli8wjBFV2M.wav', 'YVcu0pVF1npM.wav', 'YWudGD6ZHRoY.wav',
               'YmW3S0u8bj58.wav'}

    return [
        {
            'xhid': 'AS' + f['id'][:-4],
            'path': os.path.join(wave_caps_root, 'audio', 'AudioSet_SL', f['id'][:-4] + '.flac'),
            'caption': f['caption'],
            'keywords': ""
            # 'description': "",
            # 'url': f['id']
        } for f in sorted(files, key=lambda x: x['id']) if f['id'] not in missing
    ]


def get_soundbible_subset(wave_caps_root):
    with open(os.path.join(wave_caps_root, 'json_files', 'SoundBible', 'sb_final.json'), 'r') as f:
        files = json.load(f)['data']

    return [
        {
            'xhid': 'SB' + f['id'],
            'path': os.path.join(wave_caps_root, 'audio', 'SoundBible', f['id'] + '.flac'),
            'caption': f['caption'],
            'keywords': f['title']
            # 'description': f['description'],
            # 'url': f['id']
        } for f in sorted(files, key=lambda x: x['id'])
    ]


def get_bbc_subset(wave_caps_root, filter=False):
    with open(os.path.join(wave_caps_root, 'json_files', 'BBC_Sound_Effects', 'bbc_final.json'), 'r') as f:
        files = json.load(f)['data']

    return [
        {
            'xhid': 'BB' + f['id'],
            'path': os.path.join(wave_caps_root, 'audio', 'BBC_Sound_Effects', f['id'] + '.flac'),
            'caption': f['caption'],
            'keywords': ";".join(
                [p.replace('[', '').replace(']', '').replace("'", '') for p in f['category'].split(",")])
            # 'description': f['description'],
            # 'url': f['id']
        } for f in sorted(files, key=lambda x: x['id'])
    ]


def get_freesound_subset(wave_caps_root):
    with open(os.path.join(wave_caps_root, 'json_files', 'FreeSound', 'fsd_final.json'), 'r') as f:
        files = json.load(f)['data']

    return [
        {
            'xhid': 'FS' + f['id'],
            'path': os.path.join(wave_caps_root, 'audio', 'FreeSound', f['id'] + '.flac'),
            'caption': f['caption'],
            'keywords': ";".join(f['tags'])
            # 'description': f['description'],
            # 'url': f['href']
        } for f in sorted(files, key=lambda x: x['id']) if f['id'] != '592228'
    ]


class WaveCaps(DatasetBaseClass):

    def __init__(self, folder_name='wavcaps', compress=True):
        super().__init__()

        root_dir = os.path.join(get_dataset_dir(), folder_name)
        # check parameters
        assert os.path.exists(root_dir), f'Parameter \'audio_caps_root\' is invalid. {root_dir} does not exist.'

        self.wave_caps_root = root_dir

        # with open(os.path.join(root_dir, 'missing_files.json'), 'r') as f:
        #    self.missing = json.load(f)
        #    self.missing = set(["/".join(m.split('/')[-2:]) for m in self.missing])

        samples_as = get_audioset_subset(self.wave_caps_root)
        samples_soundbible = get_soundbible_subset(self.wave_caps_root)
        samples_fsd = get_freesound_subset(self.wave_caps_root)
        # samples_fsd = get_freesound_2_subset(self.wave_caps_root)
        samples_bbc = get_bbc_subset(self.wave_caps_root)

        # filter

        print("Files per data set:")
        print("AudioSet: ", len(samples_as))
        print("FreeSound: ", len(samples_fsd))
        print("SoundBible: ", len(samples_soundbible))
        print("BBC: ", len(samples_bbc))
        # print("missing_files: ", len(self.missing))

        self.samples = samples_bbc + samples_soundbible + samples_fsd + samples_as

        # self.samples = [s for s in self.samples if "/".join(s['path'].split('/')[-2:]) not in self.missing]

        # get caption sbert embeddings
        captions_sbert = 'wavcaps_captions_sbert.pkl'
        with open(os.path.join(root_dir, captions_sbert), "rb") as stream:
            self.captions_embed = pickle.load(stream)

        self.captions = [s["caption"] for s in self.samples]
        self.paths = [s["path"] for s in self.samples]
        self.compress = compress

    def __get_audio_paths__(self):
        return self.paths

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        audio = self.__get_audio__(index)
        for k in self.samples[index]:
            audio[k] = self.samples[index][k]
        audio['caption_embed'] = self.captions_embed[self.samples[index]['xhid']]
        # audio["keywords"] = ''
        audio["idx"] = index
        audio["caption_hard"] = ''
        audio["html"] = ''
        return audio

    def __str__(self):
        return 'WavCaps'
