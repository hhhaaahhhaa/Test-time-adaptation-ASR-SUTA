import os
from pathlib import Path
import numpy as np
import librosa
from torch.utils.data import Dataset
import random

from corpus import Define


class L2ArcticSingleSpeakerCorpus(object):
    """ Korean only """
    def __init__(self, spk_name: str) -> None:
        self.root = Define.L2ARCTIC
        self._init_info(spk_name)

    def _init_info(self, spk_name: str):
        self.wav_paths = []
        self.texts = []
        for i in range(593):
            wav_path = f"{self.root}/{spk_name}/wav/arctic_a{i+1:04d}.wav"
            txt_path = f"{self.root}/{spk_name}/transcript/arctic_a{i+1:04d}.txt"
            self.wav_paths.append(wav_path)
            with open(txt_path, 'r') as f:
                text = f.read()
            self.texts.append(text.strip())

    def __len__(self):
        return len(self.wav_paths)

    def get(self, idx) -> np.ndarray:
        wav, _ = librosa.load(self.wav_paths[idx], sr=16000)
        text = self.texts[idx]

        return {
            "id": f"arctic_a{idx+1:04d}",
            "wav": wav,
            "text": text
        }


class RandomSequence1(Dataset):
    def __init__(self) -> None:
        self.corpus = L2ArcticSingleSpeakerCorpus("Korean/HJK")
        self.idx_seq = list(range(len(self.corpus)))
        self.task_boundaries = []
        random.shuffle(self.idx_seq)

    def __len__(self):
        return len(self.corpus)
    
    def __getitem__(self, idx):
        return self.corpus.get(self.idx_seq[idx])


class RandomSequence2(Dataset):
    def __init__(self) -> None:
        self.corpus = L2ArcticSingleSpeakerCorpus("Korean/HKK")
        self.idx_seq = list(range(len(self.corpus)))
        self.task_boundaries = []
        random.shuffle(self.idx_seq)

    def __len__(self):
        return len(self.corpus)
    
    def __getitem__(self, idx):
        return self.corpus.get(self.idx_seq[idx])
