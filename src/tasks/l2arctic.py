import numpy as np
from torch.utils.data import Dataset
import random

from preprocess.librispeech import snr_mixer
from ..corpus.corpus import L2ArcticCorpus
from ..corpus.noise import MUSANNoiseCorpus


class SingleAccentSequence(Dataset):
    def __init__(self, accent: str="Korean", snr_level=None) -> None:
        self.corpus = L2ArcticCorpus()
        self.idx_seq = []
        for spk in self.corpus.accent2spks[accent]:
            self.idx_seq.extend([(spk, i) for i in range(len(self.corpus.spk2files[spk]))])
        self.task_boundaries = []
        random.shuffle(self.idx_seq)

        self.snr_level = snr_level
        self.noises = MUSANNoiseCorpus()

    def add_noise(self, clean_wav: np.ndarray, noise: np.ndarray) -> np.ndarray:
        # repeat noise content if too short
        noiseconcat = noise
        while len(noiseconcat) <= len(clean_wav):
            noiseconcat = np.append(noiseconcat, noise)
        noise = noiseconcat
        if len(noise) > len(clean_wav):
            noise = noise[0:len(clean_wav)]

        noisy_wav = snr_mixer(clean_wav, noise, snr=self.snr_level)
        return noisy_wav

    def __len__(self):
        return len(self.idx_seq)
    
    def __getitem__(self, idx):
        res = self.corpus.get(*self.idx_seq[idx])
        if self.snr_level is None:
            return res
        
        # add noise
        noise = self.noises.get(self.idx_seq[idx][1])
        noisy_wav = self.add_noise(clean_wav=res["wav"], noise=noise)
        res["wav"] = noisy_wav
        return res


class NoisySingleAccentSequence(Dataset):
    def __new__(cls):
        return SingleAccentSequence(accent="Korean", snr_level=5)


class SingleSpeakerSequence(Dataset):
    def __init__(self, spk: str="HJK") -> None:
        self.corpus = L2ArcticCorpus()
        self.idx_seq= [(spk, i) for i in range(len(self.corpus.spk2files[spk]))]
        self.task_boundaries = []
        random.shuffle(self.idx_seq)

    def __len__(self):
        return len(self.idx_seq)
    
    def __getitem__(self, idx):
        return self.corpus.get(*self.idx_seq[idx])
