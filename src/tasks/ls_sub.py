import os
from torch.utils.data import Dataset
import random
import pickle
import numpy as np

from ..corpus.corpus import LibriSpeechCorpus


class RandomSubset(Dataset):
    def __init__(self, size=50) -> None:
        self.corpus = LibriSpeechCorpus()
        self.idx_seq = list(range(len(self.corpus)))
        self.task_boundaries = []
        self.idx_seq = random.sample(self.idx_seq, size)

    def __len__(self):
        return len(self.idx_seq)
    
    def __getitem__(self, idx):
        return self.corpus.get(self.idx_seq[idx])
    

class BestRescoreSubset(Dataset):
    def __init__(self, size=50) -> None:
        self.corpus = LibriSpeechCorpus()
        self.idx_seq = list(range(len(self.corpus)))
        self.task_boundaries = []
        random.shuffle(self.idx_seq)
        self._prepare_subset(size=size)

    def _prepare_subset(self, size: int):
        assert os.path.isdir("results/benchmark/none/benchmark/librispeech_random"), "need to run librispeech_random(none) first"
        assert os.path.isdir("results/benchmark/rescore/benchmark/librispeech_random"), "need to run librispeech_random(rescore) first"
        with open(f"results/benchmark/none/benchmark/librispeech_random/result/results.pkl", 'rb') as f:
            orig_info = pickle.load(f)
        with open(f"results/benchmark/rescore/benchmark/librispeech_random/result/results.pkl", 'rb') as f:
            rescored_info = pickle.load(f)
        score = []
        for i in range(len(self.idx_seq)):
            wer_diff = rescored_info["wers"][i] - orig_info["wers"][i]
            score.append(-wer_diff)
        selected = np.argsort(np.array(score))[-size:]
        self.idx_seq = [self.idx_seq[k] for k in selected]
    
    def __len__(self):
        return len(self.idx_seq)
    
    def __getitem__(self, idx):
        return self.corpus.get(self.idx_seq[idx])


class WorstRescoreSubset(Dataset):
    def __init__(self, size=50) -> None:
        self.corpus = LibriSpeechCorpus()
        self.idx_seq = list(range(len(self.corpus)))
        self.task_boundaries = []
        random.shuffle(self.idx_seq)
        self._prepare_subset(size=size)

    def _prepare_subset(self, size: int):
        assert os.path.isdir("results/benchmark/none/benchmark/librispeech_random"), "need to run librispeech_random(none) first"
        assert os.path.isdir("results/benchmark/rescore/benchmark/librispeech_random"), "need to run librispeech_random(rescore) first"
        with open(f"results/benchmark/none/benchmark/librispeech_random/result/results.pkl", 'rb') as f:
            orig_info = pickle.load(f)
        with open(f"results/benchmark/rescore/benchmark/librispeech_random/result/results.pkl", 'rb') as f:
            rescored_info = pickle.load(f)
        score = []
        for i in range(len(self.idx_seq)):
            wer_diff = rescored_info["wers"][i] - orig_info["wers"][i]
            score.append(-wer_diff)
        selected = np.argsort(np.array(score))[:size]
        self.idx_seq = [self.idx_seq[k] for k in selected]
    
    def __len__(self):
        return len(self.idx_seq)
    
    def __getitem__(self, idx):
        return self.corpus.get(self.idx_seq[idx])


class TrainingSubset(Dataset):
    def __init__(self) -> None:
        self.corpus = LibriSpeechCorpus(split="train.clean.100")
        self.idx_seq = list(range(len(self.corpus)))
        self.task_boundaries = []

    def __len__(self):
        return len(self.idx_seq)
    
    def __getitem__(self, idx):
        return self.corpus.get(self.idx_seq[idx])
    

class ValidationSubset(Dataset):
    def __init__(self) -> None:
        self.corpus = LibriSpeechCorpus(split="validation.clean")
        self.idx_seq = list(range(len(self.corpus)))
        self.task_boundaries = []

    def __len__(self):
        return len(self.idx_seq)
    
    def __getitem__(self, idx):
        return self.corpus.get(self.idx_seq[idx])
