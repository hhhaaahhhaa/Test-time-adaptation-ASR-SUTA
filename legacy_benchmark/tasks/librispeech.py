from torch.utils.data import Dataset
import random

from corpus.corpus import LibriSpeechCorpus, LibriSpeechCorpusOld


class RandomSequence(Dataset):
    def __init__(self) -> None:
        self.corpus = LibriSpeechCorpus()
        self.idx_seq = list(range(len(self.corpus)))
        self.task_boundaries = []
        random.shuffle(self.idx_seq)

    def __len__(self):
        return len(self.corpus)
    
    def __getitem__(self, idx):
        return self.corpus.get(self.idx_seq[idx])
    

class RandomSequence1(Dataset):
    def __init__(self) -> None:
        self.corpus = LibriSpeechCorpusOld(extra_noise=0.005)
        self.idx_seq = list(range(len(self.corpus)))
        self.task_boundaries = []
        random.shuffle(self.idx_seq)

    def __len__(self):
        return len(self.corpus)
    
    def __getitem__(self, idx):
        return self.corpus.get(self.idx_seq[idx])


class RandomSequence2(Dataset):
    def __init__(self) -> None:
        self.corpus = LibriSpeechCorpusOld(extra_noise=0.01)
        self.idx_seq = list(range(len(self.corpus)))
        self.task_boundaries = []
        random.shuffle(self.idx_seq)

    def __len__(self):
        return len(self.corpus)
    
    def __getitem__(self, idx):
        return self.corpus.get(self.idx_seq[idx])
