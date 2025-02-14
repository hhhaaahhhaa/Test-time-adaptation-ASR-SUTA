from torch.utils.data import Dataset
import random

from corpus.corpus import ESDCorpus


class RandomSequence(Dataset):
    def __init__(self) -> None:
        self.corpus = ESDCorpus()
        self.idx_seq = list(range(len(self.corpus)))
        self.task_boundaries = []
        random.shuffle(self.idx_seq)

    def __len__(self):
        return len(self.corpus)
    
    def __getitem__(self, idx):
        return self.corpus.get(self.idx_seq[idx])


class SpeakerSequence(Dataset):
    def __init__(self) -> None:
        self.corpus = ESDCorpus()
        self.idx_seq = []
        self.task_boundaries = []
        for i in range(10):
            for j in range(5):
                for k in range(30):
                    self.idx_seq.append(i * 150 + j * 30 + k)
            self.task_boundaries.append(len(self.idx_seq))

    def __len__(self):
        return len(self.corpus)
    
    def __getitem__(self, idx):
        return self.corpus.get(self.idx_seq[idx])
    

class EmotionSequence(Dataset):
    def __init__(self) -> None:
        self.corpus = ESDCorpus()
        self.idx_seq = []
        self.task_boundaries = []
        for j in range(5):
            for i in range(10):
                for k in range(30):
                    self.idx_seq.append(i * 150 + j * 30 + k)
            self.task_boundaries.append(len(self.idx_seq))

    def __len__(self):
        return len(self.corpus)
    
    def __getitem__(self, idx):
        return self.corpus.get(self.idx_seq[idx])
    

class ContentSequence(Dataset):
    def __init__(self, extra_noise=0.0) -> None:
        self.corpus = ESDCorpus(extra_noise=extra_noise)
        self.idx_seq = []
        self.task_boundaries = []
        for k in range(30):
            for i in range(10):
                for j in range(5):
                    self.idx_seq.append(i * 150 + j * 30 + k)
            self.task_boundaries.append(len(self.idx_seq))

    def __len__(self):
        return len(self.corpus)
    
    def __getitem__(self, idx):
        return self.corpus.get(self.idx_seq[idx])
