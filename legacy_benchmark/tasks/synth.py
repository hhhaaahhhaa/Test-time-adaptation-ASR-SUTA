from torch.utils.data import Dataset
import random

from corpus.corpus import SynthCorpus


class RandomSequence(Dataset):
    def __init__(self) -> None:
        self.corpus = SynthCorpus()
        self.idx_seq = list(range(len(self.corpus)))
        self.task_boundaries = []
        random.shuffle(self.idx_seq)
    
    def __len__(self):
        return 1000
    
    def __getitem__(self, idx):
        return self.corpus.get(self.idx_seq[idx])


class ContentSequence(Dataset):
    def __init__(self) -> None:
        self.corpus = SynthCorpus()
        self.idx_seq = []
        self.task_boundaries = []
        for i in range(10):
            for j in range(10):
                for k in range(10):
                    self.idx_seq.append(i * 100 + j * 10 + k)
            self.task_boundaries.append(len(self.idx_seq))
    
    def __len__(self):
        return 1000
    
    def __getitem__(self, idx):
        return self.corpus.get(self.idx_seq[idx])


class SpeakerSequence(Dataset):
    def __init__(self) -> None:
        self.corpus = SynthCorpus()
        self.idx_seq = []
        self.task_boundaries = []
        for j in range(10):
            for i in range(10):
                for k in range(10):
                    self.idx_seq.append(i * 100 + j * 10 + k)
            self.task_boundaries.append(len(self.idx_seq))
        
    def __len__(self):
        return 1000
    
    def __getitem__(self, idx):
        return self.corpus.get(self.idx_seq[idx])
    

class NoiseSequence(Dataset):
    def __init__(self) -> None:
        self.corpus = SynthCorpus()
        self.idx_seq = []
        self.task_boundaries = []
        for k in range(10):
            for i in range(10):
                for j in range(10):
                    self.idx_seq.append(i * 100 + j * 10 + k)
            self.task_boundaries.append(len(self.idx_seq))

    def __len__(self):
        return 1000
    
    def __getitem__(self, idx):
        return self.corpus.get(self.idx_seq[idx])
