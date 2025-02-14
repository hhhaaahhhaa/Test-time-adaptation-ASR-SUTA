from torch.utils.data import Dataset
import random

from corpus.corpus import NoisyL2ArcticCorpus


class RandomSequence(Dataset):
    def __init__(self) -> None:
        self.corpus = NoisyL2ArcticCorpus()
        self.idx_seq = list(range(len(self.corpus)))
        self.task_boundaries = []
        random.shuffle(self.idx_seq)

    def __len__(self):
        return len(self.corpus)
    
    def __getitem__(self, idx):
        return self.corpus.get(self.idx_seq[idx])
    

class AccentSequence(Dataset):
    def __init__(self) -> None:
        self.corpus = NoisyL2ArcticCorpus()
        self.idx_seq = []
        self.task_boundaries = []
        for k, accent_name in self.corpus.accent2str.items():
            tmp = []
            spk_idxs = [self.corpus.spks.index(spk) for spk in self.corpus.spks if accent_name in spk]
            for spk_idx in spk_idxs:
                tmp.extend([spk_idx * self.corpus.n_per_spk + j for j in range(self.corpus.n_per_spk)])
            random.shuffle(tmp)
            self.idx_seq.extend(tmp)
            self.task_boundaries.append(len(self.idx_seq))
    
    def __len__(self):
        return len(self.corpus)
    
    def __getitem__(self, idx):
        return self.corpus.get(self.idx_seq[idx])
    

class AccentHardSequence(Dataset):
    def __init__(self) -> None:
        self.corpus = NoisyL2ArcticCorpus()
        self.idx_seq = []
        self.task_boundaries = []
        for k, accent_name in self.corpus.accent2str.items():
            tmp = []
            spk_idxs = [self.corpus.spks.index(spk) for spk in self.corpus.spks if accent_name in spk]
            for spk_idx in spk_idxs:
                tmp.extend([spk_idx * self.corpus.n_per_spk + j for j in range(self.corpus.n_per_spk)])
            self.idx_seq.extend(tmp)
            self.task_boundaries.append(len(self.idx_seq))
    
    def __len__(self):
        return len(self.corpus)
    
    def __getitem__(self, idx):
        return self.corpus.get(self.idx_seq[idx])
    

class ContentSequence(Dataset):
    def __init__(self) -> None:
        self.corpus = NoisyL2ArcticCorpus()
        self.idx_seq = []
        self.task_boundaries = []
        for j in range(self.corpus.n_per_spk):
            tmp = []
            for i in range(24):
                tmp.append(i * self.corpus.n_per_spk + j)
            random.shuffle(tmp)
            self.idx_seq.extend(tmp)
            self.task_boundaries.append(len(self.idx_seq))

    def __len__(self):
        return len(self.corpus)
    
    def __getitem__(self, idx):
        return self.corpus.get(self.idx_seq[idx])


class SpeakerSequence(Dataset):
    def __init__(self) -> None:
        self.corpus = NoisyL2ArcticCorpus()
        self.idx_seq = []
        self.task_boundaries = []
        for i in range(24):
            tmp = []
            for j in range(self.corpus.n_per_spk):
                tmp.append(i * self.corpus.n_per_spk + j)
            random.shuffle(tmp)
            self.idx_seq.extend(tmp)
            self.task_boundaries.append(len(self.idx_seq))

    def __len__(self):
        return len(self.corpus)
    
    def __getitem__(self, idx):
        return self.corpus.get(self.idx_seq[idx])


class SingleAccentSequence(Dataset):
    def __init__(self) -> None:
        self.corpus = NoisyL2ArcticCorpus()
        self.idx_seq = []
        self.task_boundaries = []
        for k, accent_name in self.corpus.accent2str.items():
            tmp = []
            spk_idxs = [self.corpus.spks.index(spk) for spk in self.corpus.spks if accent_name in spk]
            for spk_idx in spk_idxs:
                tmp.extend([spk_idx * self.corpus.n_per_spk + j for j in range(self.corpus.n_per_spk)])
            random.shuffle(tmp)
            self.idx_seq.extend(tmp)
            break
    
    def __len__(self):
        return len(self.idx_seq)
    
    def __getitem__(self, idx):
        return self.corpus.get(self.idx_seq[idx])
