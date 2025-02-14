from torch.utils.data import Dataset, Subset
import random

from corpus.corpus import CommonVoiceCorpus


class RandomSequence(Dataset):
    def __init__(self) -> None:
        self.corpus = CommonVoiceCorpus()
        self.idx_seq = list(range(len(self.corpus)))
        self.task_boundaries = []
        random.shuffle(self.idx_seq)

    def __len__(self):
        return len(self.corpus)
    
    def __getitem__(self, idx):
        return self.corpus.get(self.idx_seq[idx])
    

class FullRandomSequence(Dataset):
    def __init__(self) -> None:
        self.corpus = CommonVoiceCorpus(partial=False)
        self.idx_seq = list(range(len(self.corpus)))
        self.task_boundaries = []
        random.shuffle(self.idx_seq)

    def __len__(self):
        return len(self.corpus)
    
    def __getitem__(self, idx):
        return self.corpus.get(self.idx_seq[idx])


class GoodSequence(Dataset):    
    def __new__(cls):
        origin_ds = RandomSequence()
        import pickle
        src = "results/benchmark/none/default"
        tgt = "results/benchmark/suta/default-0"
        task_name = "commonvoice_random"
        with open (f"{src}/{task_name}/result/results.pkl", "rb") as f:
            seq1 = pickle.load(f)
        wer1, n_word1 = seq1["wers"], seq1["n_words"]
        with open (f"{tgt}/{task_name}/result/results.pkl", "rb") as f:
            seq2 = pickle.load(f)
        wer2, n_word2 = seq2["wers"], seq2["n_words"]

        subset_idxs = []
        for i, (x, y) in enumerate(zip(wer1, wer2)):
            if x - 0.1 > y:
                subset_idxs.append(i)
        print("Filtered: ", len(subset_idxs))
        return Subset(origin_ds, subset_idxs)
    