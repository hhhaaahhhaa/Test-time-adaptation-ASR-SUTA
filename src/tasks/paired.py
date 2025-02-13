from torch.utils.data import Dataset
import random

from ..corpus.corpus import LibriSpeechCorpus
from ..corpus.corpus import LibriSpeechCCorpus


class LibriSpeechSequence(Dataset):
    """ Pair origin LibriSpeech and LibriSpeech-C. """

    def __init__(self, noise_type: str, snr_level=10) -> None:
        self.corpus = LibriSpeechCorpus()
        self.idx_seq = list(range(len(self.corpus)))

        root = f"_cache/LibriSpeech-c/{noise_type}/snr={snr_level}"
        self.corrputed_corpus = LibriSpeechCCorpus(root=root)

        self.task_boundaries = []
        random.shuffle(self.idx_seq)

    def __len__(self):
        return len(self.idx_seq)
    
    def __getitem__(self, idx):
        return (self.corpus.get(self.idx_seq[idx]), self.corrputed_corpus.get(self.idx_seq[idx]))
