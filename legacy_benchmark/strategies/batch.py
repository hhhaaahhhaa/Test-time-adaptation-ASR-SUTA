from torch.utils.data import Dataset, DataLoader
import yaml
from tqdm import tqdm

from systems.suta import SUTASystem
from utils.tool import wer
from .basic import BaseStrategy

from dlhlp_lib.utils.data_structure import Queue


class SUTAStrategy(BaseStrategy):
    def __init__(self, config) -> None:
        self.config = config
        self.system_config = config["config"]
        self.system = SUTASystem(self.system_config)

        self.queue = Queue(max_size=16)
    
    def _update(self, sample):
        self.queue.update(sample)
        self.system.load_snapshot("init")
        self.system.eval()
        is_collapse = False
        for _ in range(self.system_config["steps"]):
            record = {}
            self.system.suta_adapt_auto(
                wavs=[s["wav"] for s in self.queue.data],
                batch_size=4,
                record=record,
            )
            if record.get("collapse", False):
                is_collapse = True
        if is_collapse:
            print("oh no")
    
    def run(self, ds: Dataset):
        n_words = []
        errs, losses = [], []
        transcriptions = []
        dataloader = DataLoader(
            ds,
            batch_size=1,
            num_workers=4,
            collate_fn=lambda x: x
        )
        for sample in tqdm(dataloader):
            sample = sample[0]
            n_words.append(len(sample["text"].split(" ")))

            # update
            self._update(sample)

            self.system.eval()
            trans = self.system.inference([sample["wav"]])
            err = wer(sample["text"], trans[0])
            errs.append(err)
            transcriptions.append((sample["text"], trans[0]))
        
        return {
            "wers": errs,
            "n_words": n_words,
            "transcriptions": transcriptions,
            "losses": losses,
        }
    
    def get_adapt_count(self):
        return self.system.adapt_count
            

class CSUTAStrategy(SUTAStrategy):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
    
    def _update(self, sample):
        self.queue.update(sample)
        self.system.eval()
        is_collapse = False
        for _ in range(self.system_config["steps"]):
            record = {}
            self.system.suta_adapt(
                wavs=[s["wav"] for s in self.queue.data],
                batch_size=4,
                record=record,
            )
            if record.get("collapse", False):
                is_collapse = True
        if is_collapse:
            print("oh no")
