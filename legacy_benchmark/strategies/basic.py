from torch.utils.data import Dataset
import yaml
from tqdm import tqdm

from systems.suta import SUTASystem
from utils.tool import wer


class BaseStrategy(object):
    def run(self, ds: Dataset):
        raise NotImplementedError
    
    def get_adapt_count(self) -> int:
        return 0


class NoStrategy(BaseStrategy):
    def __init__(self, config) -> None:
        self.config = config
        self.system_config = config["config"]
        self.system = SUTASystem(self.system_config)
        self.system.eval()
    
    def run(self, ds: Dataset):
        basenames = []
        n_words = []
        errs, losses = [], []
        transcriptions = []
        for sample in tqdm(ds):
            n_words.append(len(sample["text"].split(" ")))
            trans = self.system.inference([sample["wav"]])
            err = wer(sample["text"], trans[0])
            errs.append(err)
            transcriptions.append((sample["text"], trans[0]))
            basenames.append(sample["id"])

            # loss
            loss = self.system.calc_suta_loss([sample["wav"]])
            ctc_loss = self.system.calc_ctc_loss([sample["wav"]], [sample["text"]])
            loss["ctc_loss"] = ctc_loss["ctc_loss"]
            losses.append(loss)
        
        return {
            "wers": errs,
            "n_words": n_words,
            "transcriptions": transcriptions,
            "basenames": basenames,
            "losses": losses,
        }
    
    def get_adapt_count(self):
        return self.system.adapt_count


class SUTAStrategy(BaseStrategy):
    def __init__(self, config) -> None:
        self.config = config
        self.system_config = config["config"]
        self.system = SUTASystem(self.system_config)
    
    def _update(self, sample):
        self.system.load_snapshot("init")
        self.system.eval()
        is_collapse = False
        for _ in range(self.system_config["steps"]):
            record = {}
            self.system.suta_adapt(
                wavs=[sample["wav"]],
                record=record,
            )
            if record.get("collapse", False):
                is_collapse = True
        if is_collapse:
            print("oh no")
    
    def run(self, ds: Dataset):
        long_cnt = 0
        n_words = []
        errs, losses = [], []
        transcriptions = []
        for sample in tqdm(ds):
            if len(sample["wav"]) > 50 * 16000:
                long_cnt += 1
                continue
            n_words.append(len(sample["text"].split(" ")))

            # update
            self._update(sample)

            self.system.eval()
            trans = self.system.inference([sample["wav"]])[0]
            err = wer(sample["text"], trans)
            errs.append(err)
            transcriptions.append((sample["text"], trans))
            
            # loss
            # loss = self.system.calc_suta_loss([sample["wav"]])
            # ctc_loss = self.system.calc_ctc_loss([sample["wav"]], [sample["text"]])
            # loss["ctc_loss"] = ctc_loss["ctc_loss"]
            # losses.append(loss)
            
        print(long_cnt)
        
        return {
            "wers": errs,
            "n_words": n_words,
            "transcriptions": transcriptions,
            "losses": losses,
        }
    
    def get_adapt_count(self):
        return self.system.adapt_count

    def load_checkpoint(self, path):
        self.system.load(path)
            

class CSUTAStrategy(SUTAStrategy):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
    
    def _update(self, sample):
        # self.system.load_snapshot("init")
        is_collapse = False
        for _ in range(self.system_config["steps"]):
            record = {}
            self.system.eval()
            self.system.suta_adapt(
                wavs=[sample["wav"]],
                record=record
            )
            if record.get("collapse", False):
                is_collapse = True
        if is_collapse:
            pass
            # print("oh no")


class SDPLStrategy(BaseStrategy):
    def __init__(self, config) -> None:
        self.config = config
        self.system_config = config["config"]
        self.system = SUTASystem(self.system_config)
    
    def _update(self, sample):
        self.system.load_snapshot("init")
        is_collapse = False
        for _ in range(self.system_config["steps"]):
            self.system.eval()
            pl = self.system.inference([sample["wav"]])[0]
            record = {}
            self.system.train()
            self.system.ctc_adapt(
                wavs=[sample["wav"]],
                texts=[pl],
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
        for sample in tqdm(ds):
            n_words.append(len(sample["text"].split(" ")))

            # update
            self._update(sample)

            self.system.eval()
            trans = self.system.inference([sample["wav"]])[0]
            err = wer(sample["text"], trans)
            errs.append(err)
            transcriptions.append((sample["text"], trans))
        
        return {
            "wers": errs,
            "n_words": n_words,
            "transcriptions": transcriptions,
            "losses": losses,
        }
    
    def get_adapt_count(self):
        return self.system.adapt_count
    
    def load_checkpoint(self, path):
        self.system.load(path)
