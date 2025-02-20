from torch.utils.data import Dataset
import yaml
from tqdm import tqdm
from collections import defaultdict

from ..system.suta import SUTASystem
from ..utils.tool import wer
from .base import IStrategy


class NoStrategy(IStrategy):
    def __init__(self, config) -> None:
        self.config = config
        self.strategy_config = config["strategy_config"]
        self.system = SUTASystem(config["system_config"])
        self.system.eval()
    
    def run(self, ds: Dataset):
        long_cnt = 0
        basenames = []
        n_words = []
        errs, losses = [], []
        transcriptions = []
        logits = []
        for sample in tqdm(ds):
            if len(sample["wav"]) > self.strategy_config["max_length"]:
                long_cnt += 1
                continue
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

            logits.append(self.system.calc_logits([sample["wav"]])[0])
        
        print("#Too long: ", long_cnt)
        
        return {
            "wers": errs,
            "n_words": n_words,
            "transcriptions": transcriptions,
            "basenames": basenames,
            "losses": losses,
            "logits": logits,
        }
    
    def get_adapt_count(self):
        return self.system.adapt_count


class SUTAStrategy(IStrategy):
    def __init__(self, config) -> None:
        self.config = config
        self.strategy_config = config["strategy_config"]
        self.system = SUTASystem(config["system_config"])
        self.system.eval()
    
    def _init_start(self, sample) -> None:
        self.system.load_snapshot("init")
    
    def _adapt(self, sample):
        self.system.eval()
        is_collapse = False
        for _ in range(self.strategy_config["steps"]):
            record = {}
            self.system.suta_adapt(
                wavs=[sample["wav"]],
                record=record,
            )
            if record.get("collapse", False):
                is_collapse = True
        if is_collapse:
            print("oh no")

    def _update(self, sample):
        pass
    
    def inference(self, sample) -> str:
        self.system.eval()
        trans = self.system.inference([sample["wav"]])[0]
        return trans
    
    def run(self, ds: Dataset):
        long_cnt = 0
        self._log = defaultdict(list)
        for sample in tqdm(ds):
            if len(sample["wav"]) > self.strategy_config["max_length"]:
                long_cnt += 1
                continue
            self._log["n_words"].append(len(sample["text"].split(" ")))

            self._init_start(sample)
            self._adapt(sample)

            trans = self.inference(sample)
            err = wer(sample["text"], trans)
            self._log["wers"].append(err)
            self._log["transcriptions"].append((sample["text"], trans))
            self._log["basenames"].append(sample["id"])

            # loss
            # loss = self.system.calc_suta_loss([sample["wav"]])
            # ctc_loss = self.system.calc_ctc_loss([sample["wav"]], [sample["text"]])
            # loss["ctc_loss"] = ctc_loss["ctc_loss"]
            # self._log["losses"].append(loss)

            self._log["logits"].append(self.system.calc_logits([sample["wav"]])[0])

            self._update(sample)
            
        print("#Too long: ", long_cnt)
        
        return self._log
    
    def get_adapt_count(self):
        return self.system.adapt_count

    def load_checkpoint(self, path):
        self.system.load(path)
            

class CSUTAStrategy(SUTAStrategy):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
    
    def _init_start(self, sample) -> None:
        pass


class SDPLStrategy(SUTAStrategy):
    def __init__(self, config) -> None:
        self.config = config
        self.strategy_config = config["strategy_config"]
        self.system = SUTASystem(config["system_config"])
    
    def _adapt(self, sample):
        is_collapse = False
        for _ in range(self.strategy_config["steps"]):
            self.system.eval()
            pl = self.system.inference([sample["wav"]])[0]
            record = {}
            self.system.train()  # gradient update under train mode (SUTA is eval mode according to origin implementation)
            self.system.ctc_adapt(
                wavs=[sample["wav"]],
                texts=[pl],
                record=record,
            )
            if record.get("collapse", False):
                is_collapse = True
        if is_collapse:
            print("oh no")
