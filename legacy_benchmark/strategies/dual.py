from torch.utils.data import Dataset
from tqdm import tqdm

from dlhlp_lib.utils.data_structure import Queue

from systems.suta import SUTASystem
from utils.tool import wer
from .basic import BaseStrategy


class DualStrategy(BaseStrategy):
    def __init__(self, config) -> None:
        self.config = config
        self.system_config = config["config"]
        self.system = SUTASystem(self.system_config)
        
        self.slow_system = SUTASystem(self.system_config)
        self.slow_system.snapshot("start")
        self.system.snapshot("start")
        self.timestep = 0
        self.update_freq = config["strategy_config"]["update_freq"]
        self.memory = Queue(max_size=config["strategy_config"]["memory"])
        self.reset = config["strategy_config"].get("reset", False)

    def _load_start_point(self, sample):
        self.system.load_snapshot("start")

    def _adapt(self, sample):
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
    
    def _update(self, sample):
        self.memory.update(sample)
        if self.timestep % self.update_freq == 0:
            self.slow_system.load_snapshot("start")
            self.slow_system.eval()
            record = {}
            self.slow_system.suta_adapt_auto(
                wavs=[s["wav"] for s in self.memory.data],
                batch_size=1,
                record=record,
            )
            if record.get("collapse", False):
                print("oh no")
            self.slow_system.snapshot("start")
            self.memory.clear()
        self.system.history["start"] = self.slow_system.history["start"]  # fetch start point from slow system

    def reset_strategy(self):
        self.memory.clear()
        self.slow_system.load_snapshot("init")
        self.system.load_snapshot("init")
        self.slow_system.snapshot("start")
        self.system.snapshot("start")
        
    def run(self, ds: Dataset):
        long_cnt = 0
        n_words = []
        errs, losses = [], []
        transcriptions = []
        for sample in tqdm(ds):
            if len(sample["wav"]) > 20 * 16000:
                long_cnt += 1
                continue
            n_words.append(len(sample["text"].split(" ")))

            # update
            if self.reset and self.timestep in ds.task_boundaries:
                print("Reset at boundary...")
                self.reset_strategy()
            self._load_start_point(sample)
            self._adapt(sample)

            self.system.eval()
            trans = self.system.inference([sample["wav"]])
            err = wer(sample["text"], trans[0])
            errs.append(err)
            transcriptions.append((sample["text"], trans[0]))

            # # loss
            # loss = self.system.calc_suta_loss([sample["wav"]])
            # ctc_loss = self.system.calc_ctc_loss([sample["wav"]], [sample["text"]])
            # loss["ctc_loss"] = ctc_loss["ctc_loss"]
            # losses.append(loss)

            self.timestep += 1
            self._update(sample)

        print(long_cnt)
        
        return {
            "wers": errs,
            "n_words": n_words,
            "transcriptions": transcriptions,
            "losses": losses,
        }
    
    def get_adapt_count(self):
        return self.system.adapt_count + self.slow_system.adapt_count


# class DualPLStrategy(BaseStrategy):
#     def __init__(self, config) -> None:
#         self.config = config
#         self.system_config = config["config"]
#         self.system = SUTASystem(self.system_config)
        
#         # setup anchor
#         self.anchor = SUTASystem(self.system_config)
#         self.anchor.eval()
#         for param in self.anchor.model.parameters():
#             param.detach_()

#         self.slow_system = SUTASystem(self.system_config)
#         self.slow_system.snapshot("start")
#         self.system.snapshot("start")
#         self.timestep = 0
#         self.update_freq = config["strategy_config"]["update_freq"]
#         self.memory = Queue(max_size=config["strategy_config"]["memory"] * 2)
#         self.warmup = config["strategy_config"]["warmup"]

#     def _load_start_point(self, sample):
#         if self.timestep >= self.warmup:
#             self.system.load_snapshot("start")
#         else:
#             self.system.load_snapshot("init")

#     def _adapt(self, sample):
#         self.system.eval()
#         is_collapse = False
#         for _ in range(self.system_config["steps"]):
#             record = {}
#             self.system.suta_adapt(
#                 wavs=[sample["wav"]],
#                 record=record,
#             )
#             if record.get("collapse", False):
#                 is_collapse = True
#         if is_collapse:
#             print("oh no")
#         pl = self.system.inference([sample["wav"]])[0]
#         anchor_pl = self.anchor.inference([sample["wav"]])[0]
#         # if len(sample["wav"]) <= 15 * 16000:
#         self.memory.update({
#             "wav": sample["wav"],
#             "text": pl
#         })
#         self.memory.update({
#             "wav": sample["wav"],
#             "text": anchor_pl
#         })
#         self.timestep += 1
    
#     def _update(self, sample):
#         if self.timestep % self.update_freq != 0:
#             return
#         self.slow_system.load_snapshot("start")
#         self.slow_system.train()
#         record = {}
#         self.slow_system.ctc_adapt_auto(
#             wavs=[s["wav"] for s in self.memory.data],
#             texts=[s["text"] for s in self.memory.data],
#             record=record,
#             batch_size=self.update_freq
#         )
#         if record.get("collapse", False):
#             print("oh no")
#         self.slow_system.snapshot("start")
#         self.system.history["start"] = self.slow_system.history["start"]  # fetch start point from slow system

#     def run(self, ds: Dataset):
#         long_cnt = 0
#         n_words = []
#         errs, losses = [], []
#         transcriptions = []
#         for sample in tqdm(ds):
#             if len(sample["wav"]) > 20 * 16000:
#                 long_cnt += 1
#                 continue
#             n_words.append(len(sample["text"].split(" ")))

#             # update
#             self._load_start_point(sample)
#             self._adapt(sample)
#             self._update(sample)

#             self.system.eval()
#             trans = self.system.inference([sample["wav"]])
#             err = wer(sample["text"], trans[0])
#             errs.append(err)
#             transcriptions.append((sample["text"], trans[0]))
#         print(long_cnt)
        
#         return {
#             "wers": errs,
#             "n_words": n_words,
#             "transcriptions": transcriptions,
#             "losses": losses,
#         }
    
#     def get_adapt_count(self):
#         return self.system.adapt_count + self.slow_system.adapt_count
