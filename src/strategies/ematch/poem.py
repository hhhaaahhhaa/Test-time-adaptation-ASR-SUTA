from typing import Callable
import torch
from torch.utils.data import Dataset
import yaml
from tqdm import tqdm
from collections import defaultdict
import math

from ...system.suta import SUTASystem
from ...utils.tool import wer
from ..base import IStrategy


class Buffer(object):
    """ Easy implementation of buffer, use .data to access all data. """

    data: list

    def __init__(self, max_size: int=100) -> None:
        self.max_size = max_size
        self.data = []

    def update(self, x):
        self.data.append(x)
        if len(self.data) > self.max_size:
            self.data.pop(0)
    
    def clear(self):
        self.data.clear()


class POEMMartingale(object):
    def __init__(self, src_cdf: Callable[[float], float], max_bet=1.8, lr=0.0722):
        self.src_cdf = src_cdf
        self.max_bet = max_bet
        self.lr = lr
        self.betting_var = 0.0
        self.running_demom = 1e-8
        self.gain = 0.0

    def forward(self, observation: float):
        """
        The purpose of this test martingale is to validate that u_t follows uniform distribution.
        This implies that observation follows src distribution!
        However, it is hard to ensure this, here we only constrain u_t to have mean=0.5.
        """
        assert 0 <= observation <= 1
        u = self.src_cdf(observation)
        E = self.max_bet if u >= 0.5 else -self.max_bet
        if self.betting_var * E > 0 and abs(self.betting_var) > self.max_bet:
            self.gain = self.gain + math.log(1 + E * (u - 0.5))
            grad = 0
        else:
            self.gain = self.gain + math.log(1 + self.betting_var * (u - 0.5))
            grad = -(u - 0.5) / (1 + self.betting_var * (u - 0.5))

        # Optimization
        self.running_demom = self.running_demom + grad ** 2
        self.betting_var = self.betting_var - self.lr * grad / math.sqrt(self.running_demom)
        self.betting_var = max(min(self.betting_var, self.max_bet), -self.max_bet)  # clip
    
    def show(self):
        print("Betting Var: ", self.betting_var)
        print("Log-scale gain: ", self.gain)


class DPOEMUpperStrategy(IStrategy):
    def __init__(self, config) -> None:
        self.config = config
        self.strategy_config = config["strategy_config"]
        self.system = SUTASystem(config["system_config"])
        self.system.eval()

        # Set slow system
        self.slow_system = SUTASystem(config["system_config"])
        self.slow_system.eval()
        self.slow_system.snapshot("start")
        self.timestep = 0
        self.update_freq = config["strategy_config"]["update_freq"]
        self.memory = Buffer(max_size=config["strategy_config"]["memory"])

        self.system.snapshot("start")

        self._log = None

        # load target system 
        self.target_system = SUTASystem(config["system_config"])
        self.target_system.eval()

    def _init_start(self, sample):
        self.system.load_snapshot("start")

    def _adapt(self, sample, target_sample):
        self.system.eval()
        is_collapse = False
        for _ in range(self.strategy_config["steps"]):
            record = {}
            self.system.match_entropy(
                wavs=[sample["wav"]],
                target_logits=[torch.from_numpy(self.target_system.calc_logits([target_sample["wav"]]))],
                record=record,
            )
            if record.get("collapse", False):
                is_collapse = True
        if is_collapse:
            print("oh no")
    
    def _update(self, sample, target_sample):
        target_logits = torch.from_numpy(self.target_system.calc_logits([target_sample["wav"]]))
        self.memory.update((sample, target_sample))
        if (self.timestep + 1) % self.update_freq == 0:
            self.slow_system.load_snapshot("start")
            self.slow_system.eval()
            record = {}
            target_logits = [torch.from_numpy(self.target_system.calc_logits([s[1]["wav"]])) for s in self.memory.data]
            self.slow_system.match_entropy_auto(
                wavs=[s[0]["wav"] for s in self.memory.data],
                target_logits=target_logits,
                batch_size=1,
                record=record,
            )
            if record.get("collapse", False):
                print("oh no")
            self.slow_system.snapshot("start")
            self.memory.clear()
        self.system.history["start"] = self.slow_system.history["start"]  # fetch start point from slow system

    def inference(self, sample) -> str:
        self.system.eval()
        trans = self.system.inference([sample["wav"]])[0]
        return trans
    
    def run(self, ds: Dataset):
        long_cnt = 0
        self._log = defaultdict(list)
        for (target_sample, sample) in tqdm(ds):
            if len(sample["wav"]) > self.strategy_config["max_length"]:
                long_cnt += 1
                continue
            self._log["n_words"].append(len(sample["text"].split(" ")))

            self._init_start(sample)
            self._adapt(sample, target_sample)

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

            self._update(sample, target_sample)
            self.timestep += 1
            
        print("#Too long: ", long_cnt)
        
        return self._log
        
    def get_adapt_count(self):
        return self.system.adapt_count
    