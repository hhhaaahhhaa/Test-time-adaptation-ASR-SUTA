import os
import torch
from torch.utils.data import Dataset
import yaml
from tqdm import tqdm
import json
from collections import defaultdict

from ...system.suta import SUTASystem, softmax_entropy
from ...utils.tool import wer
from ...utils.distribution import Distribution
from ..base import IStrategy


class OptimalTransportStrategy(IStrategy):

    _src_distribution: Distribution
    _tgt_distribution: Distribution

    def __init__(self, config) -> None:
        self.config = config
        self.strategy_config = config["strategy_config"]
        self.system = SUTASystem(config["system_config"])
        self.system.eval()

        self._log = None

        # load target system 
        self.target_system = SUTASystem(config["system_config"])
        self.target_system.eval()

    def _get_distrubution(self, task_name: str) -> Distribution:
        from src.tasks.load import get_task
        ds = get_task(task_name)
        temperature = self.target_system.config["temp"]
        cache_path = f"results/benchmark/ot/_cache/{task_name}-temp={temperature}.json"
        
        print(f"Collect distribution from task {task_name}, cached at {cache_path}.")
        distribution = Distribution()
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        if os.path.exists(cache_path):
            distribution.load(cache_path)
            return distribution
        for sample in tqdm(ds):
            logits = torch.from_numpy(self.target_system.calc_logits([sample["wav"]]))
            ent = softmax_entropy(logits / temperature)
            # print(ent.shape)  # (1, L)
            for val in ent[0]:
                distribution.add_observation(val.item())
        distribution.refresh()
        distribution.save(cache_path)
        distribution.visualize(cache_path[:-5] + ".jpg", title=cache_path[:-5], value_name="Entropy")
        print(f"Saved distribution to {cache_path}.")

        return distribution

    def _calc_pseudo_entropy_labels(self, sample) -> torch.FloatTensor:
        temperature = self.target_system.config["temp"]
        logits = torch.from_numpy(self.target_system.calc_logits([sample["wav"]]))
        ent = softmax_entropy(logits / temperature)
        pseudo_labels = []
        for val in ent[0]:
            q = self._tgt_distribution.value2quantile(val.item())
            pl = self._src_distribution.quantile2value(q)
            pseudo_labels.append(pl)
        pseudo_labels = torch.Tensor(pseudo_labels).float().reshape(1, -1)

        return pseudo_labels
        
    def _init_start(self, sample) -> None:
        self.system.load_snapshot("init")
    
    def _adapt(self, sample):
        self.system.eval()
        is_collapse = False

        pseudo_entropy_labels = self._calc_pseudo_entropy_labels(sample)
        for _ in range(self.strategy_config["steps"]):
            record = {}
            self.system.match_entropy(
                wavs=[sample["wav"]],
                pseudo_entropy_labels=[pseudo_entropy_labels],
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
        # preparation
        self._src_distribution = self._get_distrubution(task_name="librispeech_random")
        self._tgt_distribution = self._get_distrubution(task_name=self.config["task_name"])

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


class OptimalTransportRescoreStrategy(OptimalTransportStrategy):
    def inference(self, sample) -> str:
        self.system.eval()
        res = self.system.beam_inference([sample["wav"]], n_best=5, text_only=False)
        merged_score = list(res.lm_score)[0]
        self._log["merged_score"].append(merged_score)
        nbest_trans = list(res.text)[0]
        self._log["nbest_trans"].append(nbest_trans)  # not exactly n results due to deduplication
        return nbest_trans[0]


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


class DOptimalTransportStrategy(OptimalTransportStrategy):
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
    
    def _update(self, sample):
        self.memory.update(sample)
        if (self.timestep + 1) % self.update_freq == 0:
            self.slow_system.load_snapshot("start")
            self.slow_system.eval()
            record = {}
            pseudo_entropy_labels = [self._calc_pseudo_entropy_labels(s) for s in self.memory.data]
            self.slow_system.match_entropy_auto(
                wavs=[s["wav"] for s in self.memory.data],
                pseudo_entropy_labels=pseudo_entropy_labels,
                batch_size=1,
                record=record,
            )
            if record.get("collapse", False):
                print("oh no")
            self.slow_system.snapshot("start")
            self.memory.clear()
        self.system.history["start"] = self.slow_system.history["start"]  # fetch start point from slow system
        self.timestep += 1


class DOptimalTransportRescoreStrategy(DOptimalTransportStrategy):
    def inference(self, sample) -> str:
        self.system.eval()
        res = self.system.beam_inference([sample["wav"]], n_best=5, text_only=False)
        merged_score = list(res.lm_score)[0]
        self._log["merged_score"].append(merged_score)
        nbest_trans = list(res.text)[0]
        self._log["nbest_trans"].append(nbest_trans)  # not exactly n results due to deduplication
        return nbest_trans[0]
