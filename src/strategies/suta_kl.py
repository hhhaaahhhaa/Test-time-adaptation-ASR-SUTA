import os
import numpy as np
import torch
from torch.utils.data import Dataset
import yaml
from tqdm import tqdm
from collections import defaultdict

from ..system.suta_new import SUTASystem, softmax_entropy
from ..utils.tool import wer
from ..utils.distribution import Distribution
from .base import IStrategy


class SUTAKLStrategy(IStrategy):

    _src_distribution: Distribution

    def __init__(self, config) -> None:
        self.config = config
        self.strategy_config = config["strategy_config"]
        self.system = SUTASystem(config["system_config"])
        self.system.eval()

        self.use_valid = self.strategy_config.get("use_kl_valid", False)

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
    
    def _init_start(self, sample) -> None:
        self.system.load_snapshot("init")
    
    def _adapt(self, sample):
        self.system.eval()
        is_collapse = False

        if self.use_valid:
            score = self.system.calc_kl_loss([sample["wav"]], self._src_distribution)
            best_score, best_step = score, 0
            self.system.snapshot("best")
            patience_cnt = 0
        
        for idx in range(self.strategy_config["steps"]):
            record = {}
            self.system.suta_adapt(
                wavs=[sample["wav"]],
                record=record,
                distribution=self._src_distribution
            )
            if record.get("collapse", False):
                is_collapse = True

            # validation
            if self.use_valid:
                self.system.adapt_count -= 1  # control adapt count and increase later
                score = self.system.calc_kl_loss([sample["wav"]], self._src_distribution)
                if score < best_score:
                    best_score, best_step = score, idx + 1
                    self.system.snapshot("best")
                    patience_cnt = 0
                else:
                    patience_cnt += 1
            
                # early stop
                if "kl_patience" in self.strategy_config and patience_cnt == self.strategy_config["kl_patience"]:
                    break

        if is_collapse:
            print("oh no")

        if self.use_valid:
            self.system.load_snapshot("best")
            self.system.adapt_count += best_step  # increase the count here
            self._log["best_steps"].append(best_step)

    def _update(self, sample):
        pass
    
    def inference(self, sample) -> str:
        self.system.eval()
        res = self.system.beam_inference([sample["wav"]], n_best=5, text_only=False)
        merged_score = list(res.lm_score)[0]
        self._log["merged_score"].append(merged_score)
        nbest_trans = list(res.text)[0]
        self._log["nbest_trans"].append(nbest_trans)  # not exactly n results due to deduplication
        return nbest_trans[0]
    
    def run(self, ds: Dataset):
        # preparation
        self._src_distribution = self._get_distrubution(task_name="librispeech_random")

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
