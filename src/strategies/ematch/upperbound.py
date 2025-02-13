import os
import torch
from torch.utils.data import Dataset
import yaml
from tqdm import tqdm
import json
from collections import defaultdict

from ...system.suta import SUTASystem
from ...utils.tool import wer
from ..base import IStrategy


class POEMUpperStrategy(IStrategy):
    def __init__(self, config) -> None:
        self.config = config
        self.strategy_config = config["strategy_config"]
        self.system = SUTASystem(config["system_config"])
        self.system.eval()

        self._log = None

        # load target system 
        self.target_system = SUTASystem(config["system_config"])
        self.target_system.eval()
    
    def _init_start(self, sample) -> None:
        self.system.load_snapshot("init")
    
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

    def _update(self, sample):
        pass
    
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

            self._update(sample)
            
        print("#Too long: ", long_cnt)
        
        return self._log
        
    def get_adapt_count(self):
        return self.system.adapt_count


class POEMUpperRescoreStrategy(POEMUpperStrategy):
    def inference(self, sample) -> str:
        self.system.eval()
        res = self.system.beam_inference([sample["wav"]], n_best=5, text_only=False)
        merged_score = list(res.lm_score)[0]
        self._log["merged_score"].append(merged_score)
        nbest_trans = list(res.text)[0]
        self._log["nbest_trans"].append(nbest_trans)  # not exactly n results due to deduplication
        return nbest_trans[0]
