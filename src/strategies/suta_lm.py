import os
import numpy as np
import torch
from torch.utils.data import Dataset
import yaml
from tqdm import tqdm
from collections import defaultdict

from ..system.suta_new import SUTASystem
from ..utils.tool import wer
from .base import IStrategy


class SUTALMStrategy(IStrategy):
    """ Maximum acoustic + LM score, acoustic score is from beam score """
    def __init__(self, config) -> None:
        self.config = config
        self.strategy_config = config["strategy_config"]
        self.system = SUTASystem(config["system_config"])
        self.system.eval()

        self._log = None
    
    def calc_score(self, sample, trajectory):
        res = self.system.beam_inference([sample["wav"]], n_best=5, text_only=False)
        nbest_trans = list(res.text)
        trans = nbest_trans[0]
        acoustic_score = list(res.logit_score)[0]
        lm_score = (list(res.lm_score)[0] - list(res.logit_score)[0]) * 2
        alpha = self.strategy_config.get("lm_ratio", 0.333333)
        trajectory.append((acoustic_score, lm_score, trans))
        return (1 - alpha) * acoustic_score + alpha * lm_score, nbest_trans

    def _init_start(self, sample) -> None:
        self.system.load_snapshot("init")

    def _adapt(self, sample):
        self.system.eval()
        is_collapse = False

        trajectory = []
        score, nbest_trans = self.calc_score(sample, trajectory)
        best_score, best_step = score, 0
        best_nbest_trans = nbest_trans
        patience_cnt = 0
        
        for idx in range(self.strategy_config["steps"]):
            record = {}
            self.system.suta_adapt(
                wavs=[sample["wav"]],
                record=record,
            )
            if record.get("collapse", False):
                is_collapse = True

            # validation
            self.system.adapt_count -= 1  # control adapt count and increase later
            score, nbest_trans = self.calc_score(sample, trajectory)
            if score > best_score:
                best_score, best_step = score, idx + 1
                best_nbest_trans = nbest_trans
                patience_cnt = 0
            else:
                patience_cnt += 1
        
            # early stop
            if "kl_patience" in self.strategy_config and patience_cnt == self.strategy_config["kl_patience"]:
                break

        if is_collapse:
            print("oh no")

        self.system.adapt_count += best_step  # increase the count here
        self._log["best_steps"].append(best_step)
        self._log["nbest_trans"].append(best_nbest_trans)
        self._log["trajectories"].append(trajectory)

        return best_nbest_trans[0]

    def _update(self, sample):
        pass

    def run(self, ds: Dataset):
        long_cnt = 0
        self._log = defaultdict(list)
        for sample in tqdm(ds):
            if len(sample["wav"]) > self.strategy_config["max_length"]:
                long_cnt += 1
                continue
            self._log["n_words"].append(len(sample["text"].split(" ")))

            self._init_start(sample)
            trans = self._adapt(sample)

            # trans = self.inference(sample)
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


class SUTALM2Strategy(SUTALMStrategy):
    """ Maximum acoustic + LM score, acoustic score is real CTC score instead of beam score """
    def calc_score(self, sample, trajectory):
        res = self.system.beam_inference([sample["wav"]], n_best=5, text_only=False)
        nbest_trans = list(res.text)
        trans = nbest_trans[0]
        acoustic_score = -self.system.calc_ctc_loss([sample["wav"]], [trans])["ctc_loss"]
        lm_score = (list(res.lm_score)[0] - list(res.logit_score)[0]) * 2
        alpha = self.strategy_config.get("lm_ratio", 0.333333)
        trajectory.append((acoustic_score, lm_score, trans))
        return (1 - alpha) * acoustic_score + alpha * lm_score, nbest_trans


class SUTAMLStrategy(SUTALMStrategy):
    """ Maximum Likelihood """    
    def calc_score(self, sample):
        trans = self.system.inference([sample["wav"]])[0]
        return self.system.calc_probability([sample["wav"]]), trans

    def _adapt(self, sample):
        self.system.eval()
        is_collapse = False

        score, trans = self.calc_score(sample)
        best_score, best_step = score, 0
        best_trans = trans
        patience_cnt = 0
        
        for idx in range(self.strategy_config["steps"]):
            record = {}
            self.system.suta_adapt(
                wavs=[sample["wav"]],
                record=record,
            )
            if record.get("collapse", False):
                is_collapse = True

            # validation
            self.system.adapt_count -= 1  # control adapt count and increase later
            score, trans = self.calc_score(sample)
            if score > best_score:
                best_score, best_step = score, idx + 1
                best_trans = trans
                patience_cnt = 0
            else:
                patience_cnt += 1
        
            # early stop
            if "kl_patience" in self.strategy_config and patience_cnt == self.strategy_config["kl_patience"]:
                break

        if is_collapse:
            print("oh no")

        self.system.adapt_count += best_step  # increase the count here
        self._log["best_steps"].append(best_step)

        return best_trans
