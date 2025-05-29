import os
import numpy as np
import torch
import yaml
import pickle
from tqdm import tqdm

from src.system.suta_new import SUTASystem
from src.utils.tool import wer
from .step_selector import Selector


class TrajectoryAnalyzerOld(object):
    def __init__(self, root: str) -> None:
        self.root = root
        self.config = yaml.load(open(f"{self.root}/config.yaml", "r"), Loader=yaml.FullLoader)
        self.task_name = self.config["task_name"]
        with open(f"{self.root}/result/results.pkl", 'rb') as f:
            self.info = pickle.load(f)

    # collect different stats
    def collect_suta_wer(self, selector: Selector, **kwargs):
        vals = []
        for trajectory in tqdm(self.info["trajectories"]):
            best_steps = selector.select(trajectory, **kwargs)
            vals.append(trajectory["suta-wer"][best_steps[0]])
        return vals
    
    def collect_suta_rescore_wer(self, selector: Selector, **kwargs):
        vals = []
        for trajectory in tqdm(self.info["trajectories"]):
            best_steps = selector.select(trajectory, **kwargs)
            vals.append(trajectory["suta-rescore-wer"][best_steps[0]])
        return vals
    
    def collect_mixed_wer(self, selector: Selector, **kwargs):
        vals = []
        for trajectory in tqdm(self.info["trajectories"]):
            best_steps, rescore_flag = selector.select(trajectory, **kwargs)
            if rescore_flag:
                vals.append(trajectory["suta-rescore-wer"][best_steps[0]])
            else:
                vals.append(trajectory["suta-wer"][best_steps[0]])
        return vals
    
    def collect_merged_score(self, selector: Selector, **kwargs):
        vals = []
        for trajectory in tqdm(self.info["trajectories"]):
            best_steps = selector.select(trajectory, **kwargs)
            vals.append(trajectory["merged_score"][best_steps[0]][0])
        return vals
    
    def collect_best_steps(self, selector: Selector, **kwargs):
        vals = []
        for trajectory in tqdm(self.info["trajectories"]):
            best_steps = selector.select(trajectory, **kwargs)
            if isinstance(best_steps, tuple):
                best_steps = best_steps[0]
            vals.append(best_steps[0])
        return vals


class TrajectoryAnalyzer(object):
    def __init__(self, exp_root) -> None:
        self.root = exp_root
        self.config = yaml.load(open(f"{self.root}/config.yaml", "r"), Loader=yaml.FullLoader)
        self.task_name = self.config["task_name"]
        assert self.config["strategy_name"] == "suta-traj"

        # load trajectory
        with open(f"{self.root}/result/results.pkl", 'rb') as f:
            self.info = pickle.load(f)
        self.size = len(self.info["trajectories"])

        # load system
        self.strategy_config = self.config["strategy_config"]
        self.system = SUTASystem(self.config["system_config"])
        self.system.eval()
        self.num_steps = self.strategy_config["steps"]

        # create cache
        self.cache_dir = f"{exp_root}/_cache"
        os.makedirs(self.cache_dir, exist_ok=True)

    def get_all_beam_inference_results(self):
        try:
            return self._all_beam_inference_results
        except:
            pass
        cached_path = f"{self.cache_dir}/beam-inference.pkl"
        if os.path.exists(cached_path):
            with open(cached_path, "rb") as f:
                self._all_beam_inference_results = pickle.load(f)
            return self._all_beam_inference_results
        
        # generate
        self.log(f"Generate cache to {cached_path}...")
        logs = []
        for idx in tqdm(range(self.size)):
            log = {
                "basename": self.info["basenames"][idx],
                "gt": self.info["transcriptions"][idx][0],
                "am_score": [],
                "lm_score": [],
                "trans": [],
            }
            trajectory = self.info["trajectories"][idx]
            for timestep, logits in enumerate(trajectory["logits"]):
                res = self.system.processor.decode(logits, n_best=5, alpha=0.5, beta=0.0)
                nbest_trans = list(res.text)
                trans = nbest_trans[0]
                am_score = list(res.logit_score)[0]
                lm_score = (list(res.lm_score)[0] - list(res.logit_score)[0]) * 2
                log["am_score"].append(am_score)
                log["lm_score"].append(lm_score)
                log["trans"].append(trans)
            
            logs.append(log)
        
        with open(cached_path, "wb") as f:
            pickle.dump(logs, f)

        self._all_beam_inference_results = logs
        return self._all_beam_inference_results
    
    def get_trans(self, sample_idx: int, step_idx: int, rescored=True) -> float:
        assert sample_idx < self.size and step_idx <= self.num_steps
        if rescored:
            res = self.get_all_beam_inference_results()
            return res[sample_idx]["trans"][step_idx]
        else:
            trajectory = self.info["trajectories"][sample_idx]
            logits = trajectory["logits"][step_idx]
            predicted_ids = torch.argmax(torch.from_numpy(logits), dim=-1)
            return self.system.raw_processor_no_lm.decode(predicted_ids)
    
    def get_wer(self, sample_idx: int, step_idx: int, rescored=True) -> float:
        assert sample_idx < self.size and step_idx <= self.num_steps
        gt = self.info["transcriptions"][sample_idx][0]
        return wer(gt, self.get_trans(sample_idx, step_idx, rescored))
    
    def calc_step_wer(self, step_idx: int, rescored=True) -> float:
        wers, n_words = [], []
        for idx in range(self.size):
            gt = self.info["transcriptions"][idx][0]
            wers.append(self.get_wer(idx, step_idx, rescored))
            n_words.append(len(gt.split(" ")))

        return sum(np.array(wers) * np.array(n_words)) / sum(n_words)

    def get_sample_wers(self, sample_idx, rescored=True) -> list[float]:
        trajectory = self.info["trajectories"][sample_idx]
        return [self.get_wer(sample_idx, j, rescored) for j in range(len(trajectory["logits"]))]
    
    def log(self, x):
        print("[Analyzer]: ", x)
