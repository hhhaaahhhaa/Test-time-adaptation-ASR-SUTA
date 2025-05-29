import os
import numpy as np
import torch
import torch.nn.functional as F
import pickle
import yaml
import matplotlib.pyplot as plt
from tqdm import tqdm
import scipy.special
import random

from visplot.trajectory_analyzer import TrajectoryAnalyzer


class Indicator(object):
    pass


class ReasonablityV1(Indicator):
    def __init__(self, analyzer: TrajectoryAnalyzer):
        self._analyzer = analyzer
    
    def calc_ctc_loss(self, logits: np.ndarray, trans: str) -> float:
        labels = self._analyzer.system._text_to_model_input([trans]).cpu()
        if labels.shape[1] == 0:  # empty string exception
            labels = torch.zeros((len(labels), 1))
        log_probs = torch.from_numpy(logits).log_softmax(1)
        ctc_loss = F.ctc_loss(log_probs, labels[0], input_lengths=(len(log_probs),), target_lengths=(len(labels[0]),))
        return ctc_loss.item()

    def __call__(self, sample_idx: int, step_idx: int) -> float:
        # probability
        trajectory = self._analyzer.info["trajectories"][sample_idx]
        logits = trajectory["logits"][step_idx]
        acoustic_score = np.sum(np.max(scipy.special.log_softmax(logits, axis=-1), axis=-1))
        normalized_acoustic_score = acoustic_score / logits.shape[0]

        # trans
        greedy_trans = self._analyzer.get_trans(sample_idx, step_idx, rescored=False)
        greedy_lm_score = self._analyzer.system.calc_lm_score(greedy_trans)
        normalized_greedy_lm_score = greedy_lm_score / (len(greedy_trans.split(" ")) + 1)

        rescored_trans = self._analyzer.get_trans(sample_idx, step_idx, rescored=True)
        rescored_lm_score = self._analyzer.system.calc_lm_score(rescored_trans)
        normalized_rescored_lm_score = rescored_lm_score / (len(rescored_trans.split(" ")) + 1)

        # ctc
        greedy_ctc_loss = self.calc_ctc_loss(logits, greedy_trans)
        rescored_ctc_loss = self.calc_ctc_loss(logits, rescored_trans)
        
        return normalized_acoustic_score * 20, (
            f"{normalized_acoustic_score * 20:.4f}",
            f"{normalized_greedy_lm_score:.4f}",
            f"{normalized_rescored_lm_score:.4f}",
            f"{greedy_ctc_loss:.4f}",
            f"{rescored_ctc_loss:.4f}",
        )


class GreedyScore(Indicator):
    def __init__(self, analyzer: TrajectoryAnalyzer):
        self._analyzer = analyzer
    
    def __call__(self, sample_idx: int, step_idx: int) -> float:
        greedy_trans = self._analyzer.get_trans(sample_idx, step_idx, rescored=False)
        return self._analyzer.system.calc_lm_score(greedy_trans)


class BeamScore(Indicator):
    def __init__(self, analyzer: TrajectoryAnalyzer):
        self._analyzer = analyzer
    
    def __call__(self, sample_idx: int, step_idx: int) -> float:
        res = self._analyzer.get_all_beam_inference_results()
        return res[sample_idx]["am_score"][step_idx] + 0.5 * res[sample_idx]["lm_score"][step_idx]


class BeamLM(Indicator):
    def __init__(self, analyzer: TrajectoryAnalyzer):
        self._analyzer = analyzer
    
    def __call__(self, sample_idx: int, step_idx: int) -> float:
        res = self._analyzer.get_all_beam_inference_results()
        return res[sample_idx]["lm_score"][step_idx]


class Oracle(Indicator):
    def __init__(self, analyzer: TrajectoryAnalyzer):
        self._analyzer = analyzer
    
    def __call__(self, sample_idx: int, step_idx: int) -> float:
        return -self._analyzer.get_wer(sample_idx, step_idx, rescored=True)


class MaxProb(Indicator):
    def __init__(self, analyzer: TrajectoryAnalyzer):
        self._analyzer = analyzer
    
    def __call__(self, sample_idx: int, step_idx: int=0) -> float:
        # probability
        trajectory = self._analyzer.info["trajectories"][sample_idx]
        logits = trajectory["logits"][step_idx]
        acoustic_score = np.sum(np.max(scipy.special.log_softmax(logits, axis=-1), axis=-1))
        normalized_acoustic_score = acoustic_score / logits.shape[0]
        return normalized_acoustic_score


class FSStrategy(object):
    def __init__(self, analyzer: TrajectoryAnalyzer):
        self.analyzer = analyzer
        self.filtering_indicator = MaxProb(analyzer)
        # self.selecting_indicator = Oracle(analyzer)
        self.selecting_indicator = GreedyScore(analyzer)
    
    def filter(self, sample_idx: int, threshold: float) -> list[int]:
        vals = []
        details = []
        all_steps = list(range(self.analyzer.num_steps + 1))
        for j in all_steps:
            val = self.filtering_indicator(sample_idx, j)
            # val, detail = self.filtering_indicator(sample_idx, j)
            vals.append(val)
            # details.append(detail)

        # logging
        # errs = self.analyzer.get_sample_wers(sample_idx, rescored=True)
        # mi = min(errs)
        # best_steps = np.argwhere(np.array(errs) == mi).reshape(-1)
        # print("GT: ", self.analyzer.info["transcriptions"][sample_idx][0])
        # print("best steps: ", best_steps)
        
        # for step_idx in range(len(errs)):
        #     greedy_trans = self.analyzer.get_trans(sample_idx, step_idx, rescored=False)
        #     rescored_trans = self.analyzer.get_trans(sample_idx, step_idx, rescored=True)
        #     print(f"{details[step_idx]}, {errs[step_idx]*100:.2f}%")
        #     print("G: ", greedy_trans)
        #     print("R: ", rescored_trans)
        
        res = []
        for j in all_steps:
            if vals[j] >= threshold:
                res.append(j)
        if not res:
            res = [all_steps[-1]]
        
        return res
    
    def select(self, sample_idx: int, filtered_steps: list[int]) -> int:
        tmp = [(j, self.selecting_indicator(sample_idx, j)) for j in filtered_steps]
        tmp = sorted(tmp, key=lambda x: (-x[1], x[0]))
        selected_step_idx = tmp[0][0]
        self.step_cnt += selected_step_idx
        return selected_step_idx

    def random_select(self, sample_idx: int, filtered_steps: list[int]) -> int:
        selected_step_idx = random.choice(filtered_steps)
        return selected_step_idx
    
    def random_tie_break(self, sample_idx: int, filtered_steps: list[int]) -> int:
        tmp = [(j, self.selecting_indicator(sample_idx, j)) for j in filtered_steps]
        tmp = sorted(tmp, key=lambda x: (-x[1], x[0]))
        mx = tmp[0][1]
        candidates = []
        for tup in tmp:
            if tup[1] == mx:
                candidates.append(tup[0])
        selected_step_idx = random.choice(candidates)
        return selected_step_idx
    
    def early_stop_select(self, sample_idx: int, filtered_steps: list[int], patience=3) -> int:
        patience_cnt = 0 
        best_score = -2e9
        selected_step_idx = -1
        for j in filtered_steps:
            score = self.selecting_indicator(sample_idx, j)
            if score > best_score:
                best_score, selected_step_idx = score, j
                patience_cnt = 0
            else:
                patience_cnt += 1
            if patience_cnt == patience:
                break
        # self.step_cnt += j  # used
        self.step_cnt += selected_step_idx
        return selected_step_idx

    def run(self, *args, **kwargs) -> dict:
        wers, n_words = [], []
        coverage = 0
        self.step_cnt = 0
        for sample_idx in tqdm(range(self.analyzer.size)):
            # filter and select
            filtered_steps = self.filter(sample_idx, *args, **kwargs)
            coverage += len(filtered_steps)
            #  selected_step_idx = self.select(sample_idx, filtered_steps)
            # selected_step_idx = self.random_select(sample_idx, filtered_steps)
            selected_step_idx = self.random_tie_break(sample_idx, filtered_steps)
            # selected_step_idx = self.early_stop_select(sample_idx, filtered_steps)

            wers.append(self.analyzer.get_wer(sample_idx, step_idx=selected_step_idx, rescored=True))
            gt = self.analyzer.info["transcriptions"][sample_idx][0]
            n_words.append(len(gt.split(" ")))
        print("Avg Step: ", self.step_cnt / self.analyzer.size)

        return {
            "wer": sum(np.array(wers) * np.array(n_words)) / sum(n_words),
            "coverage": coverage / (self.analyzer.num_steps + 1) / self.analyzer.size,
        }
    

if __name__ == "__main__":
    for ts in [
        "LS_AA_10", "LS_AC_10", "LS_BA_10", "LS_CM_10", "LS_GS_10",
        "LS_MU_10", "LS_NB_10", "LS_SD_10", "LS_TP_10", "LS_VC_10",
        "L2_Arabic", "L2_Chinese", "L2_Hindi", "L2_Korean", "L2_Spanish", "L2_Vietnamese",
        "ted_random", "chime_random",
        "cv-eng", "cv-aus", "cv-ind", "cv-sco", "cv-ire",
    ]:  
        # subset
        if ts not in ["LS_GS_10", "LS_NB_10", "L2_Korean", "L2_Spanish", "ted_random", "chime_random"]:
            continue
        print("Task: ", ts)
    # for exp_root in [
    #     "results/benchmark/suta-traj/step=20/cv-eng",
    #     "results/benchmark/suta-traj/step=20/LS_GS_10",
    #     "results/benchmark/suta-traj/step=20/LS_TP_10",
    #     "results/benchmark/suta-traj/step=20/L2_Korean",
    #     "results/benchmark/suta-traj/step=20/chime_random",
    #     "results/benchmark/suta-traj/data2vec-step=20/cv-eng",
    #     "results/benchmark/suta-traj/data2vec-step=20/LS_GS_10",
    #     "results/benchmark/suta-traj/data2vec-step=20/LS_TP_10",
    #     "results/benchmark/suta-traj/data2vec-step=20/L2_Korean",
    #     "results/benchmark/suta-traj/data2vec-step=20/chime_random",
    #     "results/benchmark/suta-traj/hubert-large-step=20/cv-eng",
    #     "results/benchmark/suta-traj/hubert-large-step=20/LS_GS_10",
    #     "results/benchmark/suta-traj/hubert-large-step=20/LS_TP_10",
    #     "results/benchmark/suta-traj/hubert-large-step=20/L2_Korean",
    #     "results/benchmark/suta-traj/hubert-large-step=20/chime_random",
    # ]:
        for i in range(3):
            exp_root = "results/benchmark/suta-traj/step=20/" + ts
            a, b = exp_root.rsplit("/", 1)
            analyzer = TrajectoryAnalyzer(exp_root=f"{a}-{i}/{b}")
            strategy = FSStrategy(analyzer)
            res = strategy.run(threshold=-0.05)
            # res = strategy.run(threshold=-1000)
            # print("Threshold: ", threshold)
            print(f"WER: {res['wer']*100:.2f}%")
            # print(f"Coverage: {res['coverage']*100:.2f}%")
        print()
