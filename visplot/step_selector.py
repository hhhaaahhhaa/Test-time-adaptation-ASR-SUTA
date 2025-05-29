import numpy as np
import torch

from src.utils.distribution import Distribution, kl_divergence
from src.system.suta import softmax_entropy


class Selector(object):

    def select(self, trajectory) -> list[int]:
        raise NotImplementedError


class ConstantSelector(Selector):

    def select(self, trajectory, step: int=0) -> list[int]:
        return [step]


class OracleSelector(Selector):
    def __init__(self, key="suta-wer"):
        self.key = key

    def select(self, trajectory) -> list[int]:
        wers = trajectory[self.key]
        wers = np.array(wers)
        mi_idx = np.argmin(wers)
        best_steps = np.argwhere(wers == wers[mi_idx]).reshape(-1)
        return best_steps


class MergedScoreSelector(Selector):
    def select(self, trajectory) -> list[int]:
        best_idx = np.argmax([ms[0] for ms in trajectory["merged_score"]])
        return [best_idx]


class MeanMatchSelector(Selector):
    def __init__(self, temperature=1.0):
        self.temperature = temperature

    def select(self, trajectory, mu=0.0) -> list[int]:
        selection_scores = []
        for logits in trajectory["logits"]:
            logits = torch.from_numpy(logits).unsqueeze(0)
            ent = softmax_entropy(logits / self.temperature)
            mu_q = torch.mean(ent).item()
            selection_scores.append(-(mu - mu_q) ** 2)
        best_idx = np.argmax(selection_scores)
        return [best_idx]


class Exp1Selector(Selector):
    def count_flips(self, trajectory) -> int:
        seq = trajectory["suta-wer"]
        cnt, cur = -1, -1
        for idx, val in enumerate(seq):
            if val != cur:
                cnt += 1
                last_flip = idx
            cur = val
        return cnt, last_flip

    def select(self, trajectory) -> list[int]:
        merged_scores = np.array([ms[0] for ms in trajectory["merged_score"]])
        logit_scores = np.array([ls[0] for ls in trajectory["logit_score"]])
        lm_scores = merged_scores - logit_scores
        flips, last_flip = self.count_flips(trajectory)
        if flips >= 11:  # dummy
            raise NotImplementedError
        elif flips >= 0:
            scores = lm_scores
            best_idx = np.argmax(scores)
        # elif flips == 0:
        #     scores = logit_scores
        #     best_idx = np.argmax(scores)
        else:
            scores = merged_scores
            best_idx = np.argmax(scores)
        return [best_idx]


class Exp2Selector(Selector):
    def count_flips(self, trajectory) -> int:
        seq = trajectory["suta-wer"]
        cnt, cur = -1, -1
        for idx, val in enumerate(seq):
            if val != cur:
                cnt += 1
                last_flip = idx
            cur = val
        return cnt, last_flip

    def select(self, trajectory) -> list[int]:
        merged_scores = np.array([ms[0] for ms in trajectory["merged_score"]])
        logit_scores = np.array([ls[0] for ls in trajectory["logit_score"]])
        lm_scores = merged_scores - logit_scores
        flips, last_flip = self.count_flips(trajectory)
        rescore_flag = True
        if flips >= 11:  # dummy
            raise NotImplementedError
        # elif flips >= 5:
        #     scores = lm_scores
        #     best_idx = np.argmax(scores)
        # elif flips <= 1:
        #     best_idx = last_flip
        else:
            # scores = merged_scores
            # scores = 0.1 * logit_scores + 1.9 * lm_scores
            scores = logit_scores
            best_idx = np.argmax(scores)
        return [best_idx], rescore_flag
