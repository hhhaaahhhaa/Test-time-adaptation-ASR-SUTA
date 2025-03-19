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
