import os
import numpy as np
import torch
from torch.utils.data import Dataset
import yaml
from tqdm import tqdm
from collections import defaultdict
import random

from ...system.suta_new import SUTASystem, softmax_entropy
from ...utils.tool import wer
from ...utils.distribution import Distribution
from ..base import IStrategy


class ReservoirBuffer(object):
    def __init__(self, max_size: int):
        self.max_size = max_size
        self.data = []
        self.count = 0  # Number of items seen so far

    def add(self, xs: list[float]):
        if self.max_size == -1:  # infinite
            self.data.extend(xs)
            return
        for x in xs:
            self.count += 1
            if len(self.data) < self.max_size:
                self.data.append(x)
            else:
                # Replace an existing element with probability max_size / count
                j = random.randint(0, self.count - 1)
                if j < self.max_size:
                    self.data[j] = x

    def clear(self):
        self.data.clear()
        self.count = 0  # Reset counter
