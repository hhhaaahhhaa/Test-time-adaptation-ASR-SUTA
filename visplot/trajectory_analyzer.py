import yaml
import pickle
from tqdm import tqdm

from .step_selector import Selector


class TrajectoryAnalyzer(object):
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
            vals.append(best_steps[0])
        return vals
