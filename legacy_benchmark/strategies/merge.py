from torch.utils.data import Dataset
from tqdm import tqdm

from systems.suta import SUTASystem
from utils.tool import wer
from .basic import BaseStrategy


def get_task_vector(state_dict1, state_dict2):
    task_vector = {
        name: state_dict1[name] - state_dict2[name]
    for name in state_dict1}
    return task_vector


def ema(src, tgt, alpha: float):
    res = {}
    for name in src:
        res[name] = alpha * src[name] + (1 - alpha) * tgt[name]
    return res


class EMAStartStrategy(BaseStrategy):
    def __init__(self, config) -> None:
        self.config = config
        self.system_config = config["config"]
        self.system = SUTASystem(self.system_config)

        self.timestep = 0
        self.alpha = config["strategy_config"]["alpha"]
        self._init_task_vector()

    def _init_task_vector(self):
        model_state = self.system.model.state_dict()
        self.ema_task_vector = get_task_vector(model_state, model_state)  # zero vector

    def _ema_update(self):
        model_state = self.system.model.state_dict()
        origin_model_state = self.system.history["init"][0]
        task_vector = get_task_vector(model_state, origin_model_state)
        self.ema_task_vector = ema(src=self.ema_task_vector, tgt=task_vector, alpha=self.alpha)
    
    def _load_start_point(self, sample):
        if self.system.adapt_count == 0:
            return
        origin_model_state = self.system.history["init"][0]
        merged_model_state = {
            name: origin_model_state[name] + self.ema_task_vector[name]
        for name in origin_model_state}
        self.system.history["start"] = (merged_model_state, None, None)
        self.system.load_snapshot("start")

    def _adapt(self, sample):
        self.system.eval()
        is_collapse = False
        for _ in range(self.system_config["steps"]):
            record = {}
            self.system.suta_adapt(
                wavs=[sample["wav"]],
                record=record,
            )
            self._ema_update()
            if record.get("collapse", False):
                is_collapse = True
        if is_collapse:
            print("oh no")
    
    def _update(self, sample):
        pass  # ema-start updates during adapt

    def run(self, ds: Dataset):
        long_cnt = 0
        n_words = []
        errs, losses = [], []
        transcriptions = []
        for sample in tqdm(ds):
            if len(sample["wav"]) > 20 * 16000:
                long_cnt += 1
                continue
            n_words.append(len(sample["text"].split(" ")))

            # update
            self.timestep += 1
            self._load_start_point(sample)
            self._adapt(sample)
            self._update(sample)

            self.system.eval()
            trans = self.system.inference([sample["wav"]])
            err = wer(sample["text"], trans[0])
            errs.append(err)
            transcriptions.append((sample["text"], trans[0]))
        print(long_cnt)
        
        return {
            "wers": errs,
            "n_words": n_words,
            "transcriptions": transcriptions,
            "losses": losses,
        }
    
    def get_adapt_count(self):
        return self.system.adapt_count
