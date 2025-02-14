import os
import torch
from torch.utils.data import Dataset, DataLoader
import yaml
from tqdm import tqdm
import json

from systems.suta import SUTASystem
from utils.tool import wer
from .basic import BaseStrategy


class CoTTAStrategy(BaseStrategy):
    def __init__(self, config) -> None:
        self.config = config
        self.system_config = config["config"]
        self.system = SUTASystem(self.system_config)
        self.teacher = SUTASystem(self.system_config)

        # setup teacher
        self.teacher.eval()
        for param in self.teacher.model.parameters():
            param.detach_()

        self.ema_task_vector = None
        self.alpha = 0.999
        self.restore_ratio = self.system_config["restore_ratio"]

        # log
        self.transcriptions = []

    @torch.no_grad
    def _ema_update_and_stochastic_restore(self):
        origin_model_state = self.system.history["init"][0]
        task_vector = self._get_task_vector(teacher=False)  # use system's task vector to update teacher

        if self.ema_task_vector is None:
            self.ema_task_vector = {}
            for name in origin_model_state:
                self.ema_task_vector[name] = (1 - self.alpha) * task_vector[name]
        else:
            for name in origin_model_state:
                self.ema_task_vector[name] = self.alpha * self.ema_task_vector[name] + (1 - self.alpha) * task_vector[name]
        
        for name in origin_model_state:
            param_to_update = self.teacher.model.state_dict()[name]
            param_to_update.data = (origin_model_state[name] + self.ema_task_vector[name]).data

        # stochastic_restore
        for nm, m in self.system.model.named_modules():
            for npp, p in m.named_parameters():
                if npp in ['weight', 'bias'] and p.requires_grad:
                    mask = (torch.rand(p.shape) < self.restore_ratio).float().to(self.system.model.device)
                    p.data = origin_model_state[f"{nm}.{npp}"] * mask + p * (1.-mask)

    def _get_task_vector(self, teacher=False):
        if teacher:
            model_state = self.teacher.model.state_dict()
        else:
            model_state = self.system.model.state_dict()
        model_state = self.system.model.state_dict()
        origin_model_state = self.system.history["init"][0]
        task_vector = {
            name: model_state[name] - origin_model_state[name]
        for name in model_state}
        return task_vector
    
    def _update(self, sample):
        teacher_pl_target = self.teacher.inference([sample["wav"]])[0]
        self.transcriptions[-1]["teacher"] = teacher_pl_target
        self.system.train()
        is_collapse = False
        for _ in range(self.system_config["steps"]):
            record = {}
            self.system.ctc_adapt(
                wavs=[sample["wav"]],
                texts=[teacher_pl_target],
                record=record,
            )
            if record.get("collapse", False):
                is_collapse = True
            self._ema_update_and_stochastic_restore()
        if is_collapse:
            print("oh no")
    
    def run(self, ds: Dataset):
        n_words = []
        errs, losses = [], []
        transcriptions = []
        for idx, sample in tqdm(enumerate(ds), total=len(ds)):
            self.transcriptions.append({})
            n_words.append(len(sample["text"].split(" ")))

            # update
            self._update(sample)

            self.system.eval()
            trans = self.system.inference([sample["wav"]])[0]
            err = wer(sample["text"], trans)
            errs.append(err)
            transcriptions.append((sample["text"], trans))
            self.transcriptions[-1]["system"] = trans
            self.transcriptions[-1]["gt"] = sample["text"]
            # print(sample["text"])
            # print(trans[0])
            # input()
        
        # log
        os.makedirs(self.config["output_dir"]["log_dir"], exist_ok=True)
        with open(f"{self.config['output_dir']['log_dir']}/log.json", "w") as f:
            json.dump(self.transcriptions, f, indent=4)
        
        return {
            "wers": errs,
            "n_words": n_words,
            "losses": losses,
            "transcriptions": transcriptions,
        }
    
    def get_adapt_count(self):
        return self.system.adapt_count
