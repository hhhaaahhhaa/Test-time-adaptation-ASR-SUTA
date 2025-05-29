import torch
from torch.utils.data import Dataset
import yaml
from tqdm import tqdm
from collections import defaultdict
from transformers import Wav2Vec2Processor, Wav2Vec2ProcessorWithLM

from ..system.suta_new import SUTASystem
from ..utils.tool import wer
from .base import IStrategy
from .dsuta import Buffer


class SUTATrajectory(IStrategy):
    """ Only for wav2vec2 case since we hardcode to load both w/ and w/o processors here. """
    def __init__(self, config) -> None:
        self.config = config
        self.strategy_config = config["strategy_config"]
        self.system = SUTASystem(config["system_config"])
        self.system.eval()

        self.processor_no_lm = Wav2Vec2Processor.from_pretrained(config["system_config"]["model_name"])
        self.processor_with_lm = Wav2Vec2ProcessorWithLM.from_pretrained("patrickvonplaten/wav2vec2-base-960h-4-gram")
    
    def _init_start(self, sample) -> None:
        self.system.load_snapshot("init")
    
    def _adapt(self, sample):
        self.system.eval()
        is_collapse = False
        
        trajectory = defaultdict(list)
        # trajectory
        logits = self.system.calc_logits([sample["wav"]])[0]
        trajectory["logits"].append(logits)
        for _ in range(self.strategy_config["steps"]):
            record = {}
            self.system.suta_adapt(
                wavs=[sample["wav"]],
                record=record,
            )
            if record.get("collapse", False):
                is_collapse = True

            # trajectory
            logits = self.system.calc_logits([sample["wav"]])[0]
            trajectory["logits"].append(logits)
        if is_collapse:
            print("oh no")
        self._log["trajectories"].append(trajectory)

    def _update(self, sample):
        pass
    
    def inference(self, sample) -> str:
        self.system.eval()
        self.system.processor = self.processor_no_lm
        trans = self.system.inference([sample["wav"]])[0]
        return trans
    
    def run(self, ds: Dataset):
        long_cnt = 0
        self._log = defaultdict(list)
        for sample in tqdm(ds):
            if len(sample["wav"]) > self.strategy_config["max_length"]:
                long_cnt += 1
                continue
            self._log["n_words"].append(len(sample["text"].split(" ")))

            self._init_start(sample)
            self._adapt(sample)

            trans = self.inference(sample)
            err = wer(sample["text"], trans)
            self._log["wers"].append(err)
            self._log["transcriptions"].append((sample["text"], trans))
            self._log["basenames"].append(sample["id"])

            self._update(sample)
            
        print("#Too long: ", long_cnt)
        
        return self._log
    
    def get_adapt_count(self):
        return self.system.adapt_count


class DSUTATrajectory(IStrategy):
    def __init__(self, config) -> None:
        self.config = config
        self.strategy_config = config["strategy_config"]
        self.system = SUTASystem(config["system_config"])
        self.system.eval()

        # Set slow system
        self.slow_system = SUTASystem(config["system_config"])
        self.slow_system.eval()
        self.slow_system.snapshot("start")
        self.timestep = 0
        self.update_freq = config["strategy_config"]["update_freq"]
        self.memory = Buffer(max_size=config["strategy_config"]["memory"])

        self.system.snapshot("start")

    def _init_start(self, sample):
        self.system.load_snapshot("start")

    def _adapt(self, sample):
        self.system.eval()
        is_collapse = False
        trajectory = defaultdict(list)

        # trajectory
        logits = self.system.calc_logits([sample["wav"]])[0]
        trajectory["logits"].append(logits)
        trans = self.inference(sample, trajectory=trajectory)
        err = wer(sample["text"], trans)
        trajectory["suta-rescore-wer"].append(err)

        for _ in range(self.strategy_config["steps"]):
            record = {}
            self.system.suta_adapt(
                wavs=[sample["wav"]],
                record=record,
            )
            if record.get("collapse", False):
                is_collapse = True

            # trajectory
            logits = self.system.calc_logits([sample["wav"]])[0]
            trajectory["logits"].append(logits)
            trans = self.inference(sample, trajectory=trajectory)
            err = wer(sample["text"], trans)
            trajectory["suta-rescore-wer"].append(err)
        if is_collapse:
            print("oh no")
        self._log["trajectories"].append(trajectory)
    
    def _update(self, sample):
        self.memory.update(sample)
        if (self.timestep + 1) % self.update_freq == 0:
            self.slow_system.load_snapshot("start")
            self.slow_system.eval()
            record = {}
            self.slow_system.suta_adapt(
                wavs=[s["wav"] for s in self.memory.data],
                batch_size=1,
                record=record,
            )
            if record.get("collapse", False):
                print("oh no")
            self.slow_system.snapshot("start")
            self.memory.clear()
        self.system.history["start"] = self.slow_system.history["start"]  # fetch start point from slow system
    
    def inference(self, sample, trajectory=None) -> str:  # trajectory is passed for recording
        self.system.eval()
        res = self.system.beam_inference([sample["wav"]], n_best=5, text_only=False)
        nbest_trans = list(res.text)
        if trajectory is not None:
            trajectory["nbest_trans"].append(nbest_trans)  # not exactly n results due to deduplication
            logit_score = list(res.logit_score)
            trajectory["logit_score"].append(logit_score)
            merged_score = list(res.lm_score)
            trajectory["merged_score"].append(merged_score)
        return nbest_trans[0]
    
    def run(self, ds: Dataset):
        long_cnt = 0
        self._log = defaultdict(list)
        for sample in tqdm(ds):
            if len(sample["wav"]) > self.strategy_config["max_length"]:
                long_cnt += 1
                continue
            self._log["n_words"].append(len(sample["text"].split(" ")))

            self._init_start(sample)
            self._adapt(sample)

            trans = self.inference(sample)
            err = wer(sample["text"], trans)
            self._log["wers"].append(err)
            self._log["transcriptions"].append((sample["text"], trans))
            self._log["basenames"].append(sample["id"])

            self._log["logits"].append(self.system.calc_logits([sample["wav"]])[0])

            self._update(sample)
            self.timestep += 1
            
        print("#Too long: ", long_cnt)
        
        return self._log
    
    def get_adapt_count(self):
        return self.system.adapt_count


class CMergeTrajectory(DSUTATrajectory):
    def _adapt(self, sample):
        self.system.eval()
        is_collapse = False
        trajectory = defaultdict(list)
        
        # trajectory
        logits = self.system.calc_logits([sample["wav"]])[0]
        trajectory["logits"].append(logits)
        trans = self.inference(sample, trajectory=trajectory)
        err = wer(sample["text"], trans)
        trajectory["suta-rescore-wer"].append(err)

        # record the best
        best_score, best_step = trajectory["merged_score"][-1][0], 0
        self.system.snapshot("best")

        for idx in range(self.strategy_config["steps"]):
            record = {}
            self.system.suta_adapt(
                wavs=[sample["wav"]],
                record=record,
            )
            if record.get("collapse", False):
                is_collapse = True

            # trajectory
            logits = self.system.calc_logits([sample["wav"]])[0]
            trajectory["logits"].append(logits)
            trans = self.inference(sample, trajectory=trajectory)
            err = wer(sample["text"], trans)
            trajectory["suta-rescore-wer"].append(err)

            score = trajectory["merged_score"][-1][0]
            if score < best_score:
                best_score, best_step = score, idx + 1
                self.system.snapshot("best")

        if is_collapse:
            print("oh no")
        self._log["trajectories"].append(trajectory)

    def _update(self, sample):
        # generate task vector and save to memory
        with torch.no_grad():
            tv = {}
            for k, param in self.slow_system.model.named_parameters():
                if param.requires_grad:
                    tv[k] = self.system.history["best"][0][k] - self.slow_system.history["start"][0][k]
            self.memory.update(tv)

        if (self.timestep + 1) % self.update_freq == 0:
            self.slow_system.load_snapshot("start")
            self.slow_system.eval()

            # manual optimization
            debug_delta = {}
            debug_orig = {}
            for name, param in self.slow_system.model.named_parameters():
                if param.requires_grad:
                    avg = 0
                    for tv in self.memory.data:
                        avg += tv[name]
                    delta = 0.1 * avg / len(self.memory.data)
                    assert delta.shape == param.data.shape  # sanity check
                    param.data += delta
                    # debug_delta[name] = delta
                    # debug_orig[name] = param.data.clone().detach()
                    # # break
                    # grad = -delta / self.slow_system.config["lr"]
                    # assert grad.shape == param.data.shape  # sanity check
                    # param.grad = grad.clone().detach()
            # self.slow_system.optimizer.step()
            # self.slow_system.model.zero_grad()

            # with torch.no_grad():
            #     for name, param in self.slow_system.model.named_parameters():
            #         if param.requires_grad:
            #             print("Before: ", name, param.data[:5])
            #             print("After: ", name, debug_orig[name][:5])
            #             print((param.data - debug_orig[name])[:5])
            #             print(debug_delta[name][:5])
            #             assert 1 == 2

            self.slow_system.snapshot("start")
            self.memory.clear()
            self.system.history["start"][0] = self.slow_system.history["start"][0]  # fetch start point from slow system


class CSUPTrajectory(DSUTATrajectory):
    def _adapt(self, sample):
        self.system.eval()
        is_collapse = False
        trajectory = defaultdict(list)
        
        # trajectory
        logits = self.system.calc_logits([sample["wav"]])[0]
        trajectory["logits"].append(logits)
        trans = self.inference(sample, trajectory=trajectory)
        err = wer(sample["text"], trans)
        trajectory["suta-rescore-wer"].append(err)

        for idx in range(self.strategy_config["steps"]):
            record = {}
            self.system.suta_adapt(
                wavs=[sample["wav"]],
                record=record,
            )
            if record.get("collapse", False):
                is_collapse = True

            # trajectory
            logits = self.system.calc_logits([sample["wav"]])[0]
            trajectory["logits"].append(logits)
            trans = self.inference(sample, trajectory=trajectory)
            err = wer(sample["text"], trans)
            trajectory["suta-rescore-wer"].append(err)

        if is_collapse:
            print("oh no")
        self._log["trajectories"].append(trajectory)

    def _update(self, sample):
        self.memory.update(sample)
        if (self.timestep + 1) % self.update_freq == 0:
            self.slow_system.load_snapshot("start")
            self.slow_system.eval()
            record = {}
            self.slow_system.ctc_adapt(
                wavs=[s["wav"] for s in self.memory.data],
                texts=[s["text"] for s in self.memory.data],
                batch_size=1,
                record=record,
            )
            if record.get("collapse", False):
                print("oh no")
            self.slow_system.snapshot("start")
            self.memory.clear()
        self.system.history["start"] = self.slow_system.history["start"]  # fetch start point from slow system
