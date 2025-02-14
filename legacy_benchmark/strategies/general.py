
from torch.utils.data import Dataset
import yaml
from tqdm import tqdm
import scipy.stats
import statistics

from systems.suta import SUTASystem
from utils.tool import wer

from dlhlp_lib.utils.data_structure import Queue

from .basic import BaseStrategy


class GaussianStatModel(object):
    def __init__(self, data: list[float]) -> None:
        self.mu = statistics.mean(data)
        self.std = statistics.stdev(data)
        # print(f"Gaussian model: mu={self.mu}, std={self.std}")
    
    def get_prob(self, x: float, reduction: int=1) -> float:  # reduce variance by multiple sampling
        return scipy.stats.norm(self.mu, self.std / (reduction ** 0.5)).cdf(x)
    
    def get_deviation(self, x: float, reduction: int=1) -> float:  # reduce variance by multiple sampling
        return (x - self.mu) / (self.std / (reduction ** 0.5))


class FixDomainExpert(object):
    def __init__(self, config) -> None:
        self.system_config = config["config"]
        self.system = SUTASystem(self.system_config)
        self.timestep = 0

        # create stat (gaussian model)
        self.stat_model = None
        self.K = config["strategy_config"]["K"]
        self.domain_stats = []
        
        self.system.snapshot("start")

    def reset(self) -> None:
        self.timestep = 0
        self.system.load_snapshot("init")
        self.system.history = {}
        self.system.snapshot("init")

        self.stat_model = None
        self.domain_stats = []
        self.system.snapshot("start")
    
    def collect_improvement_stats(self, data: list, snapshot_name: str) -> list[float]:
        init_losses, domain_losses = [], []
        self.system.load_snapshot("init")
        self.system.eval()
        for sample in data:
            loss = self.system.calc_suta_loss([sample["wav"]])
            init_losses.append(loss["total_loss"])
        
        self.system.load_snapshot(snapshot_name)
        self.system.eval()
        for sample in data:
            loss = self.system.calc_suta_loss([sample["wav"]])
            domain_losses.append(loss["total_loss"])
        return [x - y for x, y in zip(domain_losses, init_losses)]

    def build_stat_model(self, data: list):
        if self.timestep > self.K:
            return
        elif self.timestep > self.K // 2:
            # Collect "loss improvement" stat for domain detector
            self.domain_stats.extend(self.collect_improvement_stats(data, "domain"))
        elif self.timestep == self.K // 2:
            self.system.snapshot("domain")
        else:
            pass
        if self.timestep == self.K:
            self.stat_model = GaussianStatModel(self.domain_stats)  # stat is collected from "domain"
            self.domain_stats = []

    def is_distribution_shift(self, data) -> bool:
        if self.stat_model is None:
            return False
        
        # Collect current loss improvement stat and then judged by domain detector (Gaussian)
        data_stat = statistics.mean(self.collect_improvement_stats(data, "domain"))
        # p = self.stat_model.get_prob(data_stat, reduction=len(data))
        deviation = self.stat_model.get_deviation(data_stat, reduction=len(data))
        # print(f"Stat: {data_stat}, deviation={deviation}")
        
        # if p > 0.9999366575:  # +4std
        # if p > 0.9544997361: # +2std
        return deviation > 2

    def update(self, data):
        self.system.load_snapshot("start")
        self.system.eval()
        record = {}
        self.system.suta_adapt_auto(
            wavs=[s["wav"] for s in data],
            batch_size=1,
            record=record,
        )
        if record.get("collapse", False):
            print("oh no")
        self.system.snapshot("start")


# class DynamicDomainExpert(object):
#     def __init__(self, config) -> None:
#         self.system_config = config["config"]
#         self.system = SUTASystem(self.system_config)
#         self.timestep = 0

#         # create stat (gaussian model)
#         self.stat_model = None
#         self.domain_update_freq = 50
#         self.next_update_step = 2 * self.domain_update_freq
#         self.domain_stats = []
        
#         self.system.snapshot("start")
    
#     def reset(self) -> None:
#         self.timestep = 0
#         self.system.load_snapshot("init")
#         self.system.history = {}
#         self.system.snapshot("init")

#         self.stat_model = None
#         self.domain_update_freq = 50
#         self.next_update_step = 2 * self.domain_update_freq
#         self.domain_stats = []
#         self.system.snapshot("start")
    
#     def collect_improvement_stats(self, data, snapshot_name: str) -> list[float]:
#         init_losses, domain_losses = [], []
#         self.system.load_snapshot("init")
#         self.system.eval()
#         for sample in data:
#             loss = self.system.calc_suta_loss([sample["wav"]])
#             init_losses.append(loss["total_loss"])
        
#         self.system.load_snapshot(snapshot_name)
#         self.system.eval()
#         for sample in data:
#             loss = self.system.calc_suta_loss([sample["wav"]])
#             domain_losses.append(loss["total_loss"])
#         return [x - y for x, y in zip(domain_losses, init_losses)]

#     def build_stat_model(self, data):
#         if "domain" not in self.system.history:
#             if self.timestep >= self.domain_update_freq:  # first creation
#                 self.system.snapshot("domain")
#             else:
#                 return
        
#         if self.timestep >= self.next_update_step:
#             if "running_domain" in self.system.history:  # domain <- running_domain, running_domain <- current
#                 self.system.history["domain"] = self.system.history["running_domain"]  # update "domain"
#             self.system.snapshot("running_domain")  # snapshot current status to "running_domain"
#             self.stat_model = GaussianStatModel(self.domain_stats)  # stat is collected from "domain"
#             self.domain_stats = []
#             self.next_update_step += self.domain_update_freq

#         # Collect "loss improvement" stat for domain detector
#         compared_snapshot_name = "running_domain" if "running_domain" in self.system.history else "domain"
#         self.domain_stats.extend(self.collect_improvement_stats(data, compared_snapshot_name))

#     def is_distribution_shift(self, data) -> bool:
#         if self.stat_model is None:
#             return False
        
#         # Collect current loss improvement stat and then judged by domain detector (Gaussian)
#         data_stat = statistics.mean(self.collect_improvement_stats(data, "domain"))
#         # p = self.stat_model.get_prob(data_stat, reduction=len(data))
#         deviation = self.stat_model.get_deviation(data_stat, reduction=len(data))
#         # print(f"Stat: {data_stat}, deviation={deviation}")
        
#         # if p > 0.9999366575:  # +4std
#         # if p > 0.9544997361: # +2std
#         if deviation > 2:
#             return True
#         return False

#     def update(self, data):
#         self.system.load_snapshot("start")
#         self.system.eval()
#         record = {}
#         self.system.suta_adapt_auto(
#             wavs=[s["wav"] for s in data],
#             batch_size=1,
#             record=record,
#         )
#         if record.get("collapse", False):
#             print("oh no")
#         self.system.snapshot("start")
#         self.timestep += len(data)


class FixFreqDomainExpert(object):
    def __init__(self, config) -> None:
        self.system_config = config["config"]
        self.system = SUTASystem(self.system_config)
        self.timestep = 0

        self.domain_update_freq = 50
        
        self.system.snapshot("start")

    def reset(self) -> None:
        self.timestep = 0
        self.system.load_snapshot("init")
        self.system.history = {}
        self.system.snapshot("init")
        self.system.snapshot("start")
    
    def build_stat_model(self, data: list):
        pass

    def is_distribution_shift(self, data: list) -> bool:
        return self.timestep >= self.domain_update_freq
        
    def update(self, data):
        self.system.load_snapshot("start")
        self.system.eval()
        record = {}
        self.system.suta_adapt_auto(
            wavs=[s["wav"] for s in data],
            batch_size=1,
            record=record,
        )
        if record.get("collapse", False):
            print("oh no")
        self.system.snapshot("start")


class DualDomainResetStrategy(BaseStrategy):
    def __init__(self, config) -> None:
        self.config = config
        self.system_config = config["config"]
        self.system = SUTASystem(self.system_config)
        self.system.snapshot("start")

        # Set expert
        if config["strategy_config"]["reset"] == "fix":
            self.expert_cls = FixDomainExpert
        # elif config["strategy_config"]["reset"] == "dynamic":
        #     self.expert_cls = DynamicDomainExpert
        elif config["strategy_config"]["reset"] == "fix-freq":
            self.expert_cls = FixFreqDomainExpert
        else:
            raise NotImplementedError
        self.expert = self.expert_cls(config)
        self.fail_cnt, self.patience = 0, config["strategy_config"]["patience"]

        self.timestep = 0
        self.update_freq = config["strategy_config"]["update_freq"]
        self.memory = Queue(max_size=config["strategy_config"]["memory"])

    def init_start(self, sample) -> None:
        self.system.load_snapshot("start")
    
    def adapt(self, sample):
        self.system.eval()
        is_collapse = False
        for _ in range(self.system_config["steps"]):
            record = {}
            self.system.suta_adapt(
                wavs=[sample["wav"]],
                record=record,
            )
            if record.get("collapse", False):
                is_collapse = True
        if is_collapse:
            print("oh no")

    def update(self, sample):
        self.memory.update(sample)
        if self.timestep % self.update_freq == 0:
            if not isinstance(self.expert, FixFreqDomainExpert):  # reset before update
                self.check_and_reset()
            self.expert.update(self.memory.data)
            self.memory.clear()
        self.expert.build_stat_model([sample])
        if isinstance(self.expert, FixFreqDomainExpert):  # reset after update (for fix-frequency only)
            self.check_and_reset()
        
        self.system.history["start"] = self.expert.system.history["start"]  # fetch start point from slow system
    
    def check_and_reset(self):  # reset with patience
        if self.expert.is_distribution_shift(self.memory.data):
            self.fail_cnt += 1
        else:
            self.fail_cnt = 0

        # handle reset
        if self.fail_cnt == self.patience:
            print("========== reset ==========")
            self.expert.reset()  # reset expert
            self.reset_record.append(self.timestep)
            self.fail_cnt = 0
    
    def run(self, ds: Dataset):
        self.reset_record = []
        n_words = []
        errs, losses = [], []
        transcriptions = []
        for sample in tqdm(ds):
            n_words.append(len(sample["text"].split(" ")))

            self.init_start(sample)
            self.adapt(sample)

            self.system.eval()
            trans = self.system.inference([sample["wav"]])[0]
            err = wer(sample["text"], trans)
            errs.append(err)
            transcriptions.append((sample["text"], trans))
            
            # loss
            loss = self.system.calc_suta_loss([sample["wav"]])
            ctc_loss = self.system.calc_ctc_loss([sample["wav"]], [sample["text"]])
            loss["ctc_loss"] = ctc_loss["ctc_loss"]
            losses.append(loss)

            self.timestep += 1
            self.expert.timestep += 1  # synchronize time step
            self.update(sample)
        
        return {
            "wers": errs,
            "n_words": n_words,
            "transcriptions": transcriptions,
            "losses": losses,
            "reset_step": self.reset_record,
        }
    
    def get_adapt_count(self):
        return self.system.adapt_count
