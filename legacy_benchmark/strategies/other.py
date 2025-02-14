from torch.utils.data import Dataset
from tqdm import tqdm

from dlhlp_lib.utils.data_structure import Queue

from .basic import BaseStrategy
from systems.suta import SUTASystem
from utils.tool import wer


class CSUTAResetStrategy(BaseStrategy):
    def __init__(self, config) -> None:
        self.config = config
        self.system = SUTASystem(config)
    
    def fix_adapt(self, sample):
        for _ in range(self.config["steps"]):
            res = self.system.adapt(
                [sample["wav"]],
                em_coef=self.config["em_coef"],
                reweight=self.config["reweight"],
                temp=self.config["temp"],
                not_blank=self.config["non_blank"],
                l2_coef=0
            )
            if not res:
                break
        return res
    
    def _update(self, sample, is_boundary: bool):
        if is_boundary:
            self.system.load_snapshot("init")
        self.system.snapshot("checkpoint")
        res = self.fix_adapt(sample)
        if not res:  # suta failed
            print("oh no")
            self.system.load_snapshot("checkpoint")
    
    def run(self, ds: Dataset):
        n_words = []
        errs, losses = [], []
        self.system.snapshot("init")
        for idx, sample in tqdm(enumerate(ds)):
            n_words.append(len(sample["text"].split(" ")))

            # update
            self._update(sample, is_boundary=(idx + 1 in ds.task_boundaries))

            trans = self.system.inference([sample["wav"]])
            err = wer(sample["text"], trans[0])
            errs.append(err)
            loss = self.system.calc_loss(
                [sample["wav"]],
                em_coef=self.config["em_coef"],
                reweight=self.config["reweight"],
                temp=self.config["temp"],
                not_blank=self.config["non_blank"]
            )
            losses.append(loss)
        
        return {
            "wers": errs,
            "n_words": n_words,
            "losses": losses,
        }
    
    def get_adapt_count(self):
        return self.system.adapt_count


class MultiExpertStrategy(BaseStrategy):
    def __init__(self, config) -> None:
        self.config = config
        self.system = SUTASystem(config)

        # expert queue
        self.expert_keys = []
        self.expert_cnt = 0
        self.max_size = 10
    
    def _add_expert(self):
        self.system.snapshot(str(self.expert_cnt))
        self.expert_keys.append(str(self.expert_cnt))
        self.expert_cnt += 1
        if len(self.expert_keys) > self.max_size:
            popped_key = self.expert_keys.pop(0)
            self.system.delete_snapshot(popped_key)

    def _select_best_key_from_ref(self, refs) -> str:
        best_loss = 2e9
        for key in refs:
            if refs[key]["total_loss"] < best_loss:
                best_loss = refs[key]["total_loss"]
                best_key = key
        return best_key

    def _update(self, sample, refs):
        best_key = self._select_best_key_from_ref(refs)
        if best_key == "init":  # use suta if "init" is the best
            self.system.load_snapshot("init")
            for _ in range(self.config["steps"]):
                res = self.system.adapt(
                    [sample["wav"]],
                    em_coef=self.config["em_coef"],
                    reweight=self.config["reweight"],
                    temp=self.config["temp"],
                    not_blank=self.config["non_blank"]
                )
                if not res:
                    break
            if res:  # suta successed, add new expert
                self._add_expert()
            else:
                self.system.load_snapshot("init")
        else:
            self.system.load_snapshot(best_key)

    def run(self, ds: Dataset):
        n_words = []
        errs = []
        self.system.snapshot("init")
        for sample in tqdm(ds):
            n_words.append(len(sample["text"].split(" ")))

            # Query experts
            refs = {}
            for key in self.expert_keys + ["init"]:
                self.system.load_snapshot(key)
                loss = self.system.calc_loss(
                    [sample["wav"]],
                    em_coef=self.config["em_coef"],
                    reweight=self.config["reweight"],
                    temp=self.config["temp"],
                    not_blank=self.config["non_blank"]
                )
                refs[key] = loss

            # update  
            self._update(sample, refs)
            
            # inference
            trans = self.system.inference([sample["wav"]])
            err = wer(sample["text"], trans[0])
            errs.append(err)
        
        return {
            "wers": errs,
            "n_words": n_words
        }
    
    def get_adapt_count(self):
        return self.system.adapt_count


class MultiExpertTransStrategy(BaseStrategy):
    def __init__(self, config) -> None:
        self.config = config
        self.system = SUTASystem(config)

        # expert queue
        self.expert_keys = []
        self.expert_cnt = 0
        self.max_size = 1
    
    def _add_expert(self):
        self.system.snapshot(str(self.expert_cnt))
        self.expert_keys.append(str(self.expert_cnt))
        self.expert_cnt += 1
        if len(self.expert_keys) > self.max_size:
            popped_key = self.expert_keys.pop(0)
            self.system.delete_snapshot(popped_key)

    def _select_best_key_from_ref(self, refs) -> str:
        best_loss = 2e9
        for key in refs:
            if refs[key]["total_loss"] < best_loss:
                best_loss = refs[key]["total_loss"]
                best_key = key
        return best_key
    
    def _update(self, sample, refs):
        best_key = self._select_best_key_from_ref(refs)
        if best_key == "init":  # use suta if "init" is the best
            self.system.load_snapshot("init")

            # transription strategy
            orig_trans = self.system.inference([sample["wav"]])[0]
            step_cnt = self.config["steps"]
            while step_cnt > 0:
                res = self.system.adapt(
                    [sample["wav"]],
                    em_coef=self.config["em_coef"],
                    reweight=self.config["reweight"],
                    temp=self.config["temp"],
                    not_blank=self.config["non_blank"]
                )
                if not res:
                    break
                trans = self.system.inference([sample["wav"]])[0]
                if trans != orig_trans:  # changed
                    break
                step_cnt -= 1
            if res:  # suta successed, add new expert
                self._add_expert()
            else:
                self.system.load_snapshot("init")
        else:
            self.system.load_snapshot(best_key)

    def run(self, ds: Dataset):
        n_words = []
        errs = []
        self.system.snapshot("init")
        for sample in tqdm(ds):
            n_words.append(len(sample["text"].split(" ")))

            # Query experts
            refs = {}
            for key in self.expert_keys + ["init"]:
                self.system.load_snapshot(key)
                loss = self.system.calc_loss(
                    [sample["wav"]],
                    em_coef=self.config["em_coef"],
                    reweight=self.config["reweight"],
                    temp=self.config["temp"],
                    not_blank=self.config["non_blank"]
                )
                refs[key] = loss

            # update  
            self._update(sample, refs)
            
            # inference
            trans = self.system.inference([sample["wav"]])
            err = wer(sample["text"], trans[0])
            errs.append(err)
        
        return {
            "wers": errs,
            "n_words": n_words
        }

    def get_adapt_count(self):
        return self.system.adapt_count


class AdvancedStrategy(BaseStrategy):
    """ Offline + Online """
    def __init__(self, config) -> None:
        self.config = config
        self.system = SUTASystem(config)
        self.reset_cnt = 0
    
    def _select_best_key_from_ref(self, refs) -> str:
        best_key = "init"
        best_loss = refs["init"]["total_loss"] - 0.1
        for key in refs:
            if refs[key]["total_loss"] < best_loss:
                best_loss = refs[key]["total_loss"]
                best_key = key
        return best_key
    
    def trans_adapt(self, sample):  # transription strategy
        orig_trans = self.system.inference([sample["wav"]])[0]
        step_cnt = self.config["steps"]
        while step_cnt > 0:
            res = self.system.adapt(
                [sample["wav"]],
                em_coef=self.config["em_coef"],
                reweight=self.config["reweight"],
                temp=self.config["temp"],
                not_blank=self.config["non_blank"],
                l2_coef=self.config["l2_coef"]
            )
            if not res:
                break
            trans = self.system.inference([sample["wav"]])[0]
            if trans != orig_trans:  # changed
                break
            step_cnt -= 1
        return res
    
    def fix_adapt(self, sample):
        for _ in range(self.config["steps"]):
            res = self.system.adapt(
                [sample["wav"]],
                em_coef=self.config["em_coef"],
                reweight=self.config["reweight"],
                temp=self.config["temp"],
                not_blank=self.config["non_blank"],
                l2_coef=self.config["l2_coef"]
            )
            if not res:
                break
        return res
    
    def _update(self, sample, refs):
        best_key = self._select_best_key_from_ref(refs)
        print("compare: ", refs["init"]["total_loss"], refs["online"]["total_loss"])

        self.system.load_snapshot(best_key)
        # res = self.trans_adapt(sample)
        res = self.fix_adapt(sample)
        if not res:  # suta failed
            print("oh no")
            self.system.load_snapshot(best_key)
            return
        if best_key == "init":
            # print("replace")
            self.reset_cnt += 1
            self.system.snapshot("online")
        else:
            self.system.snapshot("online")

    def run(self, ds: Dataset):
        n_words = []
        errs, losses = [], []
        self.system.snapshot("init")
        self.system.snapshot("online")
        for sample in tqdm(ds):
            n_words.append(len(sample["text"].split(" ")))

            # self.system.load_snapshot("init")
            # orig_trans = self.system.inference([sample["wav"]])

            # Query experts
            refs = {}
            for key in ["init", "online"]:
                self.system.load_snapshot(key)
                loss = self.system.calc_loss(
                    [sample["wav"]],
                    em_coef=self.config["em_coef"],
                    reweight=self.config["reweight"],
                    temp=self.config["temp"],
                    not_blank=self.config["non_blank"]
                )
                refs[key] = loss

            # update  
            self._update(sample, refs)
            
            # inference
            trans = self.system.inference([sample["wav"]])
            err = wer(sample["text"], trans[0])
            # print(sample["text"])
            # print(orig_trans[0])
            # print(trans[0])
            errs.append(err)

            loss = self.system.calc_loss(
                [sample["wav"]],
                em_coef=self.config["em_coef"],
                reweight=self.config["reweight"],
                temp=self.config["temp"],
                not_blank=self.config["non_blank"]
            )
            losses.append(loss)
        print("Reset: ", self.reset_cnt)
        
        return {
            "wers": errs,
            "n_words": n_words,
            "losses": losses,
        }

    def get_adapt_count(self):
        return self.system.adapt_count


class ExpStrategy(BaseStrategy):
    """ experimental """
    def __init__(self, config) -> None:
        self.config = config
        self.system_config = config["config"]
        self.system = SUTASystem(self.system_config)
        
        self.slow_system = SUTASystem(self.system_config)
        self.slow_system.snapshot("start")
        self.system.snapshot("start")
        self.timestep = 0
        self.update_freq = config["strategy_config"]["update_freq"]
        self.memory = Queue(max_size=config["strategy_config"]["memory"])
        self.reset = config["strategy_config"].get("reset", False)

    def _load_start_point(self, sample):
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
            if record.get("collapse", False):
                is_collapse = True
        if is_collapse:
            print("oh no")
        # if len(sample["wav"]) <= 20 * 16000:
        self.memory.update(sample)
    
    def _update(self, sample):
        if self.timestep % self.update_freq != 0:
            return
        self.slow_system.load_snapshot("start")
        self.slow_system.eval()
        is_collapse = False
        record = {}
        self.slow_system.suta_adapt_auto(
            wavs=[s["wav"] for s in self.memory.data],
            batch_size=1,
            record=record,
        )
        if record.get("collapse", False):
            is_collapse = True
        if is_collapse:
            print("oh no")
        self.slow_system.snapshot("start")
        self.system.history["start"] = self.slow_system.history["start"]  # fetch start point from slow system

    def reset_strategy(self):
        self.memory.clear()
        self.slow_system.load_snapshot("init")
        self.system.load_snapshot("init")
        self.slow_system.snapshot("start")
        self.system.snapshot("start")
    
    def _run_one_domain(self, ds: Dataset):
        long_cnt = 0
        for sample in tqdm(ds):
            if len(sample["wav"]) > 20 * 16000:
                long_cnt += 1
                continue
            # update
            if self.timestep == 100:
                break  # Stop after 1st domain
            self._load_start_point(sample)
            self._adapt(sample)
            self._update(sample)

        self.system.snapshot("run_one_domain_start")
        print(long_cnt)

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

            self._load_start_point(sample)
            self._adapt(sample)

            self.system.eval()
            trans = self.system.inference([sample["wav"]])
            err = wer(sample["text"], trans[0])
            errs.append(err)
            transcriptions.append((sample["text"], trans[0]))

            # loss
            loss = self.system.calc_suta_loss([sample["wav"]])
            ctc_loss = self.system.calc_ctc_loss([sample["wav"]], [sample["text"]])
            loss["ctc_loss"] = ctc_loss["ctc_loss"]
            losses.append(loss)
            
            self.timestep += 1
            if self.timestep <= 100:  # only update when 1st domain
                self._update(sample)

        print(long_cnt)
        
        return {
            "wers": errs,
            "n_words": n_words,
            "transcriptions": transcriptions,
            "losses": losses,
        }
    
    # def run(self, ds: Dataset):
    #     self._run_one_domain(ds)
    #     self.system.adapt_count = 0
    #     self.system.load_snapshot("run_one_domain_start")
    #     self.system.eval()

    #     n_words = []
    #     errs, losses = [], []
    #     transcriptions = []
    #     for sample in tqdm(ds):
    #         n_words.append(len(sample["text"].split(" ")))
    #         trans = self.system.inference([sample["wav"]])
    #         err = wer(sample["text"], trans[0])
    #         errs.append(err)
    #         transcriptions.append((sample["text"], trans[0]))

    #         # loss
    #         loss = self.system.calc_suta_loss([sample["wav"]])
    #         ctc_loss = self.system.calc_ctc_loss([sample["wav"]], [sample["text"]])
    #         loss["ctc_loss"] = ctc_loss["ctc_loss"]
    #         losses.append(loss)
        
    #     return {
    #         "wers": errs,
    #         "n_words": n_words,
    #         "transcriptions": transcriptions,
    #         "losses": losses,
    #     }
        
    def get_adapt_count(self):
        return self.system.adapt_count + self.slow_system.adapt_count
