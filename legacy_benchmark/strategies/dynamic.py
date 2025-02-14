from .basic import BaseStrategy, SUTAStrategy


class TranscriptionStrategy(SUTAStrategy):
    """ Early exit if transcription changed """
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def trans_adapt(self, sample):
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
    
    def _update(self, sample):
        self.system.load_snapshot("init")
        res = self.trans_adapt(sample)
        if not res:  # suta failed
            print("oh no")
            self.system.load_snapshot("init")


class DSUTAStrategy(SUTAStrategy):
    """ Dynamic SUTA using loss plateau """
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
    
    def dynamic_adapt(self, sample):
        plateau_cnt, max_step = 3, self.config["steps"]
        best_loss = 2e9
        while max_step > 0:
            res = self.system.adapt(
                [sample["wav"]],
                em_coef=self.config["em_coef"],
                reweight=self.config["reweight"],
                temp=self.config["temp"],
                not_blank=self.config["non_blank"],
                l2_coef=self.config["l2_coef"],
            )
            if not res:
                break

            # calc loss
            loss = self.system.calc_loss(
                [sample["wav"]],
                em_coef=self.config["em_coef"],
                reweight=self.config["reweight"],
                temp=self.config["temp"],
                not_blank=self.config["non_blank"]
            )
            if loss["total_loss"] >= best_loss:
                plateau_cnt += 1
            else:
                best_loss = loss["total_loss"]
                self.system.snapshot("temp")
                plateau_cnt = 0
            # done
            if plateau_cnt == 3:
                self.system.load_snapshot("temp")
                break
            max_step -= 1
        # print("Adapt step: ", self.config["steps"] - max_step)
        return res
    
    def _update(self, sample):
        self.system.load_snapshot("init")
        res = self.dynamic_adapt(sample)
        if not res:  # suta failed
            print("oh no")
            self.system.load_snapshot("init")


class DCSUTAStrategy(DSUTAStrategy):
    def __init__(self, config) -> None:
        super().__init__(config)
    
    def _update(self, sample):
        self.system.snapshot("checkpoint")
        res = self.dynamic_adapt(sample)
        if not res:  # suta failed
            print("oh no")
            self.system.load_snapshot("checkpoint")
