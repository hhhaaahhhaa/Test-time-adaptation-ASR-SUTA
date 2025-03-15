from ..system.suta_new import SUTASystem
from .basic import SUTAStrategy


class CSUTAStrategy(SUTAStrategy):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
    
    def _init_start(self, sample) -> None:
        pass


class SDPLStrategy(SUTAStrategy):
    def __init__(self, config) -> None:
        self.config = config
        self.strategy_config = config["strategy_config"]
        self.system = SUTASystem(config["system_config"])
    
    def _adapt(self, sample):
        is_collapse = False
        for _ in range(self.strategy_config["steps"]):
            self.system.eval()
            pl = self.system.inference([sample["wav"]])[0]
            record = {}
            self.system.train()  # gradient update under train mode (SUTA is eval mode according to origin implementation)
            self.system.ctc_adapt(
                wavs=[sample["wav"]],
                texts=[pl],
                record=record,
            )
            if record.get("collapse", False):
                is_collapse = True
        if is_collapse:
            print("oh no")
