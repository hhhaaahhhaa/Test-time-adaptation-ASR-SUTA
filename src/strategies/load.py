import os
import typing

from ..utils.load import get_class_in_module
from .base import IStrategy


SRC_DIR = "src/strategies"

BASIC = {
    "none": (f"{SRC_DIR}/basic.py", "NoStrategy"),
    "suta": (f"{SRC_DIR}/basic.py", "SUTAStrategy"),
    "rescore": (f"{SRC_DIR}/basic.py", "RescoreStrategy"),
    "suta-rescore": (f"{SRC_DIR}/basic.py", "SUTARescoreStrategy"),
}

DSUTA = {
    "dsuta": (f"{SRC_DIR}/dsuta.py", "DSUTAStrategy"),
    "dsuta-reset": (f"{SRC_DIR}/dsuta_reset.py", "DSUTAResetStrategy"),
    "dsuta-rescore": (f"{SRC_DIR}/dsuta.py", "DSUTARescoreStrategy"),
}

KL = {
    "suta-kl": (f"{SRC_DIR}/suta_kl.py", "SUTAKLStrategy"),
    "suta-lm": (f"{SRC_DIR}/suta_kl.py", "SUTALMStrategy"),
}

EC = {
    # "LLM": (f"{SRC_DIR}/error_correction.py", "LLMStrategy"),
    # "aLLM": (f"{SRC_DIR}/error_correction.py", "AsyncLLMStrategy"),
}

MIX = {
    # "suta-LLM": (f"{SRC_DIR}/mix/suta.py", "SUTALLMStrategy"),
}

OTHER = {
    "csuta": (f"{SRC_DIR}/other.py", "CSUTAStrategy"),
    "sdpl": (f"{SRC_DIR}/other.py", "SDPLStrategy"),
    "awmc": (f"{SRC_DIR}/awmc.py", "AWMCStrategy"),
    "litta": (f"{SRC_DIR}/litta.py", "LITTAStrategy"),
}

EXP = {
    "overfit": (f"{SRC_DIR}/upperbound.py", "OverfitStrategy"),
    "v0": (f"{SRC_DIR}/mix/select.py", "V0Strategy"),
    "v0-ppl": (f"{SRC_DIR}/mix/select.py", "V0PPLStrategy"),
    "v0a": (f"{SRC_DIR}/mix/select.py", "V0AStrategy"),
    "v1": (f"{SRC_DIR}/mix/select.py", "V1Strategy"),
    "v2": (f"{SRC_DIR}/mix/select.py", "V2Strategy"),
    
    "ssuta-rescore": (f"{SRC_DIR}/mix/ssuta.py", "SSUTARescoreStrategy"),
    "ssuta-LLM": (f"{SRC_DIR}/mix/ssuta.py", "SSUTALLMStrategy"),
    "psuta-rescore": (f"{SRC_DIR}/mix/psuta.py", "PSUTARescoreStrategy"),

    "suta-traj": (f"{SRC_DIR}/trajectory.py", "SUTATrajectory"),
}

EMATCH = {
    "poem-upper": (f"{SRC_DIR}/ematch/upperbound.py", "POEMUpperStrategy"),
    "poem-upper-rescore": (f"{SRC_DIR}/ematch/upperbound.py", "POEMUpperRescoreStrategy"),
    "dpoem-upper": (f"{SRC_DIR}/ematch/poem.py", "DPOEMUpperStrategy"),

    "ot": (f"{SRC_DIR}/ematch/offline.py", "OptimalTransportStrategy"),
    "ot-rescore": (f"{SRC_DIR}/ematch/offline.py", "OptimalTransportRescoreStrategy"),
    "dot": (f"{SRC_DIR}/ematch/offline.py", "DOptimalTransportStrategy"),
    "dot-rescore": (f"{SRC_DIR}/ematch/offline.py", "DOptimalTransportRescoreStrategy"),
}

STRATEGY_MAPPING = {
    **BASIC,
    **DSUTA,
    **KL,
    **EC,
    **MIX,
    **OTHER,
    **EXP,
    **EMATCH,
}


def get_strategy_cls(name) -> typing.Type[IStrategy]:
    module_path, class_name = STRATEGY_MAPPING[name]
    return get_class_in_module(class_name, module_path)
