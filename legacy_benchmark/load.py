from torch.utils.data import Dataset
from typing import Type

from .tasks import (
    chime, esd, l2arctic, synth, librispeech, librispeech_c, commonvoice, ted, switchboard, l2arctic_speaker
)
from .strategies import (
    basic, dynamic, other, batch, merge, awmc, cotta, dual, unsup, general
)


STRATEGY = {
    "none": basic.NoStrategy,
    "suta": basic.SUTAStrategy,
    "csuta": basic.CSUTAStrategy,
    "sdpl": basic.SDPLStrategy,

    "dsuta": dynamic.DSUTAStrategy,
    "dcsuta": dynamic.DCSUTAStrategy,
    "trans": dynamic.TranscriptionStrategy,

    "csuta*": other.CSUTAResetStrategy,
    "multiexpert": other.MultiExpertStrategy,
    "multiexpert_trans": other.MultiExpertTransStrategy,
    "advanced": other.AdvancedStrategy,

    "suta-batch": batch.SUTAStrategy,
    "csuta-batch": batch.CSUTAStrategy,

    "awmc": awmc.AWMCStrategy,
    "cotta": cotta.CoTTAStrategy,

    "dual": dual.DualStrategy,
    # "dual-pl": dual.DualPLStrategy,
    # "ema-start": merge.EMAStartStrategy,
    "unsup": unsup.UnsupStrategy,
    "sup": unsup.SupStrategy,
    "overfit": unsup.OverfitStrategy,
    
    "unsup-filter": unsup.UnsupFilterStrategy,
    "experimental": other.ExpStrategy,

    "ddr": general.DualDomainResetStrategy
}


TASK = {
    "synth_random": synth.RandomSequence,
    "synth_content": synth.ContentSequence,
    "synth_speaker": synth.SpeakerSequence,
    "synth_noise": synth.NoiseSequence,

    "esd_random": esd.RandomSequence,
    "esd_speaker": esd.SpeakerSequence,
    "esd_emotion": esd.EmotionSequence,
    "esd_content": esd.ContentSequence,

    "l2arctic_random": l2arctic.RandomSequence,
    "l2arctic_accent": l2arctic.AccentSequence,
    "l2arctic_accent_hard": l2arctic.AccentHardSequence,
    "l2arctic_content": l2arctic.ContentSequence,
    "l2arctic_speaker": l2arctic.SpeakerSequence,

    "chime_random": chime.RandomSequence,
    "chime_real": chime.UniqueRealSequence,
    "chime_simu": chime.UniqueSimuSequence,
    
    "librispeech_random": librispeech.RandomSequence,
    "librispeech_random1": librispeech.RandomSequence1,
    "librispeech_random2": librispeech.RandomSequence2,

    "commonvoice_random": commonvoice.RandomSequence,
    "commonvoice_full_random": commonvoice.FullRandomSequence,
    "commonvoice_good": commonvoice.GoodSequence,
    "ted_random": ted.RandomSequence,

    "single": l2arctic.SingleAccentSequence,
    "swbd": switchboard.RandomSequence,

    "long1": librispeech_c.LongSequence1,
    "long2": librispeech_c.LongSequence2,
    "long3": librispeech_c.LongSequence3,
    "long4": librispeech_c.LongSequence4,
    "long5": librispeech_c.LongSequence5,
    "long6": librispeech_c.LongSequence6,
    "iid1": librispeech_c.LongIIDSequence1,
    "iid2": librispeech_c.LongIIDSequence2,
    "md_easy1": librispeech_c.MDEASY1,
    "md_easy2": librispeech_c.MDEASY2,
    "md_easy3": librispeech_c.MDEASY3,
    "md_long": librispeech_c.MDLong,

    "speaker1": l2arctic_speaker.RandomSequence1,
    "speaker2": l2arctic_speaker.RandomSequence2,
}


def get_strategy(name) -> Type[basic.BaseStrategy]:
    return STRATEGY[name]


def get_task(name) -> Dataset:
    if name.startswith("LS_"):
        types = name.split("_")
        noise_type = types[1]
        snr_level = 10
        if len(types) >= 3:
            snr_level = int(types[2])
        ds = librispeech_c.RandomSequence(noise_type, snr_level=snr_level)
        return ds

    return TASK[name]()
