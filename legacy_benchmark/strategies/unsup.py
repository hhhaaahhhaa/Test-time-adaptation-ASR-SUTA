import os
from torch.utils.data import Dataset, DataLoader, Subset
from tqdm import tqdm

from systems.suta import SUTASystem
from utils.tool import wer
from .basic import BaseStrategy

from dlhlp_lib.utils.data_structure import Queue


class UnsupStrategy(BaseStrategy):
    def __init__(self, config) -> None:
        self.config = config
        self.system_config = config["config"]
        self.system = SUTASystem(self.system_config)

        self.queue = Queue(max_size=16)
        self.bs = 4
        self.gradient_accumulation_step = 4
    
    def unsup_train(self, ds: Dataset):
        dataloader = DataLoader(
            ds,
            batch_size=1,
            num_workers=4,
            shuffle=True,
            collate_fn=lambda x: x
        )
        self.system.eval()
        for _ in range(4):
            for idx, sample in tqdm(enumerate(dataloader)):
                sample = sample[0]
                if len(sample["wav"]) <= 20 * 16000:
                    self.queue.update(sample)

                if (idx + 1) % (self.bs * self.gradient_accumulation_step) == 0:
                    record = {}
                    self.system.suta_adapt_auto(
                        wavs=[s["wav"] for s in self.queue.data],
                        record=record,
                        batch_size=2
                    )
                    if record.get("collapse", False):
                        print("oh no")
                    
        self.system.snapshot("unsup_start")
        os.makedirs(self.config["output_dir"]["ckpt_dir"], exist_ok=True)
        self.system.save(f'{self.config["output_dir"]["ckpt_dir"]}/last.ckpt')

    def run(self, ds: Dataset):
        self.unsup_train(ds)
        self.system.adapt_count = 0
        self.system.load_snapshot("unsup_start")
        self.system.eval()

        n_words = []
        errs, losses = [], []
        transcriptions = []
        for sample in tqdm(ds):
            n_words.append(len(sample["text"].split(" ")))
            trans = self.system.inference([sample["wav"]])
            err = wer(sample["text"], trans[0])
            errs.append(err)
            transcriptions.append((sample["text"], trans[0]))

            # loss
            loss = self.system.calc_suta_loss([sample["wav"]])
            ctc_loss = self.system.calc_ctc_loss([sample["wav"]], [sample["text"]])
            loss["ctc_loss"] = ctc_loss["ctc_loss"]
            losses.append(loss)
        
        return {
            "wers": errs,
            "n_words": n_words,
            "transcriptions": transcriptions,
            "losses": losses,
        }


class SupStrategy(BaseStrategy):
    def __init__(self, config) -> None:
        self.config = config
        self.system_config = config["config"]
        self.system = SUTASystem(self.system_config)

        self.queue = Queue(max_size=16)
        self.bs = 4
        self.gradient_accumulation_step = 4
    
    def sup_train(self, ds: Dataset):
        dataloader = DataLoader(
            ds,
            batch_size=1,
            num_workers=4,
            shuffle=True,
            collate_fn=lambda x: x
        )
        self.system.train()
        for _ in range(4):
            for idx, sample in tqdm(enumerate(dataloader)):
                sample = sample[0]
                if len(sample["wav"]) <= 20 * 16000:
                    self.queue.update(sample)
                
                if (idx + 1) % (self.bs * self.gradient_accumulation_step) == 0:
                    record = {}
                    self.system.ctc_adapt_auto(
                        wavs=[s["wav"] for s in self.queue.data],
                        texts=[s["text"] for s in self.queue.data], 
                        record=record,
                        batch_size=2
                    )
                    if record.get("collapse", False):
                        print("oh no")
                    
        self.system.snapshot("sup_start")
        os.makedirs(self.config["output_dir"]["ckpt_dir"], exist_ok=True)
        self.system.save(f'{self.config["output_dir"]["ckpt_dir"]}/last.ckpt')

    def run(self, ds: Dataset):
        self.sup_train(ds)
        self.system.adapt_count = 0
        self.system.load_snapshot("sup_start")
        self.system.eval()

        n_words = []
        errs, losses = [], []
        transcriptions = []
        for sample in tqdm(ds):
            n_words.append(len(sample["text"].split(" ")))
            trans = self.system.inference([sample["wav"]])
            err = wer(sample["text"], trans[0])
            errs.append(err)
            transcriptions.append((sample["text"], trans[0]))

            # loss
            loss = self.system.calc_suta_loss([sample["wav"]])
            ctc_loss = self.system.calc_ctc_loss([sample["wav"]], [sample["text"]])
            loss["ctc_loss"] = ctc_loss["ctc_loss"]
            losses.append(loss)
        
        return {
            "wers": errs,
            "n_words": n_words,
            "transcriptions": transcriptions,
            "losses": losses,
        }


class OverfitStrategy(BaseStrategy):
    def __init__(self, config) -> None:
        self.config = config
        self.system_config = config["config"]
        self.system = SUTASystem(self.system_config)

        self.queue = Queue(max_size=16)
        self.bs = 4
        self.gradient_accumulation_step = 4
    
    def sup_train(self, ds: Dataset):
        dataloader = DataLoader(
            ds,
            batch_size=1,
            num_workers=4,
            shuffle=True,
            collate_fn=lambda x: x
        )
        self.system.train()
        n_iter = 25000 // len(ds)  # Default use 25000 step
        for _ in tqdm(range(n_iter)):
            for idx, sample in tqdm(enumerate(dataloader)):
                sample = sample[0]
                if len(sample["wav"]) <= 20 * 16000:
                    self.queue.update(sample)
                
                if (idx + 1) % (self.bs * self.gradient_accumulation_step) == 0:
                    record = {}
                    self.system.ctc_adapt_auto(
                        wavs=[s["wav"] for s in self.queue.data],
                        texts=[s["text"] for s in self.queue.data], 
                        record=record,
                        batch_size=2
                    )
                    if record.get("collapse", False):
                        print("oh no")
                    
        self.system.snapshot("sup_start")
        os.makedirs(self.config["output_dir"]["ckpt_dir"], exist_ok=True)
        self.system.save(f'{self.config["output_dir"]["ckpt_dir"]}/last.ckpt')

    def run(self, ds: Dataset):
        self.sup_train(ds)
        self.system.adapt_count = 0
        self.system.load_snapshot("sup_start")
        self.system.eval()

        n_words = []
        errs, losses = [], []
        transcriptions = []
        for sample in tqdm(ds):
            n_words.append(len(sample["text"].split(" ")))
            trans = self.system.inference([sample["wav"]])
            err = wer(sample["text"], trans[0])
            errs.append(err)
            transcriptions.append((sample["text"], trans[0]))

            # loss
            loss = self.system.calc_suta_loss([sample["wav"]])
            ctc_loss = self.system.calc_ctc_loss([sample["wav"]], [sample["text"]])
            loss["ctc_loss"] = ctc_loss["ctc_loss"]
            losses.append(loss)
        
        return {
            "wers": errs,
            "n_words": n_words,
            "transcriptions": transcriptions,
            "losses": losses,
        }


class UnsupFilterStrategy(BaseStrategy):
    """ Try to improve unsup by carefully selecting GOOD samples """
    def __init__(self, config) -> None:
        self.config = config
        self.system_config = config["config"]
        self.system = SUTASystem(self.system_config)

        self.queue = Queue(max_size=16)
        self.bs = 4
        self.gradient_accumulation_step = 4
        self.threshold = 0.5
    
    def unsup_train(self, ds: Dataset):
        dataloader = DataLoader(
            ds,
            batch_size=1,
            num_workers=4,
            shuffle=True,
            collate_fn=lambda x: x
        )
        self.system.eval()
        for _ in range(4):
            for idx, sample in tqdm(enumerate(dataloader)):
                sample = sample[0]
                if len(sample["wav"]) <= 20 * 16000:
                    self.queue.update(sample)

                if (idx + 1) % (self.bs * self.gradient_accumulation_step) == 0:
                    record = {}
                    self.system.suta_adapt_auto(
                        wavs=[s["wav"] for s in self.queue.data],
                        record=record,
                        batch_size=self.bs
                    )
                    if record.get("collapse", False):
                        print("oh no")
                    
        self.system.snapshot("unsup_start")
        os.makedirs(self.config["output_dir"]["ckpt_dir"], exist_ok=True)
        self.system.save(f'{self.config["output_dir"]["ckpt_dir"]}/last.ckpt')

    def clean_ds(self, ds: Dataset) -> Dataset:
        subset_idxs = []
        self.system.eval()
        for idx, sample in tqdm(enumerate(ds)):
            if len(sample["wav"]) > 20 * 16000:
                continue
            trans = self.system.inference([sample["wav"]])
            err = wer(sample["text"], trans[0])
            if err <= self.threshold:
                subset_idxs.append(idx)
        print("Filtered: ", len(subset_idxs))
        return Subset(ds, subset_idxs)
    
    def clean_ds2(self, ds: Dataset) -> Dataset:
        import pickle
        src = "results/benchmark/none/default"
        tgt = "results/benchmark/suta/default-0"
        task_name = "l2arctic_random"
        with open (f"{src}/{task_name}/result/results.pkl", "rb") as f:
            seq1 = pickle.load(f)
        wer1, n_word1 = seq1["wers"], seq1["n_words"]
        with open (f"{tgt}/{task_name}/result/results.pkl", "rb") as f:
            seq2 = pickle.load(f)
        wer2, n_word2 = seq2["wers"], seq2["n_words"]

        subset_idxs = []
        for i, (x, y) in enumerate(zip(wer1, wer2)):
            if x - 0.1 > y:
                subset_idxs.append(i)
        print("Filtered: ", len(subset_idxs))
        return Subset(ds, subset_idxs)

    def run(self, ds: Dataset):
        # clean_ds = self.clean_ds(ds)
        clean_ds = self.clean_ds2(ds)
        self.unsup_train(clean_ds)
        self.system.adapt_count = 0
        self.system.load_snapshot("unsup_start")
        self.system.eval()

        n_words = []
        errs = []
        transcriptions = []
        for sample in tqdm(ds):
            n_words.append(len(sample["text"].split(" ")))
            trans = self.system.inference([sample["wav"]])
            err = wer(sample["text"], trans[0])
            errs.append(err)
            transcriptions.append((sample["text"], trans[0]))
        
        return {
            "wers": errs,
            "n_words": n_words,
            "transcriptions": transcriptions,
        }
