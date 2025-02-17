import os
from pathlib import Path
import numpy as np
import torch
import json
import librosa
from datasets import load_dataset
from tqdm import tqdm
import re
from builtins import str as unicode
from scipy.io import wavfile
import pickle

from . import Define


class SynthCorpus(object):
    def __init__(self) -> None:
        self.root = Define.SYNTH 
        with open(f"{self.root}/metadata.json", 'r') as f:
            self.metadata = json.load(f)

        # load noise
        self.musan_root = Define.MUSAN 
        background_noise_filename = [
            "noise-free-sound-0232",
            "noise-free-sound-0840",
            "noise-free-sound-0841",
            "noise-free-sound-0842",
            "noise-free-sound-0671",
            "noise-free-sound-0234",
            "noise-free-sound-0248",
            "noise-free-sound-0030",
            "noise-free-sound-0031",
            "noise-free-sound-0050",
        ]
        self.musan_noise_paths = [
            f"{self.musan_root}/noise/free-sound/{x}.wav" for x in background_noise_filename
        ]

    def __len__(self):
        return 1000
    
    def get_from_idxs(self, content_idx, speaker_idx, noise_idx) -> np.ndarray:
        idx = content_idx * 100 + speaker_idx * 10 + noise_idx
        return self.get(idx)
    
    def idx_transform(self, idx: int):
        assert idx < 1000
        res = [0, 0, 0]
        res[2] = idx % 10
        idx = idx // 10
        res[1] = idx % 10
        idx = idx // 10
        res[0] = idx
        return res
    
    def get(self, idx) -> np.ndarray:
        [content_idx, speaker_idx, noise_idx] = self.idx_transform(idx)
        basename = f"{content_idx * 10 + speaker_idx:03d}"
        wav_path = f"{self.root}/wavs/{basename}.wav"
        wav, _ = librosa.load(wav_path, sr=16000)
        text = self.metadata[basename]["text"]

        # add musan noise
        noise, _  = librosa.load(self.musan_noise_paths[noise_idx], sr=16000)
        if len(noise) > len(wav):
            noise = noise[:len(wav)]
        wav[:len(noise)] +=  noise * 0.2
        # wavfile.write("check.wav", 16000, (wav * 32767).astype(np.int16))
        # input()

        return {
            "id": wav_path,
            "wav": wav,
            "text": text
        }


class LibriSpeechCorpusOld(object):

    cache_dir = "_cache/LibriSpeech_old"

    def __init__(self, extra_noise=0.0) -> None:
        self.extra_noise = extra_noise
        if not os.path.exists(self.cache_dir):
            self.parse()
        with open(f"{self.cache_dir}/data_info.json", "r", encoding="utf-8") as f:
            basenames = json.load(f)
        self.wav_paths = []
        self.noise_paths = []
        self.texts = []
        for basename in basenames:
            with open(f"{self.cache_dir}/text/{basename}.txt", "r", encoding="utf-8") as f:
                text = f.read()
                self.texts.append(text.strip())
            self.wav_paths.append(f"{self.cache_dir}/wav/{basename}.wav")
            self.noise_paths.append(f"{self.cache_dir}/noise/{basename}.npy")

    def parse(self):
        basenames = []
        src_dataset = load_dataset(
            "librispeech_asr",
            split="test.other",
            streaming=True,
            trust_remote_code=True
        )
        os.makedirs(f"{self.cache_dir}/wav", exist_ok=True)
        os.makedirs(f"{self.cache_dir}/noise", exist_ok=True)
        os.makedirs(f"{self.cache_dir}/text", exist_ok=True)
        for idx, instance in tqdm(enumerate(src_dataset)):
            wav = librosa.resample(
                instance["audio"]["array"],
                orig_sr=src_dataset.features["audio"].sampling_rate,
                target_sr=16000
            )
            wavfile.write(f"{self.cache_dir}/wav/{idx:07d}.wav", 16000, (wav * 32767).astype(np.int16))
            noise = np.random.randn(*wav.shape)
            with open(f"{self.cache_dir}/noise/{idx:07d}.npy", "wb") as f:
                np.save(f, noise)
            with open(f"{self.cache_dir}/text/{idx:07d}.txt", "w", encoding="utf-8") as f:
                f.write(instance["text"])
            basenames.append(f"{idx:07d}")
        with open(f"{self.cache_dir}/data_info.json", "w", encoding="utf-8") as f:
            json.dump(basenames, f, indent=4)

    def __len__(self):
        return len(self.wav_paths)
    
    def get(self, idx) -> np.ndarray:
        wav, _ = librosa.load(self.wav_paths[idx], sr=16000)
        text = self.texts[idx]

        if self.extra_noise > 0:
            with open(self.noise_paths[idx], "rb") as f:
                noise = np.load(f)
            wav = wav + self.extra_noise * noise

        return {
            "id": self.wav_paths[idx],
            "wav": wav,
            "text": text
        }


class LibriSpeechCorpus(object):
    def __init__(self) -> None:
        self.root = "_cache/LibriSpeech"
        with open(f"{self.root}/data_info.json", "r", encoding="utf-8") as f:
            self.data_info = json.load(f)
        
        # Filter out long wavs > 20s
        self.filtered_idxs = []
        for idx, query in enumerate(self.data_info):
            if query["length"] <= 20 * 16000:
                self.filtered_idxs.append(idx)

    def __len__(self):
        return len(self.filtered_idxs)
    
    def get(self, idx):
        query = self.data_info[self.filtered_idxs[idx]]
        basename = query['basename']
        wav, _ = librosa.load(f"{self.root}/wav/{basename}.wav", sr=16000)
        text = query['text']

        return {
            "id": basename,
            "wav": wav,
            "text": text
        }


class LibriSpeechCCorpus(object):
    def __init__(self, root: str) -> None:
        self.root = root
        with open(f"{self.root}/data_info.json", "r", encoding="utf-8") as f:
            self.data_info = json.load(f)
        
        # Filter out long wavs > 20s
        # tt = 0
        self.filtered_idxs = []
        for idx, query in enumerate(self.data_info):
            if query["length"] <= 20 * 16000:
                self.filtered_idxs.append(idx)
                # tt += query["length"]
        # print(tt / 16000 / 60)  # about 5hr

    def __len__(self):
        return len(self.filtered_idxs)
    
    def get(self, idx):
        query = self.data_info[self.filtered_idxs[idx]]
        basename = query['basename']
        wav, _ = librosa.load(f"{self.root}/wav/{basename}.wav", sr=16000)
        text = query['text']

        return {
            "id": basename,
            "wav": wav,
            "text": text
        }


class CommonVoiceCorpus(object):

    cache_dir = "_cache/CommonVoice"

    def __init__(self, partial=True) -> None:
        if not os.path.exists(self.cache_dir):
            self.parse()
        with open(f"{self.cache_dir}/data_info.json", "r", encoding="utf-8") as f:
            basenames = json.load(f)
        self.wav_paths = []
        self.texts = []
        if partial:
            basenames = basenames[:5000]
        for basename in basenames:
            with open(f"{self.cache_dir}/text/{basename}.txt", "r", encoding="utf-8") as f:
                text = f.read()
                self.texts.append(text.strip())
            self.wav_paths.append(f"{self.cache_dir}/wav/{basename}.wav")

    def parse(self):
        basenames = []
        src_dataset = load_dataset(
            "mozilla-foundation/common_voice_16_1",
            "en",
            split="test",
            streaming=True,
            use_auth_token=True,
            trust_remote_code=True
        )
        os.makedirs(f"{self.cache_dir}/wav", exist_ok=True)
        os.makedirs(f"{self.cache_dir}/text", exist_ok=True)
        for idx, instance in tqdm(enumerate(src_dataset)):
            wav = librosa.resample(
                instance["audio"]["array"],
                orig_sr=src_dataset.features["audio"].sampling_rate,
                target_sr=16000
            )
            wavfile.write(f"{self.cache_dir}/wav/{idx:07d}.wav", 16000, (wav * 32767).astype(np.int16))
            with open(f"{self.cache_dir}/text/{idx:07d}.txt", "w", encoding="utf-8") as f:
                f.write(instance["sentence"])
            basenames.append(f"{idx:07d}")
        with open(f"{self.cache_dir}/data_info.json", "w", encoding="utf-8") as f:
            json.dump(basenames, f, indent=4)

    def __len__(self):
        return len(self.wav_paths)
    
    def get(self, idx) -> np.ndarray:
        wav, _ = librosa.load(self.wav_paths[idx], sr=16000)
        text = preprocess_text(self.texts[idx])

        return {
            "id": self.wav_paths[idx],
            "wav": wav,
            "text": text
        }


def preprocess_text(text):
    text = unicode(text)
    text = text.replace("i.e.", "that is")
    text = text.replace("e.g.", "for example")
    text = text.replace("Mr.", "Mister")
    text = text.replace("Mrs.", "Mistress")
    text = text.replace("Dr.", "Doctor")
    text = text.replace("-", " ")
    text = text.upper()
    text = re.sub("[^ A-Z']", "", text)
    text = ' '.join(text.split())
    
    return text


class ESDCorpus(object):
    def __init__(self, extra_noise=0.0) -> None:
        self.root = Define.ESD 
        with open(f"{self.root}/map_dict.json", 'r') as f:
            self.metadata = json.load(f)
        
        self.emotion2str = {
            1: "Angry",
            2: "Happy",
            3: "Neutral",
            4: "Sad",
            5: "Surprise"
        }

        # create gaussian noise
        self.extra_noise = extra_noise
        self.noises = []
        self.instances = []
        for idx in tqdm(range(1500)):
            [speaker_idx, emotion_idx, content_idx] = self.idx_transform(idx)
            basename = f"{11 + speaker_idx:04d}_{self.emotion2str[emotion_idx+1]}_{content_idx+1}"
            wav_path = f"{self.root}/wav/{basename}.wav"
            wav, _ = librosa.load(wav_path, sr=16000)
            text = self.metadata[str(content_idx + 1)]
            self.noises.append(np.random.randn(*wav.shape))
            self.instances.append({
                "audio": wav,
                "text": text
            })
            # wavfile.write("check.wav", 16000, ((wav + extra_noise * self.noises[-1]) * 32767).astype(np.int16))
            # input()

    def __len__(self):
        return 1500

    def get_from_idxs(self, speaker_idx, emotion_idx, content_idx) -> np.ndarray:
        idx = speaker_idx * 150 + emotion_idx * 30 + content_idx
        return self.get(idx)

    def idx_transform(self, idx: int):
        assert idx < 1500
        res = [0, 0, 0]
        res[2] = idx % 30
        idx = idx // 30
        res[1] = idx % 5
        idx = idx // 5
        res[0] = idx
        return res

    def get(self, idx) -> np.ndarray:
        instance = self.instances[idx]
        wav = instance["audio"]
        text = instance["text"]
        noise = self.noises[idx]

        return {
            "wav": wav + self.extra_noise * noise,
            "text": text
        }


class L2ArcticCorpus(object):
    def __init__(self) -> None:
        self.root = Define.L2ARCTIC
        self.n_per_spk = 50
        self._init_info()

    def _init_info(self):
        self.accent2str = []
        self.spks = []
        for accent in os.listdir(self.root):
            if not os.path.isdir(f"{self.root}/{accent}"):
                continue
            for spk in os.listdir(f"{self.root}/{accent}"):
                self.spks.append(f"{accent}/{spk}")
            self.accent2str.append(accent)
        self.accent2str = {i: x for i, x in enumerate(self.accent2str)}

        self.texts = []
        for i in range(self.n_per_spk):
            if i == 12 or i == 93:  # exception
                with open(f"{self.root}/Arabic/ABA/transcript/arctic_a{i+101:04d}.txt", 'r') as f:
                    text = f.read()
            else:
                with open(f"{self.root}/Arabic/ABA/transcript/arctic_a{i+1:04d}.txt", 'r') as f:
                    text = f.read()
            self.texts.append(text.strip())
        assert len(self.spks) == 24
        assert len(self.texts) == self.n_per_spk
        # print(self.accent2str)

    def __len__(self):
        return self.n_per_spk * 24

    def get_from_idxs(self, speaker_idx, content_idx) -> np.ndarray:
        if content_idx == 12 or content_idx == 93:  # exception
            basename = f"arctic_a{content_idx+101:04d}"
        else:
            basename = f"arctic_a{content_idx+1:04d}"
        wav_path = f"{self.root}/{self.spks[speaker_idx]}/wav/{basename}.wav"
        wav, _ = librosa.load(wav_path, sr=16000)
        text = self.texts[content_idx]
        return {
            "wav": wav,
            "text": text
        }

    def idx_transform(self, idx: int):
        assert idx < self.__len__()
        res = [0, 0]
        res[1] = idx % self.n_per_spk
        idx = idx // self.n_per_spk
        res[0] = idx
        return res

    def get(self, idx) -> np.ndarray:
        return self.get_from_idxs(*self.idx_transform(idx))


class NoisyL2ArcticCorpus(object):
    def __init__(self) -> None:
        self.root = Define.L2ARCTIC 
        self.musan_root = Define.MUSAN 
        self.n_per_spk = 50
        self._init_info()

    def _init_info(self):
        self.accent2str = []
        self.spks = []
        for accent in os.listdir(self.root):
            if not os.path.isdir(f"{self.root}/{accent}"):
                continue
            for spk in os.listdir(f"{self.root}/{accent}"):
                self.spks.append(f"{accent}/{spk}")
            self.accent2str.append(accent)
        self.accent2str = {i: x for i, x in enumerate(self.accent2str)}

        self.texts = []
        for i in range(self.n_per_spk):
            if i == 12 or i == 93:  # exception
                with open(f"{self.root}/Arabic/ABA/transcript/arctic_a{i+101:04d}.txt", 'r') as f:
                    text = f.read()
            else:
                with open(f"{self.root}/Arabic/ABA/transcript/arctic_a{i+1:04d}.txt", 'r') as f:
                    text = f.read()
            self.texts.append(text.strip())
        assert len(self.spks) == 24
        assert len(self.texts) == self.n_per_spk
        # print(self.accent2str)

    def __len__(self):
        return self.n_per_spk * 24

    def get_from_idxs(self, speaker_idx, content_idx) -> np.ndarray:
        if content_idx == 12 or content_idx == 93:  # exception
            basename = f"arctic_a{content_idx+101:04d}"
        else:
            basename = f"arctic_a{content_idx+1:04d}"
        wav_path = f"{self.root}/{self.spks[speaker_idx]}/wav/{basename}.wav"
        wav, _ = librosa.load(wav_path, sr=16000)
        text = self.texts[content_idx]

        # add musan noise
        total_idx = speaker_idx * self.n_per_spk + content_idx
        noise_path = f"{self.musan_root}/noise/free-sound/noise-free-sound-{total_idx % 843:04d}.wav"
        noise , _ = librosa.load(noise_path, sr=16000)
        if len(noise) > len(wav):
            noise = noise[:len(wav)]
        wav[:len(noise)] +=  noise * 0.1
        # wavfile.write("check.wav", 16000, (wav * 32767).astype(np.int16))
        # input()

        return {
            "id": wav_path,
            "wav": wav,
            "text": text
        }

    def idx_transform(self, idx: int):
        assert idx < self.__len__()
        res = [0, 0]
        res[1] = idx % self.n_per_spk
        idx = idx // self.n_per_spk
        res[0] = idx
        return res

    def get(self, idx) -> np.ndarray:
        return self.get_from_idxs(*self.idx_transform(idx))


class CHIMECorpus(object):
    def __init__(self, ascending=False):
        # Setup
        path = Define.CHIME 
        self.path = path
        
        split = ['et05_bus_real', 'et05_bus_simu', 'et05_caf_real', 'et05_caf_simu', 'et05_ped_simu', 'et05_str_real', 'et05_str_simu']
        apath = path + "/data/audio/16kHz/enhanced"
        tpath = path + "/data/transcriptions"

        file_list = []
        for s in split: 
            split_list = list(Path(os.path.join(apath, s)).glob("*.wav"))
            file_list += split_list
        
        text = []
        for f in tqdm(file_list, desc='Read text'):
            transcription = self.read_text(tpath, str(f))
            text.append(transcription)

        self.file_list, self.text = zip(*[(f_name, txt)
                                          for f_name, txt in sorted(zip(file_list, text), reverse=not ascending, key=lambda x:len(x[1]))])
    def read_text(self, tpath, file):
        '''Get transcription of target wave file, 
        it's somewhat redundant for accessing each txt multiplt times,
        but it works fine with multi-thread'''
        txt_list = os.path.join(tpath, "".join("/".join(file.split('/')[-2:]).split(".")[:-1])+'.trn')

        with open(txt_list, 'r') as fp:
            for line in fp:
                return ' '.join(line.split(' ')[1:]).strip('\n')
    
    def get(self, index):
        wav, _ = librosa.load(self.file_list[index], sr=16000)
        return {
            "id": self.file_list[index],
            "wav": wav,
            "text": self.text[index]
        }

    def __len__(self):
        return len(self.file_list)


class TEDCorpus(object):

    cache_dir = "_cache/TED"

    def __init__(self) -> None:
        if not os.path.exists(self.cache_dir):
            self.parse()
        with open(f"{self.cache_dir}/data_info.json", "r", encoding="utf-8") as f:
            info = json.load(f)
        
        self.wav_paths = []
        self.texts = []
        self.speaker_ids = []
        for (basename, speaker_id) in info:
            with open(f"{self.cache_dir}/text/{basename}.txt", "r", encoding="utf-8") as f:
                text = f.read()
                self.texts.append(text.strip())
            self.wav_paths.append(f"{self.cache_dir}/wav/{basename}.wav")
            self.speaker_ids.append(speaker_id)

    def parse(self):
        basenames = []
        speaker_ids = []
        src_dataset = load_dataset(
            "LIUM/tedlium",
            "release3",
            split="test",
            streaming=True,
            use_auth_token=True,
            trust_remote_code=True
        )
        os.makedirs(f"{self.cache_dir}/wav", exist_ok=True)
        os.makedirs(f"{self.cache_dir}/text", exist_ok=True)
        cnt = 0
        for idx, instance in tqdm(enumerate(src_dataset)):
            if instance["speaker_id"] == "inter_segment_gap":
                continue
            wav = librosa.resample(
                instance["audio"]["array"],
                orig_sr=src_dataset.features["audio"].sampling_rate,
                target_sr=16000
            )
            wavfile.write(f"{self.cache_dir}/wav/{cnt:07d}.wav", 16000, (wav * 32767).astype(np.int16))
            with open(f"{self.cache_dir}/text/{cnt:07d}.txt", "w", encoding="utf-8") as f:
                f.write(instance["text"])
            basenames.append(f"{cnt:07d}")
            speaker_ids.append(instance["speaker_id"])
            cnt += 1
        with open(f"{self.cache_dir}/data_info.json", "w", encoding="utf-8") as f:
            json.dump(list(zip(basenames, speaker_ids)), f, indent=4)

    def __len__(self):
        return len(self.wav_paths)
    
    def get(self, idx) -> np.ndarray:
        wav, _ = librosa.load(self.wav_paths[idx], sr=16000)
        text = self.preprocess_text(self.texts[idx])

        return {
            "id": self.wav_paths[idx],
            "wav": wav,
            "text": text
        }

    def preprocess_text(self, text):
        text = text.upper()
        text = text.replace(" '", "'")
        text = text.replace("-", " ")
        text = re.sub("[^ A-Z']", "", text)
        text = ' '.join(text.split())
        
        return text


class SwitchBoardCorpus(object):

    cache_dir = "_cache/switchboard"

    def __init__(self, enhance=False, ascending=False):
        if not os.path.exists(self.cache_dir):
            self.parse(enhance, ascending)
        with open(f"{self.cache_dir}/data_info.json", "r", encoding="utf-8") as f:
            self.data_info = json.load(f)
        
        # Filter out long wavs > 20s
        self.filtered_idxs = []
        for idx, query in enumerate(self.data_info):
            if query["length"] <= 20 * 16000:
                self.filtered_idxs.append(idx)

    def parse(self, enhance=False, ascending=False):
        self.path = Define.SWBD
        
        split = ['']
        apath = self.path + "/eval2000_wav_segment"
        tpath = self.path + "/eval2000_transcription"

        file_list = []
        for s in split: 
            if enhance: 
                split_list = list(Path(os.path.join(os.path.join(apath, 'se_wav'), s)).glob("*.wav"))
            else:  
                split_list = list(Path(os.path.join(apath, s)).glob("*.wav"))
            file_list += split_list
        
        text = []
        filtered_file_list = []
        for f in tqdm(file_list, desc='Read text'):
            transcription = self.read_text(tpath, str(f))
            # print(transcription)
            if transcription == None: 
                pass
            elif len(transcription.split()) <= 3:
                pass
            else:
                filtered_file_list.append(f)
                text.append(transcription)

        print(len(filtered_file_list), len(text))
        file_list = filtered_file_list
        self.file_list, self.text = zip(*[(f_name, txt)
                                          for f_name, txt in sorted(zip(file_list, text), reverse=not ascending, key=lambda x:len(x[1]))])


        data_info = []
        for i in range(len(self.file_list)):
            wav, _ = librosa.load(self.file_list[i], sr=16000)
            data_info.append({
                "basename": str(self.file_list[i]),
                "length": len(wav),
                "text": self.text[i],
            })
        os.makedirs(self.cache_dir, exist_ok=True)
        with open(f"{self.cache_dir}/data_info.json", "w", encoding="utf-8") as f:
            json.dump(data_info, f, indent=4)
    
    def read_text(self, tpath, file):
        '''Get transcription of target wave file, 
        it's somewhat redundant for accessing each txt multiplt times,
        but it works fine with multi-thread'''
        file = file.split('/')[-1].replace('wav', 'txt')
        txt_list = os.path.join(tpath, file)

        with open(txt_list, 'r') as fp:
            for line in fp:
                return line.strip('\n')

    def __len__(self):
        return len(self.filtered_idxs)
    
    def get(self, idx):
        query = self.data_info[self.filtered_idxs[idx]]
        basename = query['basename']
        wav, _ = librosa.load(basename, sr=16000)
        text = query['text']

        return {
            "id": basename,
            "wav": wav,
            "text": text
        }


if __name__ == "__main__":
    corpus = LibriSpeechCCorpus(root=f"_cache/LibriSpeech-c/GS/snr=5")
