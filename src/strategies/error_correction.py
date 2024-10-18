import os
from torch.utils.data import Dataset
import yaml
from tqdm import tqdm
from openai import OpenAI, AsyncOpenAI
import json
from collections import defaultdict
from tqdm.asyncio import tqdm_asyncio

from ..system.suta import SUTASystem
from ..utils.async_request import OrderPreservedAsyncRequestHandler
from ..utils.tool import wer, call_llm_AsyncOpenAI
from ..utils.prompter import Prompter
from .base import IStrategy


class RescoreStrategy(IStrategy):
    def __init__(self, config) -> None:
        self.config = config
        self.strategy_config = config["strategy_config"]
        self.system = SUTASystem(config["system_config"])
        self.system.eval()

        self._log = None
    
    def run(self, ds: Dataset):
        long_cnt = 0
        self._log = defaultdict(list)
        for sample in tqdm(ds):
            if len(sample["wav"]) > self.strategy_config["max_length"]:
                long_cnt += 1
                continue
            self._log["n_words"].append(len(sample["text"].split(" ")))
            trans = self.system.beam_inference([sample["wav"]], n_best=1)[0]
            err = wer(sample["text"], trans)
            self._log["wers"].append(err)
            self._log["transcriptions"].append((sample["text"], trans))
            self._log["basenames"].append(sample["id"])

            # loss
            loss = self.system.calc_suta_loss([sample["wav"]])
            ctc_loss = self.system.calc_ctc_loss([sample["wav"]], [sample["text"]])
            loss["ctc_loss"] = ctc_loss["ctc_loss"]
            self._log["losses"].append(loss)

            self._log["logits"].append(self.system.calc_logits([sample["wav"]])[0])
            
        print("#Too long: ", long_cnt)
        
        return self._log
        
    def get_adapt_count(self):
        return self.system.adapt_count


class LLMStrategy(IStrategy):
    def __init__(self, config) -> None:
        self.config = config
        self.strategy_config = config["strategy_config"]
        self.system = SUTASystem(config["system_config"])
        self.system.eval()
        self._log = None

        # LLM setup
        # self.prompter = Prompter("GenSEC")
        self.prompter = Prompter("v0")
        self.llm_client = OpenAI(
            api_key=os.environ.get("OPENAI_API_KEY"),
        )
        f = open('vocab.json')
        self.vocab = json.load(f)

    def inference(self, sample) -> str:
        nbest_trans = self.system.beam_inference([sample["wav"]], n_best=5)[0]
        msg = []
        if "system_prompt" in self.prompter.template:
            msg.append({"role": "system", "content": self.prompter.template['system_prompt']})
        msg.append({"role": "user", "content": self.prompter.generate_prompt({"5best": '\n'.join(nbest_trans)})})
        # print(msg)
        res = self.llm_client.chat.completions.create(
            model="gpt-3.5-turbo-0125",
            messages=msg,
            temperature=0,
        )
        llm_response = res.choices[0].message.content
        self._log["LLM"].append(llm_response)
        # print(res)

        # normalize
        # prefix = "The true transcription from the 5-best hypotheses is: "  # GenSEC
        prefix = "The corrected transcription is: "  # v0
        idx = llm_response.find(prefix)
        try:
            assert idx >= 0
            ans = llm_response[idx+len(prefix):].strip().upper()
            trans = ""
            for c in ans:
                if c == " " or c in self.vocab:
                    trans += c
        except:
            print(f"LLM format error: {res}")
            trans = nbest_trans[0]
        # print(trans)
        return trans

    def run(self, ds: Dataset):
        long_cnt = 0
        self._log = defaultdict(list)
        for sample in tqdm(ds):
            if len(sample["wav"]) > self.strategy_config["max_length"]:
                long_cnt += 1
                continue
            self._log["n_words"].append(len(sample["text"].split(" ")))

            self.system.eval()
            trans = self.inference(sample)
            err = wer(sample["text"], trans)
            self._log["wers"].append(err)
            self._log["transcriptions"].append((sample["text"], trans))
            self._log["basenames"].append(sample["id"])
            
            # loss
            loss = self.system.calc_suta_loss([sample["wav"]])
            ctc_loss = self.system.calc_ctc_loss([sample["wav"]], [sample["text"]])
            loss["ctc_loss"] = ctc_loss["ctc_loss"]
            self._log["losses"].append(loss)

            self._log["logits"].append(self.system.calc_logits([sample["wav"]])[0])
            
        print("#Too long: ", long_cnt)
        
        return self._log
    
    def get_adapt_count(self):
        return self.system.adapt_count


class AsyncLLMStrategy(IStrategy):
    def __init__(self, config) -> None:
        self.config = config
        self.strategy_config = config["strategy_config"]
        self.system = SUTASystem(config["system_config"])
        self.system.eval()
        self._log = None

        # LLM setup
        # self.prompter = Prompter("GenSEC")
        self.prompter = Prompter("v0")
        self.llm_client = AsyncOpenAI(
            api_key=os.environ.get("OPENAI_API_KEY"),
        )
        f = open('vocab.json')
        self.vocab = json.load(f)

    async def wrap_request(self, req):
        self.pb_req.update(1)
        sample, nbest_trans = req["sample"], req["nbest_trans"]
        msg = []
        if "system_prompt" in self.prompter.template:
            msg.append({"role": "system", "content": self.prompter.template['system_prompt']})
        msg.append({"role": "user", "content": self.prompter.generate_prompt({"5best": '\n'.join(nbest_trans)})})
        
        res = await call_llm_AsyncOpenAI(self.llm_client, model_name="gpt-3.5-turbo-0125", msg=msg, max_retries=5)
        res = res.choices[0].message.content

        return {
            "sample": sample,
            "nbest_trans": nbest_trans,
            "llm_response": res,
        }

    async def postprocess(self, res):
        self.pb_res.update(1)
        sample, nbest_trans, llm_response = res["sample"], res["nbest_trans"], res["llm_response"]
        self._log["LLM"].append(llm_response)
        # print(llm_response)

        # normalize
        # prefix = "The true transcription from the 5-best hypotheses is: "  # GenSEC
        prefix = "The corrected transcription is: "  # v0
        idx = llm_response.find(prefix)
        try:
            assert idx >= 0
            ans = llm_response[idx+len(prefix):].strip().upper()
            trans = ""
            for c in ans:
                if c == " " or c in self.vocab:
                    trans += c
        except:
            print(f"LLM format error: {res}")
            trans = nbest_trans[0]
        # print(trans)
    
        err = wer(sample["text"], trans)
        self._log["wers"].append(err)
        self._log["transcriptions"].append((sample["text"], trans))
        self._log["basenames"].append(sample["id"])
        
        # loss
        # loss = self.system.calc_suta_loss([sample["wav"]])
        # ctc_loss = self.system.calc_ctc_loss([sample["wav"]], [sample["text"]])
        # loss["ctc_loss"] = ctc_loss["ctc_loss"]
        # self._log["losses"].append(loss)

        # self._log["logits"].append(self.system.calc_logits([sample["wav"]])[0])

    def inference(self, sample) -> None:  # GenSEC
        nbest_trans = self.system.beam_inference([sample["wav"]], n_best=5)[0]
        assert self.request_handler.is_running(), "Request handler is dead, please check your code..."
        self.request_handler.request({
            "sample": sample,
            "nbest_trans": nbest_trans
        })

    def run(self, ds: Dataset):
        self.request_handler = OrderPreservedAsyncRequestHandler(max_req=50)
        self.request_handler.run(self.wrap_request, self.postprocess)
        long_cnt = 0
        self._log = defaultdict(list)

        # progress bar
        pb_infer = tqdm_asyncio(total=len(ds), desc="Infer", position=0)
        self.pb_req = tqdm_asyncio(total=len(ds), desc="Request ", position=1)
        self.pb_res = tqdm_asyncio(total=len(ds), desc="Response", position=2)

        for sample in ds:
            pb_infer.update(1)
            if len(sample["wav"]) > self.strategy_config["max_length"]:
                long_cnt += 1
                continue
            self._log["n_words"].append(len(sample["text"].split(" ")))

            self.system.eval()
            _ = self.inference(sample)
                    
        print("#Too long: ", long_cnt)

        # wait until run() shutdown
        self.request_handler.close()
        
        return self._log
    
    def get_adapt_count(self):
        return self.system.adapt_count