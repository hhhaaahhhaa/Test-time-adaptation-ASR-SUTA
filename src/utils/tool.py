import numpy as np
import torch
import random
import jiwer


def wer(a, b):
    a = jiwer.RemovePunctuation()(a)
    b = jiwer.RemovePunctuation()(b)
    return jiwer.wer(a, b, reference_transform=jiwer.wer_standardize, hypothesis_transform=jiwer.wer_standardize)


def seed_everything(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def batchify(data, batch_size, shuffle=False):
    """
    Batch generator for list data.
    """
    n_samples = len(data)
    indices = np.arange(n_samples)
    if shuffle:  # Shuffle at the start of epoch
        np.random.shuffle(indices)

    for start in range(0, n_samples, batch_size):
        end = min(start + batch_size, n_samples)
        batch_idx = indices[start:end]
        batch_data = [data[idx] for idx in batch_idx]
        yield batch_data


def unwrap_loss(loss_data):
    """
    Transform nested list of dict into dict of nested list, all dicts should have same keys.
    This is a recursive implementation.
    """
    def _unwrap(data: list):
        res = {}
        for x in data:
            if isinstance(x, list):  # not last level
                unwrapped_x = _unwrap(x)
            else:
                unwrapped_x = x
            for key in unwrapped_x:
                if key not in res:
                    res[key] = []
                res[key].append(unwrapped_x[key])
        return res
    return _unwrap(loss_data)


def call_llm_OpenAI(client, model_name, msg, max_retries=5, timeout=5):
    retry_count = 0
    while retry_count < max_retries:
        try:
            # Make the OpenAI ChatCompletion API call
            res = client.chat.completions.create(
                model=model_name,
                messages=msg,
                temperature=0,
                timeout=timeout,
            )
            return res
        except Exception as e:
            print(f"Error: {e}. Retrying... ({retry_count+1}/{max_retries})")
        
        # Increment retry counter
        retry_count += 1
    return res if retry_count < max_retries else None


async def call_llm_AsyncOpenAI(client, model_name, msg, max_retries=5, timeout=5):
    retry_count = 0
    while retry_count < max_retries:
        try:
            # Make the OpenAI ChatCompletion API call
            res = await client.chat.completions.create(
                model=model_name,
                messages=msg,
                temperature=0,
                timeout=timeout,
            )
            return res
        except Exception as e:
            print(f"Error: {e}. Retrying... ({retry_count+1}/{max_retries})")
        
        # Increment retry counter
        retry_count += 1
    return res if retry_count < max_retries else None
