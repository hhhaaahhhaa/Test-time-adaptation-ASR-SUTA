import os
import numpy as np
import scipy
import pickle
import scipy.special
import json
from matplotlib import pyplot as plt
from tqdm import tqdm

from src.utils.tool import wer, unwrap_loss


def avg(data, n_words=None, window_size=10):
    if n_words is None:
        n_words = [1] * len(data)
    n = len(data)
    res = []
    st = 0
    while 1:
        ed = min(n, st+window_size)
        s = 0
        for k in range(st, ed):
            s += data[k] * n_words[k]
        res.append(s / sum(n_words[st:ed]))
        st = ed
        if st >= n:
            break
    return res


def load_results(exp_root: str):
    with open (f"results/benchmark/{exp_root}/result/results.pkl", "rb") as f:
        log = pickle.load(f)
    if "losses" in log:
        log["losses"] = unwrap_loss(log["losses"])
    return log


def plot_wer(output_path: str, exp_roots: list[str], legends: list[str]=None, styles=None):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    if legends is None:
        legends = exp_roots
    if styles is None:
        styles = ["-o"] * len(exp_roots)
    
    # plot
    plt.title(f"WER")
    plt.xlabel("Time")
    plt.ylabel("WER")

    for (exp_root, legend, style) in zip(exp_roots, legends, styles):
        result = load_results(exp_root=exp_root)
        wers, n_words = result["wers"], result["n_words"]
        x = np.arange(0, len(wers), 100)
        y = avg(wers, n_words=n_words, window_size=100)
        plt.plot(x, y, style, label=legend, markersize=3)

    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_loss(loss_key: str, output_path: str, exp_roots: list[str], legends: list[str]=None, styles=None):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    if legends is None:
        legends = exp_roots
    if styles is None:
        styles = ["-o"] * len(exp_roots)
    
    # plot
    plt.title(f"{loss_key}")
    plt.xlabel("Time")
    plt.ylabel(f"{loss_key}")

    for (exp_root, legend, style) in zip(exp_roots, legends, styles):
        result = load_results(exp_root=exp_root)
        losses = result["losses"][loss_key]
        x = np.arange(0, len(losses), 100)
        y = avg(losses, window_size=100)
        plt.plot(x, y, style, label=legend, markersize=3)

    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_logits(exp_root: str):
    output_dir = f"results/benchmark/{exp_root}/vis/logits"
    os.makedirs(output_dir, exist_ok=True)
    result = load_results(exp_root=exp_root)

    f = open('vocab.json')
    vocab = json.load(f)
    # id2text = {v: k for k, v in vocab.items()}
    transcriptions = result["transcriptions"]
    logits = result["logits"]
    wers = result["wers"]
    for i, ((gt, pred), logit, wer) in tqdm(enumerate(zip(transcriptions, logits, wers))):
        if (i+1) % 100 != 0:
            continue
        logit = scipy.special.softmax(logit, axis=-1)
        # predicted_ids = np.argmax(logit, axis=-1)
        # full_pred = "".join([id2text[x] for x in predicted_ids])
        # print(gt)
        # print(pred)
        # print(full_pred)
        # print(wer)
        fig = plot_attn({
            "title": f"GT:{gt}\nPred:{pred}\nWER:{wer*100:.2f}%",
            "x_labels": [" "] * logit.shape[0],
            "y_labels": list(vocab.keys()),
            "attn": logit.T,
        })
        fig.savefig(f"{output_dir}/{i:07d}.jpg")
        plt.close(fig)
        # input()


def plot_attn(info):
    fig = plt.figure(figsize=(32, 8))
    ax = fig.add_subplot(111)
    cax = ax.matshow(info["attn"], vmin=0, vmax=1)
    # fig.colorbar(cax, ax=ax)

    ax.set_title(info["title"], fontsize=28)
    ax.set_xticks(np.arange(len(info["x_labels"])))
    ax.set_xticklabels(info["x_labels"], rotation=90, fontsize=8)
    ax.set_yticks(np.arange(len(info["y_labels"])))
    ax.set_yticklabels(info["y_labels"], fontsize=8)
    fig.tight_layout()

    return fig


def plot_asr_heatmap(probs, vocab=None, show_top_token=True, figsize=(50, 6)):
    """
    Plot a heatmap of ASR output probabilities over time.
    
    Args:
        probs (np.ndarray): ASR output probabilities with shape (T, V).
        vocab (List[str], optional): List of vocabulary tokens. Defaults to indices if None.
        show_top_token (bool): Whether to annotate the top token at each time step.
        figsize (tuple): Figure size for the plot.
    """
    T, V = probs.shape
    if vocab is None:
        vocab = [str(i) for i in range(V)]

    fig, ax = plt.subplots(figsize=figsize)
    cax = ax.imshow(probs.T, aspect='auto', origin='lower', cmap='viridis')

    ax.set_xlabel("Time Step")
    ax.set_ylabel("Vocabulary Token")
    ax.set_yticks(np.arange(V))
    ax.set_yticklabels(vocab)
    ax.set_title("ASR Output Probability Heatmap")

    fig.colorbar(cax, ax=ax, label="Probability")

    if show_top_token:
        top_tokens = np.argmax(probs, axis=1)
        for t, token_idx in enumerate(top_tokens):
            if token_idx == 0:
                continue
            ax.text(t, token_idx+1, vocab[token_idx], ha='center', va='center', color='white', fontsize=8, fontweight='bold')
    plt.tight_layout()
    return fig
