import io
import torch
import wandb
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np


def precision_k(true: np.array, pred: np.array, k: int) -> float:
    idx_pred = np.argsort(pred)[::-1]
    idx_pred = idx_pred[:k]
    idx_true = np.argsort(true)[::-1]
    idx_true = idx_true[:k]
    intersection = len(set(idx_pred).intersection(set(idx_true)))

    return intersection / k


def norm_discount_cumulative_gain(true: np.array, pred: np.array, k: int) -> float:
    idx_pred = np.argsort(pred)[::-1]
    idx_pred = idx_pred[:k]
    idx_true = np.argsort(true)[::-1]
    idx_true = idx_true[:k]

    dcg = 0
    for i in range(k):
        if i < len(idx_pred):
            if idx_pred[i] in idx_true:
                dcg += true[idx_pred[i]] / np.log2(i + 2)
    idcg = 0
    for i in range(k):
        if i < len(idx_true):
            idcg += true[idx_true[i]] / np.log2(i + 2)

    return dcg / idcg


def scatter_plot(true, pred, pam_mode, mode='train'):
    plt.figure()
    plt.scatter(true.cpu().numpy(), pred.cpu().numpy())
    plt.xlabel('True')
    plt.ylabel('Pred')
    plt.title(f'{pam_mode} True vs Pred')
    true, pred = true.to(dtype=torch.float32), pred.to(dtype=torch.float32)
    m, b = np.polyfit(true.cpu().numpy(), pred.cpu().numpy(), 1)
    plt.plot(true.cpu().numpy(), m * true.cpu().numpy() + b, color='red')
    plt.grid()
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=600)
    buf.seek(0)
    image = Image.open(buf)
    wandb.log({f'{mode}/{pam_mode}_scatter_plot': [wandb.Image(image)]})
    plt.close()
