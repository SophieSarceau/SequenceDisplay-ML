import os
import torch
import numpy as np
import pytorch_lightning as pl
from omegaconf import DictConfig
from pytorch_lightning.callbacks import ModelCheckpoint

from matplotlib import pyplot as plt


def get_predictions(model: pl.LightningModule, dataloader: pl.LightningDataModule):
    predictions = []
    targets = []
    for batch in dataloader:
        x, y = batch
        y_hat = model(x)
        predictions.append(y_hat.detach().cpu().numpy())
        targets.append(y.detach().cpu().numpy())
    return np.concatenate(predictions), np.concatenate(targets)


def visualization(preds: torch.Tensor, targets: torch.Tensor, step: str, task: str):
    current_path = os.getcwd()
    results_path = os.path.join(current_path, "graphs")
    if not os.path.exists(results_path):
        os.makedirs(results_path)
    if task == "pam mut num" or task == "pam mut prob":
        nnga_pred = preds[:, 0]
        nngt_pred = preds[:, 1]
        nngc_pred = preds[:, 2]
        nngg_pred = preds[:, 3]
        nnga_target = targets[:, 0]
        nngt_target = targets[:, 1]
        nngc_target = targets[:, 2]
        nngg_target = targets[:, 3]

    if task == "pam mut num":
        # plot the figure for nnga, nngt, nngc, nngg in four subplots in one row
        fig, axs = plt.subplots(1, 4, figsize=(20, 5))
        axs[0].scatter(nnga_target, nnga_pred, color="blue")
        a_k, a_b = np.polyfit(nnga_target, nnga_pred, 1)
        axs[0].plot(nnga_target, a_k * nnga_target + a_b, color="red")
        axs[0].set_title(f"NNGA Mutation Num ({step})")
        axs[0].set_xlabel("Ground Truth")
        axs[0].set_ylabel("Prediction")
        axs[1].scatter(nngt_target, nngt_pred, color="blue")
        t_k, t_b = np.polyfit(nngt_target, nngt_pred, 1)
        axs[1].plot(nngt_target, t_k * nngt_target + t_b, color="red")
        axs[1].set_title(f"NNGT Mutation Num ({step})")
        axs[1].set_xlabel("Ground Truth")
        axs[1].set_ylabel("Prediction")
        axs[2].scatter(nngc_target, nngc_pred, color="blue")
        c_k, c_b = np.polyfit(nngc_target, nngc_pred, 1)
        axs[2].plot(nngc_target, c_k * nngc_target + c_b, color="red")
        axs[2].set_title(f"NNGC Mutation Num ({step})")
        axs[2].set_xlabel("Ground Truth")
        axs[2].set_ylabel("Prediction")
        axs[3].scatter(nngg_target, nngg_pred, color="blue")
        g_k, g_b = np.polyfit(nngg_target, nngg_pred, 1)
        axs[3].plot(nngg_target, g_k * nngg_target + g_b, color="red")
        axs[3].set_title(f"NNGG Mutation Num ({step})")
        axs[3].set_xlabel("Ground Truth")
        axs[3].set_ylabel("Prediction")
        # save the figure locally with tight layout
        fig.tight_layout()
        plt.savefig(os.path.join(results_path, f"{step}.png"), dpi=400)
    elif task == "pam mut prob":
        # plot the figure for nnga, nngt, nngc, nngg in four subplots in one row
        fig, axs = plt.subplots(1, 4, figsize=(20, 5))
        axs[0].scatter(nnga_target, nnga_pred, color="blue")
        a_k, a_b = np.polyfit(nnga_target, nnga_pred, 1)
        axs[0].plot(nnga_target, a_k * nnga_target + a_b, color="red")
        axs[0].set_title(f"NNGA Mutation Prob ({step})")
        axs[0].set_xlabel("Ground Truth")
        axs[0].set_ylabel("Prediction")
        axs[1].scatter(nngt_target, nngt_pred, color="blue")
        t_k, t_b = np.polyfit(nngt_target, nngt_pred, 1)
        axs[1].plot(nngt_target, t_k * nngt_target + t_b, color="red")
        axs[1].set_title(f"NNGT Mutation Prob ({step})")
        axs[1].set_xlabel("Ground Truth")
        axs[1].set_ylabel("Prediction")
        axs[2].scatter(nngc_target, nngc_pred, color="blue")
        c_k, c_b = np.polyfit(nngc_target, nngc_pred, 1)
        axs[2].plot(nngc_target, c_k * nngc_target + c_b, color="red")
        axs[2].set_title(f"NNGC Mutation Prob ({step})")
        axs[2].set_xlabel("Ground Truth")
        axs[2].set_ylabel("Prediction")
        axs[3].scatter(nngg_target, nngg_pred, color="blue")
        g_k, g_b = np.polyfit(nngg_target, nngg_pred, 1)
        axs[3].plot(nngg_target, g_k * nngg_target + g_b, color="red")
        axs[3].set_title(f"NNGG Mutation Prob ({step})")
        axs[3].set_xlabel("Ground Truth")
        axs[3].set_ylabel("Prediction")
        # save the figure locally
        fig.tight_layout()
        plt.savefig(os.path.join(results_path, f"{step}.png"), dpi=400)


def visualize_results(model: pl.LightningModule, checkpoint_callback: ModelCheckpoint, 
                      datamodule: pl.LightningDataModule, cfg: DictConfig):
    # Load the best model
    model.load_state_dict(torch.load(checkpoint_callback.best_model_path)["state_dict"])
    model.eval()
    # Get the train, val, and test dataloaders
    train_loader = datamodule.train_dataloader()
    val_loader = datamodule.val_dataloader()
    test_loader = datamodule.test_dataloader()
    # Get the predictions and targets for each dataset
    train_preds, train_targets = get_predictions(model, train_loader)
    val_preds, val_targets = get_predictions(model, val_loader)
    test_preds, test_targets = get_predictions(model, test_loader)
    # Plot the results
    visualization(train_preds, train_targets, "train", cfg.general.task)
    visualization(val_preds, val_targets, "val", cfg.general.task)
    visualization(test_preds, test_targets, "test", cfg.general.task)
