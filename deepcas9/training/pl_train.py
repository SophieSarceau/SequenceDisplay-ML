import os
import time
import io
import copy
import pickle
from PIL import Image
from typing import Any
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch
from torch.optim import AdamW
import torch.nn as nn
import wandb
from scipy.stats import spearmanr, pearsonr
from sklearn.metrics import r2_score

from deepcas9.model.esm2 import ESM2
from pytorch_lightning import LightningModule, seed_everything
from pytorch_lightning.loggers import WandbLogger
from .utils import scatter_plot, precision_k, norm_discount_cumulative_gain
from torch.utils.data import DataLoader


class ESM2Finetune(LightningModule):
    def __init__(self,
                 esm_file: str,
                 fine_tune_layers: str,
                 embed_dim: int,
                 lr: float = 1e-4,
                 weight_decay: float = 0.01,
                 save_dir: str = 'esm2_finetune',
                 last_embed_region: int = -150,
                 test_loader: DataLoader = None,
                 seed: int = 1000,
                 pool_strategy: str = 'mean',):
        super(ESM2Finetune, self).__init__()
        self.esm_model = self._load_esm2(esm_file, fine_tune_layers)
        mutation_projection_layers = [
            nn.Linear(embed_dim, 256, bias=True),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(inplace=True),
            nn.Linear(256, 128, bias=True),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(inplace=True),
            nn.Linear(128, 64, bias=True),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(inplace=True),
            nn.Linear(64, 4, bias=True),
        ]
        self.mutation_projection = nn.Sequential(*mutation_projection_layers)
        self.mutation_projection.to(self.device)

        self.mse_loss = nn.MSELoss()
        self.lr = lr
        self.weight_decay = weight_decay
        self.pool_strategy = pool_strategy

        self.test_loader = test_loader
        self.last_embed_region = last_embed_region

        self.save_dir = save_dir
        self.best_val_loss = np.inf
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        self.seed = seed

        self.validation_prediction_dict = {'pred': [], 'true': [], 'val_loss': []}
        self.test_record = {}
        self.test_record_tag = False
        self.index2pam = {0: 'NNGA', 1: 'NNGT', 2: 'NNGC', 3: 'NNGG'}
        seed_everything(self.seed)

    def training_step(self, batch, batch_idx):
        loss = self._loss_fn(batch, 'train')
        self.log('train/mse_loss', loss)
        return loss

    def on_train_epoch_end(self):
        pass

    def on_train_end(self):
        if self.local_rank == 0:
            print("Training has finished.")
            print("Test results under the best validation loss: ", self.test_record)

    def validation_step(self, batch, batch_idx):
        loss = self._loss_fn(batch, 'val')
        self.log('validation/mse_loss', loss)
        self.log('val_loss', loss)
        return loss

    def on_validation_epoch_end(self):
        if self.local_rank == 0:
            self._evaluate_val()
            self._evaluate_test()
        # contribution_scores = self._contribution_score_test()
        # print(f'Contribution scores: {contribution_scores}')

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        return optimizer

    def _loss_fn(self, batch, mode):
        prot_sequence, pam_mutations = batch
        pam_mutations = pam_mutations.to(self.device)
        seq_list = [('id', prot_sequence) for prot_sequence in prot_sequence]
        sequence_representations = self.esm_model.esm_inference(seq_list, device=self.device)
        sequence_representations = sequence_representations[:, self.last_embed_region:, :]
        if self.pool_strategy == 'mean':
            sequence_representations = torch.mean(sequence_representations, dim=1)
        elif self.pool_strategy == 'sum':
            sequence_representations = torch.sum(sequence_representations, dim=1)
        pam_pred = self.mutation_projection(sequence_representations)
        loss = self.mse_loss(pam_pred, pam_mutations)

        if mode == 'val':
            self.validation_prediction_dict['pred'].append(pam_pred)
            self.validation_prediction_dict['true'].append(pam_mutations)
            self.validation_prediction_dict['val_loss'].append(loss)
        return loss

    def _evaluate_val(self):
        """
        Calculate metrics for each pam mode /w validation_prediction_dict and index2pam:
            R^2: coefficient of determination
            Spearman correlation
            Pearson correlation
            Plot the scatter plot of true vs pred
        """
        true_tensor = torch.cat(self.validation_prediction_dict['true'], dim=0)
        pred_tensor = torch.cat(self.validation_prediction_dict['pred'], dim=0)
        for i in range(4):
            true = true_tensor[:, i]
            pred = pred_tensor[:, i]
            r2 = r2_score(true.cpu().numpy(), pred.cpu().numpy())
            spearman = spearmanr(true.cpu().numpy(), pred.cpu().numpy())[0]
            pearson = pearsonr(true.cpu().numpy(), pred.cpu().numpy())[0]
            pk_10 = precision_k(true.cpu().numpy(), pred.cpu().numpy(), 10)
            ndcg_10 = norm_discount_cumulative_gain(true.cpu().numpy(), pred.cpu().numpy(), 10)
            pk_50 = precision_k(true.cpu().numpy(), pred.cpu().numpy(), 50)
            ndcg_50 = norm_discount_cumulative_gain(true.cpu().numpy(), pred.cpu().numpy(), 50)
            self.log(f'validation/R2_{self.index2pam[i]}', r2, rank_zero_only=True)
            self.log(f'validation/spearman_{self.index2pam[i]}', spearman, rank_zero_only=True)
            self.log(f'validation/pearson_{self.index2pam[i]}', pearson, rank_zero_only=True)
            self.log(f'validation/precision_k10_{self.index2pam[i]}', pk_10, rank_zero_only=True)
            self.log(f'validation/ndcg10_{self.index2pam[i]}', ndcg_10, rank_zero_only=True)
            self.log(f'validation/precision_k50_{self.index2pam[i]}', pk_50, rank_zero_only=True)
            self.log(f'validation/ndcg50_{self.index2pam[i]}', ndcg_50, rank_zero_only=True)
            scatter_plot(true, pred, self.index2pam[i], mode='validation')

        this_epoch_val_loss = torch.mean(torch.stack(self.validation_prediction_dict['val_loss']))
        if this_epoch_val_loss < self.best_val_loss:
            self.best_val_loss = this_epoch_val_loss
            self.test_record_tag = True
            self._save_model()

        self.validation_prediction_dict = {'pred': [], 'true': [], 'val_loss': []}

    def _evaluate_test(self):
        """
        Calculate metrics for each pam mode /w test_loader and index2pam:
            R^2: coefficient of determination
            Spearman correlation
            Pearson correlation
            Plot the scatter plot of true vs pred
        """
        self.eval()
        self.test_loader = self.test_loader
        true_tensor, pred_tensor = self._get_true_pred_tensor(self.test_loader)
        for i in range(4):
            true = true_tensor[:, i]
            pred = pred_tensor[:, i]
            r2 = r2_score(true.cpu().numpy(), pred.cpu().numpy())
            spearman = spearmanr(true.cpu().numpy(), pred.cpu().numpy())[0]
            pearson = pearsonr(true.cpu().numpy(), pred.cpu().numpy())[0]
            pk_10 = precision_k(true.cpu().numpy(), pred.cpu().numpy(), 10)
            ndcg_10 = norm_discount_cumulative_gain(true.cpu().numpy(), pred.cpu().numpy(), 10)
            pk_50 = precision_k(true.cpu().numpy(), pred.cpu().numpy(), 50)
            ndcg_50 = norm_discount_cumulative_gain(true.cpu().numpy(), pred.cpu().numpy(), 50)
            self.log(f'test/R2_{self.index2pam[i]}', r2, rank_zero_only=True)
            self.log(f'test/spearman_{self.index2pam[i]}', spearman, rank_zero_only=True)
            self.log(f'test/pearson_{self.index2pam[i]}', pearson, )
            self.log(f'test/precision_k10_{self.index2pam[i]}', pk_10, rank_zero_only=True)
            self.log(f'test/ndcg10_{self.index2pam[i]}', ndcg_10, rank_zero_only=True)
            self.log(f'test/precision_k50_{self.index2pam[i]}', pk_50, rank_zero_only=True)
            self.log(f'test/ndcg50_{self.index2pam[i]}', ndcg_50, rank_zero_only=True)
            scatter_plot(true, pred, self.index2pam[i], mode='test')
            if self.test_record_tag:
                self.test_record[f'R2_{self.index2pam[i]}'] = r2
                self.test_record[f'spearman_{self.index2pam[i]}'] = spearman
                self.test_record[f'pearson_{self.index2pam[i]}'] = pearson
                self.test_record[f'precision_k10_{self.index2pam[i]}'] = pk_10
                self.test_record[f'ndcg10_{self.index2pam[i]}'] = ndcg_10
                self.test_record[f'precision_k50_{self.index2pam[i]}'] = pk_50
                self.test_record[f'ndcg50_{self.index2pam[i]}'] = ndcg_50
        if self.test_record_tag:
            self._save_test_pred(true_tensor, pred_tensor)
        self.test_record_tag = False

    def _get_true_pred_tensor(self, loader):
        true_list = []
        pred_list = []
        for batch in tqdm(loader):
            prot_sequence, pam_mutations = batch
            pam_mutations = pam_mutations.to(self.device)
            seq_list = [('id', prot_sequence) for prot_sequence in prot_sequence]
            sequence_representations = self.esm_model.esm_inference(seq_list, device=self.device).squeeze()
            if len(sequence_representations.shape) == 2:
                sequence_representations = sequence_representations.unsqueeze(0)  # Add batch dimension
            sequence_representations = sequence_representations[:, self.last_embed_region:, :]
            if self.pool_strategy == 'mean':
                sequence_representations = torch.mean(sequence_representations, dim=1)
            elif self.pool_strategy == 'sum':
                sequence_representations = torch.sum(sequence_representations, dim=1)
            pam_pred = self.mutation_projection(sequence_representations)
            true_list.append(pam_mutations)
            pred_list.append(pam_pred)
        return torch.cat(true_list, dim=0), torch.cat(pred_list, dim=0)

    def _load_esm2(self, esm_file, fine_tune_layers):
        esm = ESM2(esm_file=esm_file)
        # Check the number of layers in the ESM2 model
        total_layer_num = len(list(esm.model.layers))
        print(f"Total number of layers in ESM-2: {total_layer_num}")

        if fine_tune_layers == 0:
            pass
        elif abs(fine_tune_layers) > total_layer_num:
            print("Error: the unfrozen layer num is larger than permitted.")
            esm.unfreeze_layers(layer_idxs=[])
        elif fine_tune_layers != "":
            layer_idxs = [total_layer_num - 1 - int(i) for i in range(abs(fine_tune_layers))]
            esm.unfreeze_layers(layer_idxs=layer_idxs)

        esm.to(self.device)

        return esm

    def _save_model(self):
        state_dict = copy.deepcopy(self.state_dict())
        torch.save(state_dict, os.path.join(self.save_dir, 'best_model.pth'))
        wandb.save(os.path.join(self.save_dir, 'best_model.pth'))
        print(f"Save best model to {os.path.join(self.save_dir, 'best_model.pth')}")

    def _save_test_pred(self, true, pred):
        # convert tensor to cpu numpy array
        true = true.cpu().numpy()
        pred = pred.cpu().numpy()
        best_test_results = {'true': true, 'pred': pred}
        pickle.dump(best_test_results, open(os.path.join(self.save_dir, 'best_test_results.pkl'), 'wb'))
        print(f"Save best test results to {os.path.join(self.save_dir, 'best_test_results.pkl')}")

    def reload_ft_model(self, model_path):
        # use torch.load with map_location to load model trained on GPU to CPU
        state_dict = torch.load(model_path, map_location=self.device)
        self.load_state_dict(state_dict)
        print(f"Reload model from {model_path}")

    def inference_single_prot(self, prot_sequence: str) -> Any:
        """
        Inference the PAM mutation prediction for a single protein sequence
        """
        self.eval()
        with torch.no_grad():
            seq_list = [('id', prot_sequence)]
            sequence_representations = self.esm_model.esm_inference(seq_list, device=self.device).squeeze()
            sequence_representations = sequence_representations[self.last_embed_region:, :]
            if self.pool_strategy == 'mean':
                sequence_representations = torch.mean(sequence_representations, dim=0)
            elif self.pool_strategy == 'sum':
                sequence_representations = torch.sum(sequence_representations, dim=0)
            pam_pred = self.mutation_projection(sequence_representations.unsqueeze(dim=0))
        return pam_pred[0].tolist()

    def _contribution_score_test(self):
        """
        Calculate the contribution score for every amino acid in the sequence
        and draw the heatmap: last 150 amino acids length x 4 PAM modes
        """
        torch.set_grad_enabled(True)
        self.eval()  # Set the model to evaluation mode
        contribution_scores = []

        # Assuming a single batch from the DataLoader for demonstration
        for batch in self.test_loader:
            prot_sequence, _ = batch

            # Forward pass
            seq_list = [('id', seq) for seq in prot_sequence]
            sequence_representations = self.esm_model.esm_inference_gradient(seq_list,
                                                                             device=self.device,
                                                                             require_grad=True).squeeze()
            sequence_representations.requires_grad = True
            sequence_representations = sequence_representations[:, self.last_embed_region:, :]
            sequence_representations = torch.sum(sequence_representations, dim=1)
            pam_pred = self.mutation_projection(sequence_representations)

            # Calculate gradients for each PAM output
            for i in range(4):
                self.zero_grad()
                pam_pred[:, i].backward(retain_graph=(i < 3))  # Only retain graph if not the last PAM mode
                contribution_scores.append(
                    sequence_representations.grad.abs().mean(dim=0).cpu().numpy())  # Get the mean abs grad

        # Convert list of numpy arrays to a single numpy array for easier plotting
        contribution_scores = np.stack(contribution_scores, axis=0)

        # Plotting the heatmap
        fig, ax = plt.subplots(figsize=(10, 5))
        cax = ax.matshow(contribution_scores, interpolation='nearest', cmap='hot')
        fig.colorbar(cax)
        ax.set_title('Contribution Scores Heatmap')
        ax.set_xlabel('Amino Acid Position')
        ax.set_ylabel('PAM Mode')
        ax.set_yticklabels([''] + [self.index2pam[i] for i in range(4)])
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=600)
        buf.seek(0)
        image = Image.open(buf)
        wandb.log({'test/contribution_scores': [wandb.Image(image, caption="Contribution Scores")]})

        return contribution_scores


class ESM2FtEnsemble(LightningModule):
    def __init__(self,
                 esm_file: str,
                 fine_tune_layers: str,
                 embed_dim: int,
                 lr: float = 1e-4,
                 weight_decay: float = 0.01,
                 save_dir: str = 'esm2_finetune',
                 last_embed_region: int = -150,
                 seed: int = 1000,
                 pool_strategy: str = 'mean',
                 ensemble_num: int = 1):
        super(ESM2FtEnsemble, self).__init__()
        self.esm_model = self._load_esm2(esm_file, fine_tune_layers)
        mutation_projection_layers = [
            nn.Linear(embed_dim, 256, bias=True),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(inplace=True),
            nn.Linear(256, 128, bias=True),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(inplace=True),
            nn.Linear(128, 64, bias=True),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(inplace=True),
            nn.Linear(64, 4, bias=True),
        ]
        self.mutation_projection = nn.Sequential(*mutation_projection_layers)
        self.mutation_projection.to(self.device)

        self.mse_loss = nn.MSELoss()
        self.lr = lr
        self.weight_decay = weight_decay
        self.pool_strategy = pool_strategy

        self.last_embed_region = last_embed_region

        # self.save_dir = f'{save_dir}_{time.strftime("%Y-%m-%d-%H-%M-%S")}'
        self.save_dir = save_dir
        self.best_val_loss = np.inf
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        self.seed = seed

        self.train_prediction_dict = {'pred': [], 'true': []}
        self.validation_prediction_dict = {'pred': [], 'true': [], 'val_loss': []}
        self.best_val_results = {}
        self.index2pam = {0: 'NNGA', 1: 'NNGT', 2: 'NNGC', 3: 'NNGG'}
        seed_everything(self.seed)

        self.ensemble_num = ensemble_num

    def training_step(self, batch, batch_idx):
        loss = self._loss_fn(batch, 'train')
        self.log('train/mse_loss', loss)
        return loss

    def on_train_epoch_end(self):
        pass

    def on_train_end(self):
        if self.local_rank == 0:
            print("Training has finished.")
            print("Test results under the best validation loss: ", self.best_val_results)

    def validation_step(self, batch, batch_idx):
        loss = self._loss_fn(batch, 'val')
        self.log('validation/mse_loss', loss)
        self.log('val_loss', loss)
        return loss

    def on_validation_epoch_end(self):
        if self.local_rank == 0:
            self._evaluate_val()

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        return optimizer

    def _loss_fn(self, batch, mode):
        prot_sequence, pam_mutations = batch
        pam_mutations = pam_mutations.to(self.device)
        seq_list = [('id', prot_sequence) for prot_sequence in prot_sequence]
        sequence_representations = self.esm_model.esm_inference(seq_list, device=self.device)
        sequence_representations = sequence_representations[:, self.last_embed_region:, :]
        if self.pool_strategy == 'mean':
            sequence_representations = torch.mean(sequence_representations, dim=1)
        elif self.pool_strategy == 'sum':
            sequence_representations = torch.sum(sequence_representations, dim=1)
        pam_pred = self.mutation_projection(sequence_representations)
        loss = self.mse_loss(pam_pred, pam_mutations)

        if mode == 'train':
            self.train_prediction_dict['pred'].append(pam_pred)
            self.train_prediction_dict['true'].append(pam_mutations)
        if mode == 'val':
            self.validation_prediction_dict['pred'].append(pam_pred)
            self.validation_prediction_dict['true'].append(pam_mutations)
            self.validation_prediction_dict['val_loss'].append(loss)
        return loss

    def _evaluate_val(self):
        """
        Calculate metrics for each pam mode /w validation_prediction_dict and index2pam:
            R^2: coefficient of determination
            Spearman correlation
            Pearson correlation
            Plot the scatter plot of true vs pred
        """
        train_true_tensor = torch.cat(self.train_prediction_dict['true'], dim=0)
        train_pred_tensor = torch.cat(self.train_prediction_dict['pred'], dim=0)
        true_tensor = torch.cat(self.validation_prediction_dict['true'], dim=0)
        pred_tensor = torch.cat(self.validation_prediction_dict['pred'], dim=0)

        this_epoch_val_loss = torch.mean(torch.stack(self.validation_prediction_dict['val_loss']))
        if this_epoch_val_loss < self.best_val_loss:
            self.best_val_loss = this_epoch_val_loss
            self.test_record_tag = True
            self._save_model()
            self._save_val_pred(true_tensor, pred_tensor)

        for i in range(4):
            train_true = train_true_tensor[:, i]
            train_pred = train_pred_tensor[:, i]
            true = true_tensor[:, i]
            pred = pred_tensor[:, i]
            r2 = r2_score(true.cpu().numpy(), pred.cpu().numpy())
            spearman = spearmanr(true.cpu().numpy(), pred.cpu().numpy())[0]
            pearson = pearsonr(true.cpu().numpy(), pred.cpu().numpy())[0]
            pk_10 = precision_k(true.cpu().numpy(), pred.cpu().numpy(), 10)
            ndcg_10 = norm_discount_cumulative_gain(true.cpu().numpy(), pred.cpu().numpy(), 10)
            pk_50 = precision_k(true.cpu().numpy(), pred.cpu().numpy(), 50)
            ndcg_50 = norm_discount_cumulative_gain(true.cpu().numpy(), pred.cpu().numpy(), 50)
            self.log(f'validation/R2_{self.index2pam[i]}', r2, rank_zero_only=True)
            self.log(f'validation/spearman_{self.index2pam[i]}', spearman, rank_zero_only=True)
            self.log(f'validation/pearson_{self.index2pam[i]}', pearson, rank_zero_only=True)
            self.log(f'validation/precision_k10_{self.index2pam[i]}', pk_10, rank_zero_only=True)
            self.log(f'validation/ndcg10_{self.index2pam[i]}', ndcg_10, rank_zero_only=True)
            self.log(f'validation/precision_k50_{self.index2pam[i]}', pk_50, rank_zero_only=True)
            self.log(f'validation/ndcg50_{self.index2pam[i]}', ndcg_50, rank_zero_only=True)
            scatter_plot(true, pred, self.index2pam[i], mode='validation')
            scatter_plot(train_true, train_pred, self.index2pam[i], mode='train')
            if self.test_record_tag:
                self.best_val_results[f'R2_{self.index2pam[i]}'] = r2
                self.best_val_results[f'spearman_{self.index2pam[i]}'] = spearman
                self.best_val_results[f'pearson_{self.index2pam[i]}'] = pearson
                self.best_val_results[f'precision_k10_{self.index2pam[i]}'] = pk_10
                self.best_val_results[f'ndcg10_{self.index2pam[i]}'] = ndcg_10
                self.best_val_results[f'precision_k50_{self.index2pam[i]}'] = pk_50
                self.best_val_results[f'ndcg50_{self.index2pam[i]}'] = ndcg_50
        self.test_record_tag = False

        self.train_prediction_dict = {'pred': [], 'true': []}
        self.validation_prediction_dict = {'pred': [], 'true': [], 'val_loss': []}

    def _load_esm2(self, esm_file, fine_tune_layers):
        esm = ESM2(esm_file=esm_file)
        # Check the number of layers in the ESM2 model
        total_layer_num = len(list(esm.model.layers))
        print(f"Total number of layers in ESM-2: {total_layer_num}")

        if fine_tune_layers == 0:
            pass
        elif abs(fine_tune_layers) > total_layer_num:
            print("Error: the unfrozen layer num is larger than permitted.")
            esm.unfreeze_layers(layer_idxs=[])
        elif fine_tune_layers != "":
            layer_idxs = [total_layer_num - 1 - int(i) for i in range(abs(fine_tune_layers))]
            esm.unfreeze_layers(layer_idxs=layer_idxs)

        esm.to(self.device)

        return esm

    def _save_val_pred(self, true, pred):
        # convert tensor to cpu numpy array
        true = true.cpu().numpy()
        pred = pred.cpu().numpy()
        best_val_results = {'true': true, 'pred': pred}
        pickle.dump(best_val_results, open(os.path.join(self.save_dir, f'best_val_results_{self.ensemble_num}.pkl'), 'wb'))

    def _save_model(self):
        state_dict = copy.deepcopy(self.state_dict())
        torch.save(state_dict, os.path.join(self.save_dir, f'esm2_ft_{self.ensemble_num}.pth'))
        wandb.save(os.path.join(self.save_dir, f'esm2_ft_{self.ensemble_num}.pth'))
        print(f"Save best model to {os.path.join(self.save_dir, f'esm2_ft_{self.ensemble_num}.pth')}")

    def reload_ft_model(self, model_path):
        # use torch.load with map_location to load model trained on GPU to CPU
        state_dict = torch.load(model_path, map_location=self.device)
        self.load_state_dict(state_dict)
        print(f"Reload model from {model_path}")

    def inference_single_prot(self, prot_sequence: str) -> Any:
        """
        Inference the PAM mutation prediction for a single protein sequence
        """
        self.eval()
        with torch.no_grad():
            seq_list = [('id', prot_sequence)]
            sequence_representations = self.esm_model.esm_inference(seq_list, device=self.device).squeeze()
            sequence_representations = sequence_representations[self.last_embed_region:, :]
            if self.pool_strategy == 'mean':
                sequence_representations = torch.mean(sequence_representations, dim=0)
            elif self.pool_strategy == 'sum':
                sequence_representations = torch.sum(sequence_representations, dim=0)
            pam_pred = self.mutation_projection(sequence_representations.unsqueeze(dim=0))
        return pam_pred[0].tolist()

    def _contribution_score_test(self):
        """
        Calculate the contribution score for every amino acid in the sequence
        and draw the heatmap: last 150 amino acids length x 4 PAM modes
        """
        torch.set_grad_enabled(True)
        self.eval()  # Set the model to evaluation mode
        contribution_scores = []

        # Assuming a single batch from the DataLoader for demonstration
        for batch in self.test_loader:
            prot_sequence, _ = batch

            # Forward pass
            seq_list = [('id', seq) for seq in prot_sequence]
            sequence_representations = self.esm_model.esm_inference_gradient(seq_list,
                                                                             device=self.device,
                                                                             require_grad=True).squeeze()
            sequence_representations.requires_grad = True
            sequence_representations = sequence_representations[:, self.last_embed_region:, :]
            sequence_representations = torch.sum(sequence_representations, dim=1)
            pam_pred = self.mutation_projection(sequence_representations)

            # Calculate gradients for each PAM output
            for i in range(4):
                self.zero_grad()
                pam_pred[:, i].backward(retain_graph=(i < 3))  # Only retain graph if not the last PAM mode
                contribution_scores.append(
                    sequence_representations.grad.abs().mean(dim=0).cpu().numpy())  # Get the mean abs grad

        # Convert list of numpy arrays to a single numpy array for easier plotting
        contribution_scores = np.stack(contribution_scores, axis=0)

        # Plotting the heatmap
        fig, ax = plt.subplots(figsize=(10, 5))
        cax = ax.matshow(contribution_scores, interpolation='nearest', cmap='hot')
        fig.colorbar(cax)
        ax.set_title('Contribution Scores Heatmap')
        ax.set_xlabel('Amino Acid Position')
        ax.set_ylabel('PAM Mode')
        ax.set_yticklabels([''] + [self.index2pam[i] for i in range(4)])
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=600)
        buf.seek(0)
        image = Image.open(buf)
        wandb.log({'test/contribution_scores': [wandb.Image(image, caption="Contribution Scores")]})

        return contribution_scores
