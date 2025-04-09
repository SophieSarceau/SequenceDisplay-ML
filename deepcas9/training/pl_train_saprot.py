import os
import time
import io
import copy
from PIL import Image
from typing import Any
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch
from torch.optim import AdamW
import torch.nn as nn
import wandb
from scipy.stats import spearmanr, pearsonr
from sklearn.metrics import r2_score

from deepcas9.model.saprot import SaProt
from deepcas9.vs.dataloader import SPVSDataset
from pytorch_lightning import LightningModule, seed_everything
from pytorch_lightning.loggers import WandbLogger
from .utils import scatter_plot, precision_k, norm_discount_cumulative_gain
from torch.utils.data import DataLoader
import pickle


class SaProtFinetune(LightningModule):
    def __init__(self,
                 saprot_model: nn.Module,
                 alphabet: object,
                 fine_tune_layers: str,
                 embed_dim: int,
                 lr: float = 1e-4,
                 weight_decay: float = 0.01,
                 save_dir: str = 'saprot_finetune',
                 last_embed_region: int = -150,
                 test_loader: DataLoader = None,
                 seed: int = 1000,
                 pool_strategy: str = 'mean',):
        super(SaProtFinetune, self).__init__()
        self.saprot_model = self._load_saprot(saprot_model, alphabet, fine_tune_layers)
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

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        return optimizer

    def _loss_fn(self, batch, mode):
        saprot_tokens, pam_mutations = batch
        pam_mutations = pam_mutations.to(self.device)
        sequence_representations = self.saprot_model.saprot_inference(saprot_tokens, device=self.device)
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
        for batch in loader:
            saprot_tokens, pam_mutations = batch
            pam_mutations = pam_mutations.to(self.device)
            sequence_representations = self.saprot_model.saprot_inference(saprot_tokens, device=self.device)
            if len(sequence_representations.shape) == 2:
                sequence_representations = sequence_representations.unsqueeze(0)
            sequence_representations = sequence_representations[:, self.last_embed_region:, :]
            if self.pool_strategy == 'mean':
                sequence_representations = torch.mean(sequence_representations, dim=1)
            elif self.pool_strategy == 'sum':
                sequence_representations = torch.sum(sequence_representations, dim=1)
            pam_pred = self.mutation_projection(sequence_representations)
            true_list.append(pam_mutations)
            pred_list.append(pam_pred)
        return torch.cat(true_list, dim=0), torch.cat(pred_list, dim=0)

    def _load_saprot(self, saprot_model, alphabet, fine_tune_layers):
        saprot = SaProt(saprot_model, alphabet)
        # Check the number of layers in the SaProt model
        total_layer_num = len(list(saprot.model.layers))
        print(f"Total number of layers in SaProt: {total_layer_num}")

        if fine_tune_layers == 0:
            pass
        elif abs(fine_tune_layers) > total_layer_num:
            print("Error: the unfrozen layer num is larger than permitted.")
            saprot.unfreeze_layers(layer_idxs=[])
        elif fine_tune_layers != "":
            layer_idxs = [total_layer_num - 1 - int(i) for i in range(abs(fine_tune_layers))]
            saprot.unfreeze_layers(layer_idxs=layer_idxs)

        saprot.to(self.device)

        return saprot

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
            sequence_representations = self.saprot_model.saprot_inference(seq_list, device=self.device)
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
            sequence_representations = self.saprot_model.saprot_inference(seq_list, device=self.device)
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


class SaProtFtEnsemble(LightningModule):
    def __init__(self,
                 saprot_model: nn.Module,
                 alphabet: object,
                 fine_tune_layers: str,
                 embed_dim: int,
                 lr: float = 1e-4,
                 weight_decay: float = 0.01,
                 save_dir: str = 'saprot_finetune',
                 last_embed_region: int = -150,
                 seed: int = 1000,
                 pool_strategy: str = 'mean',
                 task: str = 'saprot-12layers-[ft-11]',
                 ensemble_num: int = 1):
        super(SaProtFtEnsemble, self).__init__()
        self.saprot_model = self._load_saprot(saprot_model, alphabet, fine_tune_layers)
        self.saprot_model.to(self.device)
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

        self.save_dir = save_dir
        self.best_val_loss = np.inf
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        self.seed = seed

        self.train_prediction_dict = {'pred': [], 'true': []}
        self.validation_prediction_dict = {'pred': [], 'true': [], 'val_loss': []}
        self.index2pam = {0: 'NNGA', 1: 'NNGT', 2: 'NNGC', 3: 'NNGG'}
        self.best_val_results = {}
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
            print("Best validation results: ", self.best_val_results)

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
        saprot_tokens, pam_mutations = batch
        pam_mutations = pam_mutations.to(self.device)
        sequence_representations = self.saprot_model.saprot_inference(saprot_tokens, device=self.device)
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

    def _load_saprot(self, saprot_model, alphabet, fine_tune_layers):
        saprot = SaProt(saprot_model, alphabet)
        # Check the number of layers in the SaProt model
        total_layer_num = len(list(saprot.model.layers))
        print(f"Total number of layers in SaProt: {total_layer_num}")

        if fine_tune_layers == 0:
            pass
        elif abs(fine_tune_layers) > total_layer_num:
            print("Error: the unfrozen layer num is larger than permitted.")
            saprot.unfreeze_layers(layer_idxs=[])
        elif fine_tune_layers != "":
            layer_idxs = [total_layer_num - 1 - int(i) for i in range(abs(fine_tune_layers))]
            saprot.unfreeze_layers(layer_idxs=layer_idxs)

        saprot.to(self.device)

        return saprot

    def _save_val_pred(self, true, pred):
        # convert tensor to cpu numpy array
        true = true.cpu().numpy()
        pred = pred.cpu().numpy()
        best_val_results = {'true': true, 'pred': pred}
        pickle.dump(best_val_results, open(os.path.join(self.save_dir, f'best_val_results_{self.ensemble_num}.pkl'), 'wb'))

    def _save_model(self):
        state_dict = copy.deepcopy(self.state_dict())
        torch.save(state_dict, os.path.join(self.save_dir, f'saprot_ft_{self.ensemble_num}.pth'))
        wandb.save(os.path.join(self.save_dir, f'saprot_ft_{self.ensemble_num}.pth'))
        print(f"Save best model to {os.path.join(self.save_dir, f'saprot_ft_{self.ensemble_num}.pth')}")

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
            # seq_list = [('id', prot_sequence)]
            sequence_representations = self.saprot_model.saprot_inference(prot_sequence, device=self.device)
            # sequence_representations = sequence_representations.squeeze(dim=0)
            sequence_representations = sequence_representations[:, self.last_embed_region:, :]
            # print(sequence_representations.shape)
            if self.pool_strategy == 'mean':
                sequence_representations = torch.mean(sequence_representations, dim=1)
            elif self.pool_strategy == 'sum':
                sequence_representations = torch.sum(sequence_representations, dim=1)
            pam_pred = self.mutation_projection(sequence_representations)
        return pam_pred, sequence_representations

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
            sequence_representations = self.saprot_model.saprot_inference(seq_list, device=self.device)
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

class SaProtVS(LightningModule):
    def __init__(self,
                 saprot_model: nn.Module,
                 alphabet: object,
                 fine_tune_layers: str,
                 save_dir: str = 'saprot_finetune',
                 last_embed_region: int = -150,
                 embed_dim: int = 480,
                 vs_token_path: str = None,
                 seed: int = 1000,
                 pool_strategy: str = 'mean',
                 args: object = None):
        super(SaProtVS, self).__init__()
        self.saprot_model = self._load_saprot(saprot_model, alphabet, fine_tune_layers)
        self.saprot_model.to(self.device)
        if vs_token_path is not None:
            print("Model is in virtual screening mode.")
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
        self.pool_strategy = pool_strategy

        self.vs_token_path = vs_token_path
        self.last_embed_region = last_embed_region

        self.save_dir = save_dir
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        self.seed = seed

        self.index2pam = {0: 'NNGA', 1: 'NNGT', 2: 'NNGC', 3: 'NNGG'}
        seed_everything(self.seed)

        self.args = args

    def _load_saprot(self, saprot_model, alphabet, fine_tune_layers):
        saprot = SaProt(saprot_model, alphabet)
        # Check the number of layers in the SaProt model
        total_layer_num = len(list(saprot.model.layers))
        print(f"Total number of layers in SaProt: {total_layer_num}")

        if fine_tune_layers == 0:
            pass
        elif abs(fine_tune_layers) > total_layer_num:
            print("Error: the unfrozen layer num is larger than permitted.")
            saprot.unfreeze_layers(layer_idxs=[])
        elif fine_tune_layers != "":
            layer_idxs = [total_layer_num - 1 - int(i) for i in range(abs(fine_tune_layers))]
            saprot.unfreeze_layers(layer_idxs=layer_idxs)

        saprot.to(self.device)

        return saprot

    def reload_ft_model(self, model_path):
        # use torch.load with map_location to load model trained on GPU to CPU
        state_dict = torch.load(model_path, map_location=self.device)
        # # print(state_dict.keys())
        self.load_state_dict(state_dict)
        print(f"Reload model from {model_path}")

    def inference_single_prot(self, saprot_token: str) -> Any:
        """
        Inference the PAM mutation prediction for a single protein sequence
        """
        self.eval()
        with torch.no_grad():
            sequence_representations = self.saprot_model.saprot_inference([saprot_token], device=self.device)
            sequence_representations = sequence_representations[self.last_embed_region:, :]
            if self.pool_strategy == 'mean':
                sequence_representations = torch.mean(sequence_representations, dim=0)
            elif self.pool_strategy == 'sum':
                sequence_representations = torch.sum(sequence_representations, dim=0)
            pam_pred = self.mutation_projection(sequence_representations.unsqueeze(dim=0))
        return pam_pred[0].tolist()

    def inference_multiple_prot(self, saprot_tokens: list) -> Any:
        """
        Inference the PAM mutation prediction for multiple protein sequences
        """
        self.eval()
        with torch.no_grad():
            sequence_representations = self.saprot_model.saprot_inference(saprot_tokens, device=self.device)
            sequence_representations = sequence_representations[:, self.last_embed_region:, :]
            if self.pool_strategy == 'mean':
                sequence_representations = torch.mean(sequence_representations, dim=1)
            elif self.pool_strategy == 'sum':
                sequence_representations = torch.sum(sequence_representations, dim=1)
            pam_pred = self.mutation_projection(sequence_representations)
        return pam_pred.tolist()

    def load_ft_weight(self, model_path):
        self.reload_ft_model(model_path)

    def virtual_screening(self, save_dir):
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        self.eval()
        # loop vs_token_path for every file
        for file in os.listdir(self.vs_token_path):
            # check if the file has been processed
            if os.path.exists(f'{save_dir}/{file}'):
                print(f"{file} has been processed.")
                continue
            print(f"Processing {file}")
            pam_pred_list = []
            mutation_list = []
            vs_dataset = SPVSDataset(os.path.join(self.vs_token_path, file))
            vs_loader = DataLoader(vs_dataset, batch_size=self.args.batch_size,
                                   shuffle=False, num_workers=self.args.num_workers)
            # Initialize tqdm with total number of batches
            with tqdm(total=len(vs_loader), desc=f"Processing {file}") as pbar:
                for idx, batch in enumerate(vs_loader):
                    saprot_tokens, mutation = batch
                    pam_pred = self.inference_multiple_prot(saprot_tokens)
                    pam_pred_list.extend(pam_pred)
                    mutation_list.extend(mutation)
                    # Update the progress bar
                    pbar.update(1)
            with open(f'{save_dir}/{file}', 'wb') as f:
                pickle.dump({'pred': pam_pred_list, 'mutation': mutation_list}, f)
