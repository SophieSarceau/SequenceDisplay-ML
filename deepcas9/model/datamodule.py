import os
import torch
import pytorch_lightning as pl
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from utils import data_loader


class SlugCas9Data(Dataset):
    def __init__(self, esm2_features, target):
        self.esm2_features = esm2_features.astype("float32")
        self.target = target.astype("float32")
        self.length = len(esm2_features)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        return self.esm2_features[idx], self.target[idx]


class SlugCas9Module(pl.LightningDataModule):
    def __init__(self, cfg):
        super(SlugCas9Module, self).__init__()
        self.cfg = cfg
        self.task = cfg.general.task

    def setup(self, stage=None):
        if self.task == "pam mut num":
            nnks, nnk_feature, target, prot_seq = data_loader.load_nnk_mut_num(
                self.cfg.general.mut_num_file,
                self.cfg.general.esm_file,
                self.cfg.general.seq_file,
            )
        elif self.task == "pam mut prob":
            nnks, nnk_feature, target, prot_seq = data_loader.load_nnk_mut_prob(
                self.cfg.general.mut_prob_file,
                self.cfg.general.esm_file,
                self.cfg.general.seq_file,
            )

        slugcas9_dataset = SlugCas9Data(nnk_feature, target)
        # random split the dataset into train val and test
        self.train, self.val, self.test = random_split(
            slugcas9_dataset,
            [int(0.8*len(slugcas9_dataset)), int(0.1*len(slugcas9_dataset)), len(slugcas9_dataset)-int(0.8*len(slugcas9_dataset))-int(0.1*len(slugcas9_dataset))],
            generator=torch.Generator().manual_seed(42)
        )

    def train_dataloader(self):
        return DataLoader(self.train, batch_size=self.cfg.train.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val, batch_size=self.cfg.train.batch_size, shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.test, batch_size=self.cfg.train.batch_size, shuffle=False)
