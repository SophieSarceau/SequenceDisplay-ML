import argparse
import os
import torch
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import WandbLogger

import wandb
from deepcas9.model.esm2 import ESM2
from deepcas9.model.saprot import load_esm_saprot
from deepcas9.training.pl_dataloader import load_saprot_ft_dataset
from deepcas9.training.pl_train_saprot import SaProtFinetune
from deepcas9.utils.misc import load_config, init_wandb

import warnings
warnings.filterwarnings("ignore")


if __name__ == '__main__':
    # Load configuration
    config = load_config('./config/config_saprot_train.yaml')
    args = argparse.Namespace(**config)
    print(args)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    seed_everything(args.seed)

    wandb_logger = init_wandb(args)

    # load model, alphabet, batch_converter for data preprocessing
    model, alphabet = load_esm_saprot(args.saprot_file)
    batch_converter = alphabet.get_batch_converter(model_name='saprot')

    # load dataloader
    train_loader, val_loader, test_loader = load_saprot_ft_dataset(args, batch_converter)

    # load pl object
    pl_model = SaProtFinetune(saprot_model=model,
                              alphabet=alphabet,
                              fine_tune_layers=args.fine_tune_layers,
                              embed_dim=args.embed_dim,
                              lr=args.lr,
                              weight_decay=args.weight_decay,
                              save_dir=f'{args.save_dir}_{args.wandb_task}',
                              last_embed_region=args.last_embed_region,
                              test_loader=test_loader,
                              seed=args.seed,
                              pool_strategy=args.pooling).to(device)

    # train
    early_stop_callback = EarlyStopping(monitor='val_loss',
                                        patience=args.patience,
                                        verbose=True,
                                        mode='min')

    torch.set_float32_matmul_precision('high')

    trainer = Trainer(strategy=args.strategy,
                      accelerator="gpu",
                      devices=args.gpu_num,
                      num_nodes=1,
                      num_sanity_val_steps=0,
                      max_epochs=args.max_epochs,
                      logger=wandb_logger,
                      callbacks=[early_stop_callback],
                      sync_batchnorm=True)

    trainer.fit(pl_model,
                train_dataloaders=train_loader,
                val_dataloaders=val_loader)

    # finish wandb
    wandb.finish()
