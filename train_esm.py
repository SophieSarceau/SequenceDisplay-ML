import argparse
import os
import sklearn
import torch
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

import wandb
from deepcas9.training.pl_dataloader import load_ft_dataset
from deepcas9.training.pl_train import ESM2Finetune
from deepcas9.utils.misc import load_config, init_wandb

import warnings
warnings.filterwarnings("ignore")


if __name__ == '__main__':
    # Load configuration
    config = load_config('./config/config_esm2_train.yaml')
    args = argparse.Namespace(**config)
    print(args)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    seed_everything(args.seed)

    wandb_logger = init_wandb(args)

    # load dataloader
    train_loader, val_loader, test_loader = load_ft_dataset(args)

    # load pl object
    pl_model = ESM2Finetune(esm_file=args.esm_file,
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
