import argparse
import os
import torch
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

import wandb
from deepcas9.training.pl_dataloader import load_ft_ensemble_dataset
from deepcas9.training.pl_train import ESM2FtEnsemble
from deepcas9.utils.misc import load_config, init_wandb

import warnings
warnings.filterwarnings("ignore")


if __name__ == '__main__':
    # Load configuration
    config = load_config('./config/config_esm2_ensemble.yaml')
    args = argparse.Namespace(**config)
    print(args)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    seed_everything(args.seed)

    # load dataloader
    folds = load_ft_ensemble_dataset(args)

    for i, (train_loader, val_loader) in enumerate(folds):
        print(f'Fold {i}')

        wandb_logger = init_wandb(args)

        # load pl object
        pl_model = ESM2FtEnsemble(esm_file=args.esm_file,
                                fine_tune_layers=args.fine_tune_layers,
                                embed_dim=args.embed_dim,
                                lr=args.lr,
                                weight_decay=args.weight_decay,
                                save_dir=f'{args.save_dir}_{args.wandb_task}',
                                last_embed_region=args.last_embed_region,
                                seed=args.seed,
                                pool_strategy=args.pooling,
                                ensemble_num=i+1).to(device)

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
                          # limit_train_batches=50,
                          # max_epochs=3,
                          max_epochs=args.max_epochs,
                          logger=wandb_logger,
                          callbacks=[early_stop_callback],
                          sync_batchnorm=True)

        trainer.fit(pl_model,
                    train_dataloaders=train_loader,
                    val_dataloaders=val_loader)

        # finish wandb
        wandb.finish()
