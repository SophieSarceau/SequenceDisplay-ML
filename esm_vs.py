import os
import sklearn
import argparse
import torch
from pytorch_lightning import seed_everything

from deepcas9.vs.dataloader import load_vs_dataloader
from deepcas9.training.pl_vs_esm2 import ESM2Finetune
from deepcas9.utils.misc import load_config

import warnings
warnings.filterwarnings("ignore")


if __name__ == '__main__':
    # Load configuration
    config = load_config('./config/config_esm2_vs.yaml')
    args = argparse.Namespace(**config)
    print(args)

    save_dir = args.save_dir
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    seed_everything(args.seed)

    # load dataloader
    vs_loader = load_vs_dataloader(args)

    # load pl object
    pl_model = ESM2Finetune(esm_file=args.esm_file,
                            fine_tune_layers=args.fine_tune_layers,
                            embed_dim=args.embed_dim,
                            last_embed_region=args.last_embed_region,
                            vs_loader=vs_loader,
                            seed=args.seed,
                            pool_strategy=args.pooling).to(device)

    fold_num = 5
    for fold in range(fold_num):
        pl_model.reload_ft_model(model_path=os.path.join(args.esm_ensemble_path, f'esm2_ft_{fold+1}.pth'))
        pl_model.virtual_screening(save_dir=os.path.join(save_dir, f"model{fold+1}_esm2_vs_result"))
