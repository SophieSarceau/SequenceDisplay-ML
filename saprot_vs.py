import argparse
import os
import torch
import sklearn
from pytorch_lightning import seed_everything

from deepcas9.model.saprot import load_esm_saprot
from deepcas9.vs.dataloader import load_sp_vs_dataloader
from deepcas9.training.pl_train_saprot import SaProtVS
from deepcas9.utils.misc import load_config

import warnings
warnings.filterwarnings("ignore")

if __name__ == '__main__':
    # parser = argparse.ArgumentParser(description='DeepCas9-SaProt-VS')
    # parser.add_argument('--last_embed_region', type=int, default=-150, help='last embed region')
    # parser.add_argument('--batch_size', type=int, default=12, help='batch size')
    # parser.add_argument('--num_workers', type=int, default=2, help='num workers')
    # parser.add_argument('--save_dir', type=str, default='esm2_t12_35M_UR50D', help='save dir')
    # parser.add_argument('--saprot_model', type=str, default='data/params/SaProt_35M_AF2/SaProt_35M_AF2.pt', help='SaProt model path')
    # parser.add_argument('--vs_token_path', type=str, default='virtual_screening/saprot_tokens', help='virtual screening token path')
    # parser.add_argument('--mutation_file_path', type=str, default='data/processed/5nnk/5nnk_nngg_mut_num.csv', help='mutation file path')
    # parser.add_argument('--seq_file_path', type=str, default='data/protein/5nnk', help='seq file path')
    # parser.add_argument('--model_num', type=int, default=1, help='model num')
    # parser.add_argument('--seed', type=int, default=256, help='seed')
    # args = parser.parse_args()
    # print(args)
    # Load configuration
    config = load_config('./config/config_saprot_vs.yaml')
    args = argparse.Namespace(**config)
    print(args)

    save_dir = args.save_dir
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # load model, alphabet, batch_converter for data preprocessing
    model, alphabet = load_esm_saprot(args.saprot_file)

    seed_everything(args.seed)

    # load pl object
    pl_model = SaProtVS(saprot_model=model,
                        alphabet=alphabet,
                        fine_tune_layers=args.fine_tune_layers,
                        last_embed_region=args.last_embed_region,
                        vs_token_path=args.vs_token_path,
                        seed=args.seed,
                        pool_strategy=args.pooling,
                        args=args).to(device)

    fold_num = 5
    for fold in range(fold_num):
        pl_model.reload_ft_model(model_path=os.path.join(args.saprot_ensemble_path, f"saprot_ft_{fold+1}.pth"))
        pl_model.virtual_screening(save_dir=os.path.join(save_dir, f"model{fold+1}_saprot_vs_result"))
