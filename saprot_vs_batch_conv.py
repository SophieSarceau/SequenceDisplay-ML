import argparse
import os
from pytorch_lightning import seed_everything

from deepcas9.model.saprot import load_esm_saprot
from deepcas9.vs.dataloader import save_sp_vs_dataloader

import warnings
warnings.filterwarnings("ignore")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='DeepCas9-SaProt-VS')
    parser.add_argument('--saprot_model', type=str, default="./data/params/SaProt_35M_AF2.pt", help='SaProt model path')
    parser.add_argument('--mutation_file_path', type=str, default='./data/processed/5nnk/5nnk_nngg_mut_num.csv', help='mutation file path')
    parser.add_argument('--seq_file_path', type=str, default='./data/protein/5nnk', help='seq file path')
    parser.add_argument('--seed', type=int, default=256, help='seed')
    parser.add_argument('--save_path', type=str, default='./virtual_screening/saprot_tokens', help='save path')
    parser.add_argument('--processed_path', type=str, default='./virtual_screening/saprot_tokens', help='processed path')
    args = parser.parse_args()
    print(args)

    save_path = args.save_path
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # load model, alphabet, batch_converter for data preprocessing
    model, alphabet = load_esm_saprot(args.saprot_model)
    batch_converter = alphabet.get_batch_converter(model_name='saprot')

    seed_everything(args.seed)

    save_sp_vs_dataloader(args, batch_converter, args.save_path, args.processed_path)
