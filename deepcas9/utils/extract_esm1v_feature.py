# Copyright (c) Meta Platforms, Inc. and affiliates.

import esm
import time
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.parallel import DataParallel
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
import numpy as np
import argparse
from tqdm import tqdm

import esm


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--esm_file', type=str, default="../../data/params/esm1v_t33_650M_UR90S_1.pt", help='esm file')
    parser.add_argument('--pdb_dir', type=str, default="../../data/protein/5nnk", help='pdb dir')
    parser.add_argument('--save_dir', type=str, default="../../data/feature/5nnk", help='save dir')
    args = parser.parse_args()

    return args


def load_esm(esm_file):
    start_time = time.time()
    model, alphabet = esm.pretrained.load_model_and_alphabet(esm_file)
    batch_converter = alphabet.get_batch_converter()
    model.eval()
    model = model.to('cuda')
    end_time = time.time()
    print("Load ESM2 model time: ", end_time - start_time, "s")

    return model, batch_converter, alphabet


def esm_inference(esm2, esm2_batch_converter, alphabet, seq_list):
    # convert the sequence to tokens
    _, _, esm2_batch_tokens = esm2_batch_converter(seq_list)
    batch_lens = (esm2_batch_tokens != alphabet.padding_idx).sum(1)
    esm2_batch_tokens = esm2_batch_tokens.to("cuda")
    # use data parallel
    if torch.cuda.device_count() > 1:
        esm2 = DataParallel(esm2)
    with torch.no_grad():
        results = esm2(esm2_batch_tokens, repr_layers=[33], return_contacts=True)
    token_representations = results["representations"][33]

    sequence_representations = []
    for i, tokens_len in enumerate(batch_lens):
        # sequence_representations.append(token_representations[i, 1 : tokens_len - 1].mean(0).cpu())
        sequence_representations.append(token_representations[i, 1: tokens_len - 1].cpu())

    return np.array(sequence_representations)


class SequenceDataset(Dataset):
    def __init__(self, pdb_dir):
        self.pdb_dir = pdb_dir
        self.pdb_files = os.listdir(pdb_dir)

    def __len__(self):
        return len(self.pdb_files)

    def __getitem__(self, idx):
        pdb_file = self.pdb_files[idx]
        pdb_id = pdb_file.replace(".fasta", '')
        with open(os.path.join(self.pdb_dir, pdb_file), "r") as f:
            seq = f.readlines()[1].strip()
        return pdb_id, seq


def extract_feature(esm2, esm2_batch_converter, alphabet, pdb_dir, save_dir):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    saved_pdb = os.listdir(save_dir)

    dataset = SequenceDataset(pdb_dir)
    dataloader = DataLoader(dataset, batch_size=8)

    for pdb_ids, seqs in tqdm(dataloader):
        seq_list = list(zip(pdb_ids, seqs))
        features = esm_inference(esm2, esm2_batch_converter, alphabet, seq_list)
        for pdb_id, feature in zip(pdb_ids, features):
            if pdb_id in saved_pdb:
                continue
            np.save(os.path.join(save_dir, f"{pdb_id}.npy"), feature)


if __name__ == "__main__":
    args = parse_args()

    esm1v, esm1v_batch_converter, alphabet = load_esm(args.esm_file)
    extract_feature(esm1v, esm1v_batch_converter, alphabet, args.pdb_dir, args.save_dir)
