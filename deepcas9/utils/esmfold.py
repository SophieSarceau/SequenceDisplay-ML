import os
import esm
import time
import torch
import argparse
import biotite.structure.io as bsio
from tqdm import tqdm
from Bio import SeqIO


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--fasta_file', type=str, default="../../data/preliminary_study/nnk.fasta", help='fasta file path')
    parser.add_argument('--save_dir', type=str, default="../../data/preliminary_study/nnk_esmfold_pdb", help='save dir')
    args = parser.parse_args()

    return args

def load_esmfold():
    start_time = time.time()
    model = esm.pretrained.esmfold_v1()
    model = model.eval().cuda()
    end_time = time.time()
    print("Load ESMFold model time: ", end_time - start_time, "s")

    return model

def load_sequence(fasta_file_path):
    sequences = {}
    for record in SeqIO.parse(fasta_file_path, "fasta"):
        sequences[record.id] = str(record.seq)

    return sequences

def esmfold_inference(model, sequences):
    pdbs = {}
    with torch.no_grad():
        for id, sequence in tqdm(sequences.items()):
            output = model.infer_pdb(sequence)
            pdbs[id] = output

    return pdbs

def save_pdb(pdbs, save_dir):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    for id, pdb in pdbs.items():
        with open(f"{save_dir}/{id}.pdb", "w") as f:
            f.write(pdb)

def save_plddt(pdbs, save_dir):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    for id, pdb in pdbs.items():
        struct = bsio.load_structure(save_dir + f"/{id}.pdb", extra_fields=["b_factor"])
        with open(f"{save_dir}/{id}.txt", "w") as f:
            f.write(str(struct.b_factor.mean()))


if __name__ == "__main__":
    args = parse_args()
    model = load_esmfold()
    sequences = load_sequence(args.fasta_file)
    pdbs = esmfold_inference(model, sequences)
    save_pdb(pdbs, args.save_dir)
    save_plddt(pdbs, args.save_dir)
