import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import multiprocessing as mp

import warnings
warnings.filterwarnings('ignore')

MIN_AVAIL_NUM = 100
CPU_NUM = 24


def get_protein_features(esm_file: str, nnks: str) -> np.array:
    # get the protein features
    feature_path = os.path.join(esm_file, nnks + '.npy')
    nnks_feature = np.load(feature_path, allow_pickle=True)
    nnks_feature = nnks_feature[-150:, :]
    # nnks_feature = np.mean(nnks_feature, axis=0)

    return nnks_feature


def process_row_finetune(row, nnk_num, seq_file):
    if row['count'] < MIN_AVAIL_NUM:
        return None
    nnk = ''.join(row[['nnk' + str(j + 1) for j in range(nnk_num)]])
    mut_num = [row['NNGA'], row['NNGT'], row['NNGC'], row['NNGG']]
    with open(os.path.join(seq_file, nnk + '.fasta'), 'r') as f:
        lines = f.readlines()
        prot_seq = lines[1].strip()

    return nnk, mut_num, prot_seq


def process_row(row, nnk_num, esm_file, seq_file):
    if row['count'] < MIN_AVAIL_NUM:
        return None
    nnk = ''.join(row[['nnk' + str(j + 1) for j in range(nnk_num)]])
    feature = get_protein_features(esm_file, nnk)
    mut_num = [row['NNGA'], row['NNGT'], row['NNGC'], row['NNGG']]
    with open(os.path.join(seq_file, nnk + '.fasta'), 'r') as f:
        lines = f.readlines()
        prot_seq = lines[1].strip()
    return nnk, feature, mut_num, prot_seq


def load_nnk_mutnum_finetune(data_file: str, seq_file: str) -> tuple:
    """
    Load the mutation results of Slug-Cas9.
    Params:
        data_file: str, the path of the file containing the mutation results of Slug-Cas9
        seq_file: str, the path of the folder containing the protein sequence of Slug-Cas9
    Returns:
        nnks: np.array, the nnk combinations
        mut_num: np.array, the mutation results of Slug-Cas9
        prot_seq: np.array, the protein sequence of Slug-Cas9
    """
    # read the csv file
    df = pd.read_csv(data_file)
    # get the headers
    headers = list(df.columns)
    nnk_num = np.sum([1 for header in headers if 'nnk' in header])
    print("The number of nnks in Slug-Cas9 is: ", nnk_num)

    nnks = []
    mut_num = []
    prot_seq = []

    with mp.Pool(CPU_NUM) as pool:
        args = [(i, df.iloc[i], nnk_num, seq_file) for i in range(len(df))]
        results = [pool.apply_async(process_row, arg[1:]) for arg in tqdm(args)]
        results = [res.get() for res in results]
    nnks, mut_num, prot_seq = zip(*[res for res in results if res is not None])
    nnks = np.array(nnks, dtype=str)
    print("The number of nnks in Slug-Cas9 is: ", len(nnks))
    mut_num = np.array(mut_num)
    prot_seq = np.array(prot_seq, dtype=str)

    return nnks, mut_num, prot_seq

def load_nnk_mut_num(data_file: str, esm_file: str, seq_file: str) -> tuple:
    """
    Load the mutation results of Slug-Cas9.
    Params:
        data_file: str, the path of the file containing the mutation results of Slug-Cas9
        esm_file: str, the path of the folder containing the ESM features of Slug-Cas9
        seq_file: str, the path of the folder containing the protein sequence of Slug-Cas9
    Returns:
        nnks: np.array, the nnk combinations
        nnk_feature: np.array, the ESM features of Slug-Cas9
        mut_num: np.array, the mutation results of Slug-Cas9
        prot_seq: np.array, the protein sequence of Slug-Cas9
    """
    # read the csv file
    df = pd.read_csv(data_file)
    # get the headers
    headers = list(df.columns)
    nnk_num = np.sum([1 for header in headers if 'nnk' in header])
    print("The number of nnks in Slug-Cas9 is: ", nnk_num)

    nnks = []
    nnk_feature = []
    mut_num = []
    prot_seq = []

    with mp.Pool(CPU_NUM) as pool:
        args = [(i, df.iloc[i], nnk_num, esm_file, seq_file) for i in range(len(df))]
        results = [pool.apply_async(process_row, arg[1:]) for arg in tqdm(args)]
        results = [res.get() for res in results]
    nnks, nnk_feature, mut_num, prot_seq = zip(*[res for res in results if res is not None])
    nnks = np.array(nnks, dtype=str)
    print("The number of nnks in Slug-Cas9 is: ", len(nnks))
    nnk_feature = np.array(nnk_feature)
    nnk_feature = (nnk_feature - np.mean(nnk_feature, axis=0)) / np.std(nnk_feature, axis=0)
    nnk_feature = (nnk_feature - np.min(nnk_feature, axis=0)) / (np.max(nnk_feature, axis=0) - np.min(nnk_feature, axis=0)) + 0.1
    mut_num = np.array(mut_num)
    prot_seq = np.array(prot_seq, dtype=str)

    return nnks, nnk_feature, mut_num, prot_seq


def load_nnk_mut_prob(data_file: str, esm_file: str, seq_file: str) -> tuple:
    """
    Load the mutation results of Slug-Cas9.
    Params:
        data_file: str, the path of the file containing the mutation probability
        results of Slug-Cas9
        esm_file: str, the path of the folder containing the ESM features of Slug-Cas9
        seq_file: str, the path of the folder containing the protein sequence of Slug-Cas9
    Returns:
        nnks: np.array, the nnk combinations
        nnk_feature: np.array, the ESM features of Slug-Cas9
        mut_prob: np.array, the mutation probability results of Slug-Cas9
        prot_seq: np.array, the protein sequence of Slug-Cas9
    """
    # read the csv file
    df = pd.read_csv(data_file)
    # get the headers
    headers = list(df.columns)
    nnk_num = np.sum([1 for header in headers if 'nnk' in header])
    print("The number of nnks in Slug-Cas9 is: ", nnk_num)

    nnks = []
    nnk_feature = []
    mut_prob = []
    prot_seq = []

    for i in range(len(df)):
        if df.iloc[i]['count'] < MIN_AVAIL_NUM:
            continue
        nnk = ''
        for j in range(nnk_num):
            nnk += df.iloc[i]['nnk' + str(j + 1)]
        nnks.append(nnk)
        feature = get_protein_features(esm_file, nnk)
        nnk_feature.append(feature)
        mut_prob.append([df.iloc[i]['A_G1'], df.iloc[i]['A_G2'], df.iloc[i]['A_G3'],
                         df.iloc[i]['A_G4'], df.iloc[i]['A_G5'], df.iloc[i]['A_G6'],
                         df.iloc[i]['T_G1'], df.iloc[i]['T_G2'], df.iloc[i]['T_G3'],
                         df.iloc[i]['T_G4'], df.iloc[i]['T_G5'], df.iloc[i]['T_G6'],
                         df.iloc[i]['C_G1'], df.iloc[i]['C_G2'], df.iloc[i]['C_G3'],
                         df.iloc[i]['C_G4'], df.iloc[i]['C_G5'], df.iloc[i]['C_G6'],
                         df.iloc[i]['G_G1'], df.iloc[i]['G_G2'], df.iloc[i]['G_G3'],
                         df.iloc[i]['G_G4'], df.iloc[i]['G_G5'], df.iloc[i]['G_G6']])
        # read the protein sequence
        with open(os.path.join(seq_file, nnk + '.fasta'), 'r') as f:
            lines = f.readlines()
            prot_seq.append(lines[1].strip())
    nnks = np.array(nnks, dtype=str)
    print("The number of nnks in Slug-Cas9 is: ", len(nnks))
    nnk_feature = np.array(nnk_feature)
    nnk_feature = (nnk_feature - np.mean(nnk_feature, axis=0)) / np.std(nnk_feature, axis=0)
    nnk_feature = (nnk_feature - np.min(nnk_feature, axis=0)) / (np.max(nnk_feature, axis=0) - np.min(nnk_feature, axis=0)) + 0.1
    mut_prob = np.array(mut_prob)
    prot_seq = np.array(prot_seq, dtype=str)

    return nnks, nnk_feature, mut_prob, prot_seq


def load_nnk_prob_comb(data_file: str, esm_file: str, seq_file: str) -> tuple:
    """
    Load the mutation probability combination results of Slug-Cas9.
    Params:
        data_file: str, the path of the file containing the mutation probability
        results of Slug-Cas9
        esm_file: str, the path of the folder containing the ESM features of Slug-Cas9
        seq_file: str, the path of the folder containing the protein sequence of Slug-Cas9
    Returns:
        nnks: np.array, the nnk combinations
        nnk_feature: np.array, the ESM features of Slug-Cas9
        mut_prob_comb: np.array, the mutation probability results of Slug-Cas9
        prot_seq: np.array, the protein sequence of Slug-Cas9
    """
    # read the csv file
    df = pd.read_csv(data_file)
    # get the headers
    headers = list(df.columns)
    nnk_num = np.sum([1 for header in headers if 'nnk' in header])
    print("The number of nnks in Slug-Cas9 is: ", nnk_num)

    nnks = []
    nnk_feature = []
    mut_prob_comb = []
    prot_seq = []

    for i in range(len(df)):
        if df.iloc[i]['count'] < MIN_AVAIL_NUM:
            continue
        nnk = ''
        for j in range(nnk_num):
            nnk += df.iloc[i]['nnk' + str(j + 1)]
        nnks.append(nnk)
        faeture = get_protein_features(esm_file, nnk)
        nnk_feature.append(faeture)
        mut_prob_comb.append([df.iloc[i]['NNGA'], df.iloc[i]['NNGT'],
                              df.iloc[i]['NNGC'], df.iloc[i]['NNGG']])
        # read the protein sequence
        with open(os.path.join(seq_file, nnk + '.fasta'), 'r') as f:
            lines = f.readlines()
            prot_seq.append(lines[1].strip())
    nnks = np.array(nnks, dtype=str)
    print("The number of nnks in Slug-Cas9 is: ", len(nnks))
    nnk_feature = np.array(nnk_feature)
    nnk_feature = (nnk_feature - np.mean(nnk_feature, axis=0)) / np.std(nnk_feature, axis=0)
    nnk_feature = (nnk_feature - np.min(nnk_feature, axis=0)) / (np.max(nnk_feature, axis=0) - np.min(nnk_feature, axis=0)) + 0.1
    mut_prob_comb = np.array(mut_prob_comb)
    prot_seq = np.array(prot_seq, dtype=str)

    return nnks, nnk_feature, mut_prob_comb, prot_seq
