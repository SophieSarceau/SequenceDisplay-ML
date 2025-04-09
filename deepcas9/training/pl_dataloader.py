import torch
import argparse
import pandas as pd
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.model_selection import KFold


SlugCas9_string = ('DQFFWEWFWEFELFKIKIWIAGQPVRFTPFIFMFGEDGLDPVQVVVVVVVVVVVVVVVLLVVLLVVLQVLCVVQVLDDPVDQADDPCLLVLLVCLLAH'
                   'EDHSNSVSSVLSVCSNCLAAAPDPQDPDPDDPDDCLDLNNLQVVVVVVLPPDALSVVLVVVVVVPDNDDSNVHYHNVSSLVSLVSSVVSNVVVGVDD'
                   'PVSSVVSNCSSPDDDDPLQPDDPPDPGGDNRDLVSSVQVVFAAAQLGRVHTFAFCLQQLNLLLQQLLQQLQKAFPDPDHRHDALVLSVQCCVVAQL'
                   'VDQFGALCSSCVSVVHDSVRIDRFDADPVRGGGGRGNVLSVLLVVQDPPPVCSVDSVSSSQLLVLQARDDALVSSLVSNVVDPDDDDPRSSNSSSP'
                   'DDDSYDGDNHHSVLSVVCSVVSSHDSDHPVRVCVVVVRDRPDDQLQVDPFQDLVCLVVQDDQSSLSVSLSLVRQVQRVCCVPSNDHQEYEYDYDPD'
                   'SHVVSVVVVVVVVVVVVVVLVVVLVVCCVVVVQPLSVVCSLVVVLCVLLVQAALQQRDGNPVNCCRVPVVQKDWDFQAFCLQQVDPDSLRTHIHGP'
                   'VLRVQCWNHHSLCSQVVVSGPDHPVVSLVSLVVSVVDCSRHPPSSSCSNPVNDDSQQPVSSLVSCCVAFFDPDSSLVVSQVSVVSNCVSNVHNYQY'
                   'FYTYPSNLVVLCVLLVPDPDPLQFDCVSRLSSLSSSVVLVLLVPQPLNVVVNVCRRPVPDDRPDDRRDRDTSVCVSVSSDDNVSSVSSVPDDPYFF'
                   'AYDFDLDADDPFFDPDKWWWDDDPNFIFTKDKDAQLQDQQDQPVVVCCVPPVVQWPCVVPPVPLVVVVVVQCVVPVVGSRSQNVVCVVPVDARFRD'
                   'DPVPPGDGDRMTMGTDDTDDDWDWPVVVPVPDPTTMTTDDRAFSWKWWWQDPFGIDIFTDGSVQWDDDDFKIFGDVVNVVVSCVVGVNDPPIFTAD'
                   'IDGAQWWKQWPNDIWGFRHAPDNVQQKGWTAHRGRGLQSVCVVVVPDDPSTDIDHGGNPTPDMFTWHADSSRDIDGDPPDRDDDGMGGPDD')


class ESM2FtDataset(Dataset):
    def __init__(self,
                 mutation_file_path: str,
                 seq_file_path: str
                 ):
        self.mutation_file_path = mutation_file_path
        self.seq_file_path = seq_file_path
        self.sequence_feature, self.pam_mutations = self._load_data()

    def __len__(self):
        return len(self.sequence_feature)

    def __getitem__(self, idx):
        sequence_feature_tensor = self.sequence_feature[idx]
        pam_mutations_tensor = torch.tensor(self.pam_mutations[idx], dtype=torch.float32)
        return sequence_feature_tensor, pam_mutations_tensor

    def _load_data(self):
        pam_mutations, prot_sequences = [], []
        mutate_df = pd.read_csv(self.mutation_file_path)
        for idx, row in mutate_df.iterrows():
            prot_file = f'{self.seq_file_path}/{row["nnk1"]}{row["nnk2"]}{row["nnk3"]}{row["nnk4"]}{row["nnk5"]}.fasta'
            with open(prot_file, 'r') as f:
                lines = f.readlines()
                sequence = lines[1].strip()
                prot_sequences.append(sequence)
            pam_mutations.append([row['NNGA'], row['NNGT'], row['NNGC'], row['NNGG']])
        return prot_sequences, pam_mutations


class SaProtFtDataset(Dataset):
    def __init__(self,
                 mutation_file_path: str,
                 seq_file_path: str,
                 batch_converter: object
                 ):
        self.foldseek_wt_seq = SlugCas9_string.lower()
        self.mutation_file_path = mutation_file_path
        self.seq_file_path = seq_file_path
        self.batch_converter = batch_converter
        self.sequence_feature, self.pam_mutations = self._load_data()

    def __len__(self):
        return len(self.sequence_feature)

    def __getitem__(self, idx):
        sequence_feature_tensor = self.sequence_feature[idx]
        pam_mutations_tensor = torch.tensor(self.pam_mutations[idx], dtype=torch.float32)
        return sequence_feature_tensor, pam_mutations_tensor

    def _load_data(self):
        pam_mutations, prot_sequences = [], []
        mutate_df = pd.read_csv(self.mutation_file_path)
        for idx, row in mutate_df.iterrows():
            prot_file = f'{self.seq_file_path}/{row["nnk1"]}{row["nnk2"]}{row["nnk3"]}{row["nnk4"]}{row["nnk5"]}.fasta'
            with open(prot_file, 'r') as f:
                lines = f.readlines()
                sequence = lines[1].strip()
                prot_sequences.append(sequence)
            pam_mutations.append([row['NNGA'], row['NNGT'], row['NNGC'], row['NNGG']])
        new_sequences = self._foldseek_token_converter(prot_sequences)
        seq_list = [('id', prot_sequence) for prot_sequence in new_sequences]
        _, _, saprot_batch_tokens = self.batch_converter(seq_list)
        return saprot_batch_tokens, pam_mutations

    def _foldseek_token_converter(self, prot_sequence):
        new_prot_sequence = []
        for i in range(len(prot_sequence)):
            fs_token_list = []
            for aa, tdi in zip(prot_sequence[i], self.foldseek_wt_seq):
                fs_token_list.append(aa)
                fs_token_list.append(tdi)
            new_prot_sequence.append(''.join(fs_token_list))
        return new_prot_sequence


def load_ft_dataset(args: argparse.Namespace) -> tuple:
    ft_dataset = ESM2FtDataset(
        mutation_file_path=args.mutation_file_path,
        seq_file_path=args.seq_file_path,
    )
    train_length = int(0.8 * len(ft_dataset))
    val_length = int(0.1 * len(ft_dataset))
    test_length = len(ft_dataset) - train_length - val_length
    train_dataset, val_dataset, test_dataset = random_split(ft_dataset, [train_length, val_length, test_length])
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False)

    return train_loader, val_loader, test_loader

def load_ft_ensemble_dataset(args: argparse.Namespace) -> tuple:
    ft_dataset = ESM2FtDataset(
        mutation_file_path=args.mutation_file_path,
        seq_file_path=args.seq_file_path,
    )
    five_folds_num = [int(0.2 * len(ft_dataset)) for _ in range(5)]
    five_folds_num[-1] = len(ft_dataset) - sum(five_folds_num[:-1])
    five_folds = random_split(ft_dataset, five_folds_num)
    folds = []
    for i, fold in enumerate(five_folds):
        train_dataset = torch.utils.data.ConcatDataset([five_folds[j] for j in range(5) if j != i])
        test_dataset = fold
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False)
        folds.append((train_loader, test_loader))
    # kf = KFold(n_splits=5, shuffle=True, random_state=args.seed)
    # folds = []
    # for train_index, test_index in kf.split(ft_dataset):
    #     train_dataset = torch.utils.data.Subset(ft_dataset, train_index)
    #     test_dataset = torch.utils.data.Subset(ft_dataset, test_index)
    #     train_loader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True)
    #     test_loader = DataLoader(test_dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False)
    #     folds.append((train_loader, test_loader))

    return folds

def load_saprot_ft_dataset(args: argparse.Namespace, batch_converter: object) -> tuple:
    ft_dataset = SaProtFtDataset(
        mutation_file_path=args.mutation_file_path,
        seq_file_path=args.seq_file_path,
        batch_converter=batch_converter
    )
    train_length = int(0.8 * len(ft_dataset))
    val_length = int(0.1 * len(ft_dataset))
    test_length = len(ft_dataset) - train_length - val_length
    train_dataset, val_dataset, test_dataset = random_split(ft_dataset, [train_length, val_length, test_length])
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False)

    return train_loader, val_loader, test_loader

def load_sp_ft_ensemble_dataset(args: argparse.Namespace, batch_converter: object) -> tuple:
    ft_dataset = SaProtFtDataset(
        mutation_file_path=args.mutation_file_path,
        seq_file_path=args.seq_file_path,
        batch_converter=batch_converter
    )
    five_folds_num = [int(0.2 * len(ft_dataset)) for _ in range(5)]
    five_folds_num[-1] = len(ft_dataset) - sum(five_folds_num[:-1])
    five_folds = random_split(ft_dataset, five_folds_num)
    folds = []
    for i, fold in enumerate(five_folds):
        train_dataset = torch.utils.data.ConcatDataset([five_folds[j] for j in range(5) if j != i])
        test_dataset = fold
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False)
        folds.append((train_loader, test_loader))

    return folds
