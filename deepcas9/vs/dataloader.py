import os
import torch
import pandas as pd
import tqdm
import numpy as np
from torch.utils.data import Dataset, DataLoader, random_split


AA_DICT = {
    'Ala': 'A',
    'Arg': 'R',
    'Asn': 'N',
    'Asp': 'D',
    'Cys': 'C',
    'Glu': 'E',
    'Gln': 'Q',
    'Gly': 'G',
    'His': 'H',
    'Ile': 'I',
    'Leu': 'L',
    'Lys': 'K',
    'Met': 'M',
    'Phe': 'F',
    'Pro': 'P',
    'Ser': 'S',
    'Thr': 'T',
    'Trp': 'W',
    'Tyr': 'Y',
    'Val': 'V'
}


slugcas9_WT = ('MNQKFILGLAIGITSVGYGLIDYETKNIIDAGVRLFPEANVENNEGRRSKRGSRRLKRRRIHRLERVKKLLEDYNLLDQSQIPQSTNPYAIRVKGLSEALSKD'
               'ELVIALLHIAKRRGIHKIDVIDSNDDVGNELSTKEQLNKNSKLLKDKFVCQIQLERMNEGQVRGEKNRFKTADIIKEIIQLLNVQKNFHQLDENFINKYIEL'
               'VEMRREYFEGPGKGSPYGWEGDPKAWYETLMGHCTYFPDELRSVKYAYSADLFNALNDLNNLVIQRDGLSKLEYHEKYHIIENVFKQKKKPTLKQIANEINV'
               'NPEDIKGYRITKSGKPQFTEFKLYHDLKSVLFDQSILENEDVLDQIAEILTIYQDKDSIKSKLTELDILLNEEDKENIAQLTGYTGTHRLSLKCIRLVLEEQ'
               'WYSSRNQMEIFTHLNIKPKKINLTAANKIPKAMIDEFILSPVVKRTFGQAINLINKIIEKYGVPEDIIIELARENNSKDKQKFINEMQKKNENTRKRINEII'
               'GKYGNQNAKRLVEKIRLHDEQEGKCLYSLESIPLEDLLNNPNHYEVDHIIPRSVSFDNSYHNKVLVKQSEASKKSNLTPYQYFNSGKSKLSYNQFKQHILN'
               'LSKSQDRISKKKKEYLLEERDINKFEVQKEFINRNLVDTRYATRELTNYLKAYFSANNMNVKVKTINGSFTDYLRKVWKFKKERNHGYKHHAEDALIIANAD'
               'FLFKENKKLKAVNSVLEKPEIESKQLDIQVDSEDNYSEMFIIPKQVQDIKDFRNFKYSHRVDKKPNRQLINDTLYSTRKKDNSTYIVQTIKDIYAKDNTTLK'
               'KQFDKSPEKFLMYQHDPRTFEKLEVIMKQYANEKNPLAKYHEETGEYLTKYSKKNNGPIVKSLKYIGNKLGSHLDVTHQFKSSTKKLVKLSIKPYRFDVYLT'
               'DKGYKFITISYLDVLKKDNYYYIPEQKYDKLKLGKAIDKNAKFIASFYKNDLIKLDGEIYKIIGVNSDTRNMIELDLPDIRYKEYCELNNIKGEPRIKKTIG'
               'KKVNSIEKLTTDVLGNVFTNTQYTKPQLLFKRGN')

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

def _load_mutations():
    mutation_combinations = []
    for nnk1 in AA_DICT.values():
        for nnk2 in AA_DICT.values():
            for nnk3 in AA_DICT.values():
                for nnk4 in AA_DICT.values():
                    for nnk5 in AA_DICT.values():
                        mutation_combinations.append(f'{nnk1}{nnk2}{nnk3}{nnk4}{nnk5}')
    return set(mutation_combinations)


class VSDataset(Dataset):
    def __init__(self,
                 mutation_file_path: str,
                 seq_file_path: str
                 ):
        self.mutation_file_path = mutation_file_path
        self.seq_file_path = seq_file_path
        self.pam_order = ['NNGA', 'NNGT', 'NNGC', 'NNGG']
        self.predicted_sequences, self.mutation = self._load_data()

    def __len__(self):
        return len(self.predicted_sequences)

    def __getitem__(self, idx):
        sequence_feature_tensor = self.predicted_sequences[idx]
        mutation = self.mutation[idx]
        return sequence_feature_tensor, mutation

    def _load_data(self):
        mutation_combinations = _load_mutations()
        print(f'Number of mutation combinations: {len(mutation_combinations)}')
        mutate_df = pd.read_csv(self.mutation_file_path)
        mutate_df['combined'] = mutate_df.apply(lambda x: ''.join([AA_DICT[x[f"nnk{i}"]] for i in range(1, 6)]), axis=1)
        mutation_combinations.difference_update(set(mutate_df['combined']))
        print(f'Number of mutation combinations after remove training data: {len(mutation_combinations)}')
        prot_sequences = []
        mutation_list = []
        for mutation in tqdm.tqdm(mutation_combinations):
            original_seq = slugcas9_WT
            original_seq = original_seq[:983] + mutation[0] + original_seq[984:]
            original_seq = original_seq[:984] + mutation[1] + original_seq[985:]
            original_seq = original_seq[:989] + mutation[2] + original_seq[990:]
            original_seq = original_seq[:1011] + mutation[3] + original_seq[1012:]
            original_seq = original_seq[:1015] + mutation[4] + original_seq[1016:]
            prot_sequences.append(original_seq)
            mutation_list.append(mutation)
        return prot_sequences, mutation_list

class SPVSDataset(Dataset):
    def __init__(self,
                 sp_token_file: str,
                 ):
        self.sp_token_file = sp_token_file
        self.pam_order = ['NNGA', 'NNGT', 'NNGC', 'NNGG']
        self.sp_tokens, self.mutation = self._load_data()

    def __len__(self):
        return len(self.sp_tokens)

    def __getitem__(self, idx):
        sequence_feature_tensor = self.sp_tokens[idx]
        mutation = self.mutation[idx]
        return sequence_feature_tensor, mutation

    def _load_data(self):
        sp_tokens = np.load(self.sp_token_file, allow_pickle=True)
        sp_tokens = sp_tokens.item()
        return sp_tokens['tokens'], sp_tokens['mutation']

class SPVSBatchConvert():
    def __init__(self,
                 mutation_file_path: str,
                 seq_file_path: str,
                 batch_converter: object,
                 save_path: str,
                 processed_path: str
                 ):
        self.foldseek_wt_seq = SlugCas9_string.lower()
        self.mutation_file_path = mutation_file_path
        self.seq_file_path = seq_file_path
        self.pam_order = ['NNGA', 'NNGT', 'NNGC', 'NNGG']
        self.batch_converter = batch_converter
        self.save_path = save_path
        self.processed_path = processed_path

    def read_processed_data(self):
        files = os.listdir(self.processed_path)
        mutations = []
        for file in tqdm.tqdm(files):
            sample = np.load(os.path.join(self.processed_path, file), allow_pickle=True)
            sample = sample.item()
            mutations.extend(sample['mutation'])

        return mutations

    def save_data(self):
        mutation_combinations = _load_mutations()
        print(f'Number of mutation combinations: {len(mutation_combinations)}')
        mutate_df = pd.read_csv(self.mutation_file_path)
        mutate_df['combined'] = mutate_df.apply(lambda x: ''.join([AA_DICT[x[f"nnk{i}"]] for i in range(1, 6)]), axis=1)
        mutate_processed = self.read_processed_data()
        mutation_combinations.difference_update(set(mutate_df['combined']))
        mutation_combinations.difference_update(set(mutate_processed))
        print(f'Number of mutation combinations after remove training data: {len(mutation_combinations)}')
        prot_df_sequences = []
        mutation_df_list = []
        for mutation in tqdm.tqdm(mutate_df['combined']):
            original_seq = slugcas9_WT
            original_seq = original_seq[:983] + mutation[0] + original_seq[984:]
            original_seq = original_seq[:984] + mutation[1] + original_seq[985:]
            original_seq = original_seq[:989] + mutation[2] + original_seq[990:]
            original_seq = original_seq[:1011] + mutation[3] + original_seq[1012:]
            original_seq = original_seq[:1015] + mutation[4] + original_seq[1016:]
            prot_df_sequences.append(original_seq)
            mutation_df_list.append(mutation)
        prot_sequences = []
        mutation_list = []
        for mutation in tqdm.tqdm(mutation_combinations):
            original_seq = slugcas9_WT
            original_seq = original_seq[:983] + mutation[0] + original_seq[984:]
            original_seq = original_seq[:984] + mutation[1] + original_seq[985:]
            original_seq = original_seq[:989] + mutation[2] + original_seq[990:]
            original_seq = original_seq[:1011] + mutation[3] + original_seq[1012:]
            original_seq = original_seq[:1015] + mutation[4] + original_seq[1016:]
            prot_sequences.append(original_seq)
            mutation_list.append(mutation)
        print(f"Number of mutation combinations: {len(mutation_list)}")
        mutation_list = list(set(mutation_list))
        print(f"Number of mutation combinations after remove duplicates: {len(mutation_list)}")
        new_df_sequences = self._foldseek_token_converter(prot_df_sequences)
        seq_df_list = [('id', prot_df_sequence) for prot_df_sequence in new_df_sequences]
        _, _, saprot_df_batch_tokens = self.batch_converter(seq_df_list)
        saprot_df_tokens = {
            'tokens': saprot_df_batch_tokens,
            'mutation': mutation_df_list
        }
        print(f"Saving df saprot tokens")
        np.save(f"{self.save_path}/saprot_df_tokens.npy", saprot_df_tokens)
        new_sequences = self._foldseek_token_converter(prot_sequences)
        seq_list = [('id', prot_sequence) for prot_sequence in new_sequences]
        # convert the sequence to tokens every 10000 sequences
        for i in range(0, len(seq_list), 10000):
            # check if the file already exists
            if os.path.exists(f'{self.save_path}/saprot_tokens_{i}.npy'):
                continue
            _, _, saprot_batch_tokens = self.batch_converter(seq_list[i:i+10000])
            # save the tokens and mutation list into one file
            saprot_tokens = {
                'tokens': saprot_batch_tokens,
                'mutation': mutation_list[i:i+10000]
            }
            print(f"Saving saprot_tokens_{i}.npy")
            np.save(f'{self.save_path}/saprot_tokens_{i}.npy', saprot_tokens)
        # convert the last batch
        if i+10000 < len(seq_list):
            _, _, saprot_batch_tokens = self.batch_converter(seq_list[i+10000:])
            saprot_tokens = {
                'tokens': saprot_batch_tokens,
                'mutation': mutation_list[i+10000:]
            }
            print(f"Saving saprot_tokens_{i}.npy")
            np.save(f'{self.save_path}/saprot_tokens_{i}.npy', saprot_tokens)

    def _foldseek_token_converter(self, prot_sequence):
        new_prot_sequence = []
        for i in tqdm.tqdm(range(len(prot_sequence))):
            fs_token_list = []
            for aa, tdi in zip(prot_sequence[i], self.foldseek_wt_seq):
                fs_token_list.append(aa)
                fs_token_list.append(tdi)
            new_prot_sequence.append(''.join(fs_token_list))
        print(f"Number of new sequences after foldseek token converter: {len(new_prot_sequence)}")

        return new_prot_sequence

def load_vs_dataloader(args):
    vs_dataset = VSDataset(mutation_file_path=args.mutation_file_path,
                           seq_file_path=args.seq_file_path)
    vs_loader = DataLoader(vs_dataset,
                           batch_size=args.batch_size,
                           num_workers=args.num_workers)
    return vs_loader

def load_sp_vs_dataloader(args, batch_converter):
    vs_dataset = SPVSDataset(mutation_file_path=args.mutation_file_path,
                             seq_file_path=args.seq_file_path,
                             batch_converter=batch_converter)
    vs_loader = DataLoader(vs_dataset,
                           batch_size=args.batch_size,
                           num_workers=args.num_workers)
    return vs_loader

def save_sp_vs_dataloader(args, batch_converter, save_path, processed_path):
    vs_batch_convert = SPVSBatchConvert(mutation_file_path=args.mutation_file_path,
                                        seq_file_path=args.seq_file_path,
                                        batch_converter=batch_converter,
                                        save_path=save_path,
                                        processed_path=processed_path)
    vs_batch_convert.save_data()
    return None
