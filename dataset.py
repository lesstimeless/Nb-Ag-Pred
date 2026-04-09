from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
import random
import ast

torch.manual_seed(42)
random.seed(42)

BASE_DIR = Path(__file__).resolve().parent
excel_path = BASE_DIR / "Data/Processed/processed_sequences_expanded.xlsx"

token_vocab = {
    'PAD': 0,
    '[CLS]': 1,
    '[SEP]': 2,
    '[Mask]': 3,
    'A': 4,
    'R': 5,
    'N': 6,
    'D': 7,
    'C': 8,
    'E': 9,
    'Q': 10,
    'G': 11,
    'H': 12,
    'I': 13,
    'L': 14,
    'K': 15,
    'M': 16,
    'F': 17,
    'P': 18,
    'S': 19,
    'T': 20,
    'W': 21,
    'Y': 22,
    'V': 23,
}

# 9 properties: Kyle-Doolittle hydrophobicity (scalar), mw (scalar; kDa), flexibility (scalar), side chain types (one-hot)
## 6 side chain types: polar, non-polar, acidic, basic, aromatic, neutral
## Flexibility scale from Huang, F. et. al. (2003), except Cys, Met, Tyr, Trp
### Cys: usually forms disulfide bonds, so least flexible, set to 0
### Met: 3C and 1S --> larger than Glu, smaller than Phe --> use fitted value of 6.75
### Tyr: larger than Phe --> use fitted value of 5.7
### Trp: largest side chain, but more flexibe than Pro and Cys --> use fitted value of 0.15
## hydrophobicity values are standardized to preserve the sign
## MW and flexibility are normalized to [0,1] based on min-max normalization
physicochem_vocab = {
    '[CLS]': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    '[SEP]': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    '[Mask]': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],

    'A': [0.78, 0.10, 0.46, 0, 1, 0, 0, 0, 1],
    'R': [-1.37, 0.76, 0.11, 1, 0, 0, 1, 0, 0],
    'N': [-1.03, 0.44, 0.51, 1, 0, 0, 0, 0, 1],
    'D': [-1.03, 0.44, 0.53, 1, 0, 1, 0, 0, 0],
    'C': [1.027, 0.35, 0.00, 1, 0, 0, 0, 0, 1],
    'E': [-1.03, 0.55, 0.22, 1, 0, 1, 0, 0, 0],
    'Q': [-1.03, 0.55, 0.18, 1, 0, 0, 0, 0, 1],
    'G': [0.03, 0.00, 1.00, 0, 1, 0, 0, 0, 1],
    'H': [-0.93, 0.62, 0.12, 1, 0, 0, 1, 0, 0],
    'I': [1.71, 0.43, 0.05, 0, 1, 0, 0, 0, 1],
    'L': [1.47, 0.43, 0.25, 0, 1, 0, 0, 0, 1],
    'K': [-1.17, 0.55, 0.10, 1, 0, 0, 1, 0, 0],
    'M': [0.82, 0.57, 0.17, 0, 1, 0, 0, 0, 1],
    'F': [1.13, 0.69, 0.19, 0, 0, 0, 0, 1, 1],
    'P': [-0.38, 0.31, 0.0023, 0, 1, 0, 0, 0, 1],
    'S': [-0.10, 0.23, 0.64, 1, 0, 0, 0, 0, 1],
    'T': [-0.07, 0.34, 0.28, 1, 0, 0, 0, 0, 1],
    'W': [-0.14, 1.00, 0.0038, 0, 1, 0, 0, 1, 1],
    'Y': [-0.27, 0.82, 0.14, 1, 0, 0, 0, 1, 1],
    'V': [1.61, 0.32, 0.07, 0, 1, 0, 0, 0, 1]
}


def load_data(excel_path, n_ensembles=3):
    pos_df = pd.read_excel(excel_path, sheet_name="Positive Samples")
    neg_df = pd.read_excel(excel_path, sheet_name="Negative Samples")

    train_pos, test_pos = train_test_split(pos_df, test_size=0.2, random_state=42)
    train_pos, finetune_pos = train_test_split(train_pos, test_size=0.25, random_state=42)  # 0.2 of original for finetuning
    train_neg, test_neg = train_test_split(neg_df, test_size=0.2, random_state=42)
    train_neg, finetune_neg = train_test_split(train_neg, test_size=0.25, random_state=42) # 0.2 of original for finetuning

    # We now samples 10 times from positive set only to address class imbalance, and shuffle the resulting dataframes
    len_train_neg = len(train_neg)
    len_finetune_neg = len(finetune_neg)
    # print(f"Number of training samples: {len(train_neg)} negative")
    # print(f"Number of finetuning samples: {len(finetune_neg)} negative")

    train_dfs = []
    finetune_dfs = []

    # For each ensemble, we sample a different subset of the positive samples (without replacement) to combine with all negative samples, and shuffle the resulting dataframe
    for i in range(n_ensembles):
        sampled_train_pos = train_pos.sample(n=3*len_train_neg, replace=False, random_state=42+i)  # Different random state for each ensemble to get different samples
        sampled_finetune_pos = finetune_pos.sample(n=3*len_finetune_neg, replace=False, random_state=42+i)

        train_df_i = pd.concat([sampled_train_pos, train_neg], axis=0)\
            .sample(frac=1, random_state=42+i).reset_index(drop=True)
        finetune_df_i = pd.concat([sampled_finetune_pos, finetune_neg], axis=0)\
            .sample(frac=1, random_state=42+i).reset_index(drop=True)
        train_dfs.append(train_df_i)
        finetune_dfs.append(finetune_df_i)
    # For test set, we use all positive and negative samples, but shuffle them
    # train_df = pd.concat([train_pos, train_neg], axis=0).sample(frac=1, random_state=42).reset_index(drop=True)
    # finetune_df = pd.concat([finetune_pos, finetune_neg], axis=0).sample(frac=1, random_state=42).reset_index(drop=True)
    test_df = pd.concat([test_pos, test_neg], axis=0).sample(frac=1, random_state=42).reset_index(drop=True)
    
    return train_df, finetune_df, test_df

class NbAgDataset():
    def __init__(self, df, mlm=True):
        """
        df: train_df/finetune_df/test_df
        mlm: True for pretraining, False for finetuning/testing (since we only need the labels for finetuning)
        """
        self.df = df
        self.mlm = mlm


    def __len__(self):
        return len(self.df)

    def max_token_length(self):
        max_length = (self.df['Nb_sequence'].str.len() + 
              self.df['Ag_sequence'].str.len() + 3).max()
        return max_length

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        Nb_sequence = list(row['Nb_sequence'])
        Ag_sequence = list(row['Ag_sequence'])
        cdr_indices = ast.literal_eval(row['CDR_indices'])  # list of indices for CDR residues in Nb sequence (before adding special tokens)
        four_fr_indices = ast.literal_eval(row['four_FR_indices']) # list of indicies for 4 FR residues involved in binding in Nb sequence (before adding special tokens)
        rest_fr_indices = ast.literal_eval(row['rest_FR_indices']) # list of indices for the rest of the FR residues in Nb sequence (before adding special tokens)

        # Full sequence with special tokens
        full_seq = ['[CLS]'] + Nb_sequence + ['[SEP]'] + Ag_sequence + ['[SEP]']
        seq_len = len(full_seq)

        # Token IDs
        token_ids = torch.tensor([token_vocab[aa] for aa in full_seq], dtype=torch.long)

        # Physicochemical properties
        physioco_ids = torch.tensor([physicochem_vocab[aa] for aa in full_seq], dtype=torch.float)

        # Segment IDs (0 for Nb, 1 for Ag)
        sep_idx = len(Nb_sequence) + 1  # +1 for [CLS]
        segment_ids = torch.tensor([0]*(sep_idx+1) + [1]*(seq_len-sep_idx-1), dtype=torch.long)

        # mlm labels (only for pretraining)
        mlm_labels = torch.full_like(token_ids, -100)  # Initialize all labels to -100 (ignore index)
        if self.mlm:
            # for Nb: Randomly mask 40% CDR idx, 25% of 4 FR residues, 6% rest of FR residues (excluding special tokens)
            num_cdr_masked = int(0.4 * len(cdr_indices))
            num_four_res_masked = int(0.25 * len(four_fr_indices))
            num_rest_fr_masked = int(0.06 * len(rest_fr_indices))

            # Mask CDR indices
            cdr_mask = random.sample(cdr_indices, num_cdr_masked)
            for idx in cdr_mask:
                rand = random.random()
                mlm_labels[idx+1] = token_ids[idx+1]  # +1 to account for [CLS]
                if rand < 0.8:
                    token_ids[idx+1] = token_vocab['[Mask]']
                elif rand < 0.9:
                    token_ids[idx+1] = random.randint(4, 23)  # Random amino acid token ID

            # Mask 4 FR indices
            four_fr_mask = random.sample(four_fr_indices, num_four_res_masked)
            for idx in four_fr_mask:
                rand = random.random()
                mlm_labels[idx+1] = token_ids[idx+1]
                if rand < 0.8:
                    token_ids[idx+1] = token_vocab['[Mask]']
                elif rand < 0.9:
                    token_ids[idx+1] = random.randint(4, 23)  # Random amino acid token ID
            
            # Mask rest of FR indices
            rest_fr_mask = random.sample(rest_fr_indices, num_rest_fr_masked)
            for idx in rest_fr_mask:
                rand = random.random()
                mlm_labels[idx+1] = token_ids[idx+1]
                if rand < 0.8:
                    token_ids[idx+1] = token_vocab['[Mask]']
                elif rand < 0.9:
                    token_ids[idx+1] = random.randint(4, 23)  # Random amino acid token ID
            
            # Mask 15% of Ag residues (excluding special tokens)
            ag_indices = list(range(sep_idx+1, seq_len-1))
            num_ag_masked = int(0.15 * len(ag_indices))
            ag_mask = random.sample(ag_indices, num_ag_masked)
            for idx in ag_mask:
                rand = random.random()
                mlm_labels[idx] = token_ids[idx]
                if rand < 0.8:
                    token_ids[idx] = token_vocab['[Mask]']
                elif rand < 0.9:
                    token_ids[idx] = random.randint(4, 23)  # Random amino acid token ID
            
            return {
                "token_ids": token_ids,
                "physioco_ids": physioco_ids,
                "segment_ids": segment_ids,
                "mlm_labels": mlm_labels,
                "seq_len": seq_len
            }
        return {
            "token_ids": token_ids,
            "physioco_ids": physioco_ids,
            "segment_ids": segment_ids,
            "seq_len": seq_len,
            "label": 0 if len(row['Nb_binding_idx']) < 3 else 1  # binary classification label for finetuning/testing
        }

# for testing
if __name__ == "__main__":
    train_dfs, finetune_dfs, test_df = load_data(excel_path)
    print(train_dfs[0].shape, finetune_dfs[0].shape, test_df.shape)
    # dataset = NbAgDataset(test_df)
    # print(dataset.max_token_length())
    # dataset = NbAgDataset(train_dfs[0], mlm=True)
    # sample = dataset[0]
    # print(sample)
