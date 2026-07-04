import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

class CoLESDataset(Dataset):
    """
    Dataset for Contrastive Learning of Event Sequences (CoLES).
    For each user, it samples two disjoint sub-sequences from their transaction history,
    which serve as positive pairs (different views of the same user's behavior).
    """
    def __init__(self, transactions_path, seq_len=15):
        self.seq_len = seq_len
        df = pd.read_csv(transactions_path)
        
        # Parse timestamps and sort
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.sort_values(by=["user_id", "timestamp"])
        
        # Preprocess features:
        # 1. Map MCCs to indices starting from 1 (0 is reserved for padding)
        self.mcc_list = sorted(df["mcc"].unique())
        self.mcc_to_idx = {mcc: idx + 1 for idx, mcc in enumerate(self.mcc_list)}
        df["mcc_idx"] = df["mcc"].map(self.mcc_to_idx)
        
        # 2. Scale amounts using log scaling
        df["log_amount"] = np.log1p(df["amount"])
        self.amount_mean = df["log_amount"].mean()
        self.amount_std = df["log_amount"].std()
        df["scaled_amount"] = (df["log_amount"] - self.amount_mean) / (self.amount_std + 1e-8)
        
        # Group transactions by user
        self.user_sequences = {}
        for user_id, group in df.groupby("user_id"):
            mccs = group["mcc_idx"].values
            amounts = group["scaled_amount"].values
            self.user_sequences[user_id] = (mccs, amounts)
            
        self.user_ids = list(self.user_sequences.keys())

    def __len__(self):
        return len(self.user_ids)

    def _get_subsequence(self, mccs, amounts, start_idx):
        """Extracts and pads a subsequence of length seq_len."""
        end_idx = min(start_idx + self.seq_len, len(mccs))
        actual_len = end_idx - start_idx
        
        sub_mccs = np.zeros(self.seq_len, dtype=np.int64)
        sub_amounts = np.zeros(self.seq_len, dtype=np.float32)
        
        sub_mccs[:actual_len] = mccs[start_idx:end_idx]
        sub_amounts[:actual_len] = amounts[start_idx:end_idx]
        
        return sub_mccs, sub_amounts, actual_len

    def __getitem__(self, idx):
        user_id = self.user_ids[idx]
        mccs, amounts = self.user_sequences[user_id]
        
        n_tx = len(mccs)
        
        # We need to sample two non-overlapping sub-sequences
        # If the sequence is short, we can split it in the middle or allow slight overlap if necessary.
        if n_tx >= 2 * self.seq_len:
            # Safe non-overlapping split
            mid = n_tx // 2
            start1 = np.random.randint(0, mid - self.seq_len + 1)
            start2 = np.random.randint(mid, n_tx - self.seq_len + 1)
        else:
            # Overlapping or padded split
            start1 = 0
            start2 = max(0, n_tx - self.seq_len)
            if start1 == start2 and n_tx > self.seq_len:
                start2 = np.random.randint(1, n_tx - self.seq_len + 1)
                
        mccs1, amounts1, len1 = self._get_subsequence(mccs, amounts, start1)
        mccs2, amounts2, len2 = self._get_subsequence(mccs, amounts, start2)
        
        return (
            torch.tensor(mccs1), torch.tensor(amounts1), torch.tensor(len1),
            torch.tensor(mccs2), torch.tensor(amounts2), torch.tensor(len2)
        )


class DownstreamDataset(Dataset):
    """
    Dataset for downstream supervised classification (e.g. Credit Default Prediction).
    Returns the full transaction history sequence of a user, along with a target label.
    """
    def __init__(self, transactions_path, users_path, coles_dataset=None, max_seq_len=40):
        self.max_seq_len = max_seq_len
        
        # If a pre-configured coles_dataset is passed, reuse its scaling params and MCC mapping
        if coles_dataset is not None:
            self.mcc_to_idx = coles_dataset.mcc_to_idx
            self.amount_mean = coles_dataset.amount_mean
            self.amount_std = coles_dataset.amount_std
        else:
            # Otherwise compute them from scratch
            df_tx = pd.read_csv(transactions_path)
            self.mcc_list = sorted(df_tx["mcc"].unique())
            self.mcc_to_idx = {mcc: idx + 1 for idx, mcc in enumerate(self.mcc_list)}
            df_tx["log_amount"] = np.log1p(df_tx["amount"])
            self.amount_mean = df_tx["log_amount"].mean()
            self.amount_std = df_tx["log_amount"].std()
            
        # Load datasets
        tx_df = pd.read_csv(transactions_path)
        user_df = pd.read_csv(users_path)
        
        tx_df["timestamp"] = pd.to_datetime(tx_df["timestamp"])
        tx_df = tx_df.sort_values(by=["user_id", "timestamp"])
        tx_df["mcc_idx"] = tx_df["mcc"].map(self.mcc_to_idx).fillna(0).astype(np.int64)
        tx_df["log_amount"] = np.log1p(tx_df["amount"])
        tx_df["scaled_amount"] = (tx_df["log_amount"] - self.amount_mean) / (self.amount_std + 1e-8)
        
        # Group transactions
        self.user_sequences = {}
        for user_id, group in tx_df.groupby("user_id"):
            mccs = group["mcc_idx"].values
            amounts = group["scaled_amount"].values
            self.user_sequences[user_id] = (mccs, amounts)
            
        # Map user default labels
        self.labels = dict(zip(user_df["user_id"], user_df["credit_default"]))
        self.user_ids = user_df["user_id"].values

    def __len__(self):
        return len(self.user_ids)

    def __getitem__(self, idx):
        user_id = self.user_ids[idx]
        label = self.labels[user_id]
        
        mccs, amounts = self.user_sequences.get(user_id, ([], []))
        n_tx = len(mccs)
        
        # Truncate to max_seq_len from the END of history (most recent transactions)
        if n_tx > self.max_seq_len:
            mccs = mccs[-self.max_seq_len:]
            amounts = amounts[-self.max_seq_len:]
            actual_len = self.max_seq_len
        else:
            actual_len = n_tx
            
        # Pad sequence
        pad_mccs = np.zeros(self.max_seq_len, dtype=np.int64)
        pad_amounts = np.zeros(self.max_seq_len, dtype=np.float32)
        
        if actual_len > 0:
            pad_mccs[:actual_len] = mccs
            pad_amounts[:actual_len] = amounts
            
        return (
            torch.tensor(pad_mccs),
            torch.tensor(pad_amounts),
            torch.tensor(actual_len),
            torch.tensor(label, dtype=torch.float32),
            torch.tensor(user_id, dtype=torch.long)
        )
