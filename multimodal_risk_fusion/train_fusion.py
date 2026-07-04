import os
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.metrics import roc_auc_score, accuracy_score
import mlflow

from multimodal_model import MultimodalRiskFusionModel

class MultimodalDataset(Dataset):
    """
    Dataset for Multimodal Risk Fusion.
    Returns:
    1. Sequential inputs: MCC and scaled Amount sequences.
    2. Tabular inputs: static/device features.
    3. Graph inputs: sequence of merchant IDs (to look up graph embeddings).
    4. Label: account-level fraud/risk flag.
    """
    def __init__(self, num_users=300, seq_len=20):
        self.seq_len = seq_len
        random.seed(42)
        np.random.seed(42)
        
        self.mcc_list = [5411, 5812, 5541, 5311, 5732, 5912, 4814, 5814, 6011, 7997]
        self.mcc_to_idx = {mcc: idx + 1 for idx, mcc in enumerate(self.mcc_list)}
        
        self.num_merchants = 50 # 50 distinct merchants in user-merchant graph
        
        self.data = []
        
        for user_id in range(num_users):
            # Latent fraud risk factor (0 to 1)
            latent_fraud_risk = np.random.beta(1, 8)
            
            # 1. Generate Tabular Device Metadata
            vpn_used = 1 if (latent_fraud_risk > 0.4 and random.random() < 0.7) or random.random() < 0.05 else 0
            is_mobile = 1 if random.random() < 0.8 else 0
            os_code = random.choice([0, 1]) if is_mobile else random.choice([2, 3])
            ip_changes = random.randint(3, 12) if latent_fraud_risk > 0.4 else random.randint(0, 2)
            num_failed_logins = random.randint(2, 6) if latent_fraud_risk > 0.5 else random.randint(0, 1)
            time_since_login = np.random.exponential(scale=2.0) if latent_fraud_risk > 0.4 else np.random.exponential(scale=24.0)
            
            tabular_feats = np.array([
                float(vpn_used), 
                float(is_mobile), 
                float(os_code), 
                float(ip_changes) / 10.0, 
                float(num_failed_logins) / 5.0,
                np.clip(time_since_login / 100.0, 0.0, 1.0)
            ], dtype=np.float32)

            # 2. Generate Sequential Features (MCC and Amount)
            actual_len = random.randint(10, seq_len)
            
            mccs = []
            amounts = []
            merchants = []
            
            for _ in range(actual_len):
                if latent_fraud_risk > 0.4 and random.random() < 0.5:
                    mcc = 6011 if random.random() < 0.6 else 5732 # high-risk cash outs
                    amount = np.random.lognormal(mean=np.log(200.0), sigma=0.5)
                    merchant = random.randint(40, 50) # suspicious merchants (nodes 40-50 in graph)
                else:
                    mcc = random.choice(self.mcc_list)
                    amount = np.random.lognormal(mean=np.log(30.0), sigma=0.5)
                    merchant = random.randint(1, 39) # regular merchants (nodes 1-39 in graph)
                    
                mccs.append(self.mcc_to_idx[mcc])
                amounts.append((np.log1p(amount) - 3.5) / 1.2) # rough manual scaling
                merchants.append(merchant)
                
            # Pad sequences
            pad_mccs = np.zeros(seq_len, dtype=np.int64)
            pad_amounts = np.zeros(seq_len, dtype=np.float32)
            pad_merchants = np.zeros(seq_len, dtype=np.int64)
            
            pad_mccs[:actual_len] = mccs
            pad_amounts[:actual_len] = amounts
            pad_merchants[:actual_len] = merchants
            
            # 3. Target Label (Account Fraud Compromise)
            # Probability driven by high risk device, high ip changes, failed logins, and suspicious merchants
            risk_score = (latent_fraud_risk * 0.4 + 
                          vpn_used * 0.2 + 
                          (ip_changes / 12.0) * 0.2 + 
                          (num_failed_logins / 6.0) * 0.2)
            fraud_label = 1 if random.random() < np.clip(risk_score, 0.01, 0.99) else 0
            
            self.data.append({
                "mccs": torch.tensor(pad_mccs),
                "amounts": torch.tensor(pad_amounts),
                "length": torch.tensor(actual_len),
                "tabular": torch.tensor(tabular_feats),
                "merchants": torch.tensor(pad_merchants),
                "label": torch.tensor(fraud_label, dtype=torch.float32)
            })

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        return (
            item["mccs"],
            item["amounts"],
            item["length"],
            item["tabular"],
            item["merchants"],
            item["label"]
        )

def train_multimodal_fusion():
    epochs = 15
    batch_size = 32
    lr = 0.002
    seq_len = 20

    dataset = MultimodalDataset(num_users=400, seq_len=seq_len)
    
    # Split
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    num_mccs = len(dataset.mcc_list)
    num_merchants = dataset.num_merchants

    model = MultimodalRiskFusionModel(
        num_mccs=num_mccs,
        num_merchants=num_merchants,
        tabular_in_dim=6,
        hidden_seq=32,
        hidden_tab=16,
        hidden_fusion=32
    )

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    mlflow.set_experiment("Multimodal_Risk_Fusion")
    
    with mlflow.start_run():
        mlflow.log_params({
            "epochs": epochs,
            "batch_size": batch_size,
            "lr": lr,
            "seq_len": seq_len,
            "num_mccs": num_mccs,
            "num_merchants": num_merchants
        })
        
        print("Starting Multimodal Risk Fusion Training...")
        for epoch in range(epochs):
            model.train()
            train_loss = 0.0
            
            for batch in train_loader:
                mccs, amounts, lengths, tabulars, merchants, labels = batch
                
                optimizer.zero_grad()
                
                logits = model(mccs, amounts, lengths, tabulars, merchants).squeeze(-1)
                loss = criterion(logits, labels)
                
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item() * mccs.size(0)
                
            train_loss /= len(train_dataset)
            
            # Validation
            model.eval()
            val_loss = 0.0
            all_preds = []
            all_labels = []
            
            with torch.no_grad():
                for batch in val_loader:
                    mccs, amounts, lengths, tabulars, merchants, labels = batch
                    logits = model(mccs, amounts, lengths, tabulars, merchants).squeeze(-1)
                    loss = criterion(logits, labels)
                    val_loss += loss.item() * mccs.size(0)
                    
                    preds = torch.sigmoid(logits).numpy()
                    all_preds.extend(preds)
                    all_labels.extend(labels.numpy())
            
            val_loss /= len(val_dataset)
            
            # Metrics
            val_auc = roc_auc_score(all_labels, all_preds)
            val_acc = accuracy_score(all_labels, [1 if p > 0.5 else 0 for p in all_preds])
            
            print(f"Epoch {epoch+1:02d}/{epochs:02d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val AUC: {val_auc:.4f} | Val Acc: {val_acc:.2%}")
            mlflow.log_metrics({"train_loss": train_loss, "val_loss": val_loss, "val_auc": val_auc, "val_acc": val_acc}, step=epoch)

        # Save model
        model_save_path = "multimodal_fusion.pt"
        torch.save(model.state_dict(), model_save_path)
        print(f"Multimodal Fusion model saved to {model_save_path}")
        mlflow.log_artifact(model_save_path)

if __name__ == "__main__":
    train_multimodal_fusion()
