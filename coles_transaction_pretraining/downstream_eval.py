import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

from dataset import DownstreamDataset, CoLESDataset
from model import TransactionEncoder

def extract_features_and_embeddings():
    data_dir = "data"
    tx_path = os.path.join(data_dir, "transactions.csv")
    users_path = os.path.join(data_dir, "users.csv")
    
    # 1. Initialize dataset
    # We create a dummy CoLESDataset first to get the correct MCC mapping and scaling parameters
    coles_ds = CoLESDataset(tx_path)
    num_mccs = len(coles_ds.mcc_list)
    
    ds = DownstreamDataset(tx_path, users_path, coles_dataset=coles_ds)
    loader = DataLoader(ds, batch_size=32, shuffle=False)
    
    # 2. Initialize encoder and load pre-trained weights
    encoder = TransactionEncoder(num_mccs=num_mccs, embedding_dim=64)
    model_path = "coles_encoder.pt"
    if os.path.exists(model_path):
        encoder.load_state_dict(torch.load(model_path))
        print("Loaded pre-trained weights successfully.")
    else:
        print("[WARNING] Pre-trained weights not found. Using randomly initialized encoder.")
        
    encoder.eval()
    
    # 3. Generate representations
    embeddings_list = []
    labels_list = []
    user_ids_list = []
    
    # For baseline feature engineering, we will also load raw transactions
    raw_tx = pd.read_csv(tx_path)
    
    with torch.no_grad():
        for batch in loader:
            mccs, amounts, lengths, labels, user_ids = batch
            # Get pre-trained embeddings
            embs = encoder(mccs, amounts, lengths)
            
            embeddings_list.append(embs.numpy())
            labels_list.append(labels.numpy())
            user_ids_list.append(user_ids.numpy())
            
    embeddings = np.concatenate(embeddings_list, axis=0)
    labels = np.concatenate(labels_list, axis=0)
    user_ids = np.concatenate(user_ids_list, axis=0)
    
    # 4. Engineer Baseline aggregates for comparison
    baselines = []
    for uid in user_ids:
        user_tx = raw_tx[raw_tx["user_id"] == uid]
        tx_count = len(user_tx)
        avg_amt = user_tx["amount"].mean() if tx_count > 0 else 0
        max_amt = user_tx["amount"].max() if tx_count > 0 else 0
        std_amt = user_tx["amount"].std() if tx_count > 1 else 0
        cash_ratio = sum(1 for m in user_tx["mcc"] if m == 6011) / tx_count if tx_count > 0 else 0
        elect_ratio = sum(1 for m in user_tx["mcc"] if m == 5732) / tx_count if tx_count > 0 else 0
        
        baselines.append([tx_count, avg_amt, max_amt, std_amt, cash_ratio, elect_ratio])
        
    baselines = np.array(baselines)
    
    return user_ids, baselines, embeddings, labels

def evaluate_downstream():
    user_ids, baselines, embeddings, labels = extract_features_and_embeddings()
    
    # Train / Test split
    X_train_base, X_test_base, X_train_emb, X_test_emb, y_train, y_test = train_test_split(
        baselines, embeddings, labels, test_size=0.3, random_state=42
    )
    
    # Combined features
    X_train_comb = np.hstack([X_train_base, X_train_emb])
    X_test_comb = np.hstack([X_test_base, X_test_emb])
    
    print("\n" + "="*50)
    print("DOWNSTREAM EVALUATION: CREDIT DEFAULT PREDICTION")
    print("="*50)
    
    # 1. Model on baseline features only
    clf_base = RandomForestClassifier(n_estimators=100, random_state=42)
    clf_base.fit(X_train_base, y_train)
    preds_base = clf_base.predict_proba(X_test_base)[:, 1]
    auc_base = roc_auc_score(y_test, preds_base)
    print(f"ROC-AUC with Baseline Hand-Crafted Features:     {auc_base:.4f}")
    
    # 2. Model on CoLES embeddings only
    clf_emb = RandomForestClassifier(n_estimators=100, random_state=42)
    clf_emb.fit(X_train_emb, y_train)
    preds_emb = clf_emb.predict_proba(X_test_emb)[:, 1]
    auc_emb = roc_auc_score(y_test, preds_emb)
    print(f"ROC-AUC with CoLES Self-Supervised Embeddings:  {auc_emb:.4f}")
    
    # 3. Model on Combined features
    clf_comb = RandomForestClassifier(n_estimators=100, random_state=42)
    clf_comb.fit(X_train_comb, y_train)
    preds_comb = clf_comb.predict_proba(X_test_comb)[:, 1]
    auc_comb = roc_auc_score(y_test, preds_comb)
    print(f"ROC-AUC with Combined Features (Base + CoLES):  {auc_comb:.4f}")
    print("="*50)
    
    # Let's write the evaluation results to a summary file as well
    with open("evaluation_summary.txt", "w", encoding="utf-8") as f:
        f.write("Downstream Task: Credit Default Prediction\n")
        f.write(f"Baseline AUC: {auc_base:.4f}\n")
        f.write(f"CoLES Embedding AUC: {auc_emb:.4f}\n")
        f.write(f"Combined Feature AUC: {auc_comb:.4f}\n")
        
    print("Evaluation summary saved to evaluation_summary.txt")

if __name__ == "__main__":
    evaluate_downstream()
