import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score, mean_absolute_error

from train_transformer import ClickstreamDataset
from transformer_model import SASRecTransformer

def extract_transformer_embeddings():
    data_dir = "data"
    events_path = os.path.join(data_dir, "clickstream_events.csv")
    users_path = os.path.join(data_dir, "clickstream_users.csv")
    
    # 1. Initialize dataset
    max_len = 50
    ds = ClickstreamDataset(events_path, max_len=max_len)
    # Loader without shuffling to keep aligned
    loader = DataLoader(ds, batch_size=32, shuffle=False)
    
    # 2. Load pre-trained model weights
    num_events = 9
    d_model = 32
    model = SASRecTransformer(num_events=num_events, d_model=d_model, nhead=2, num_layers=2, max_len=max_len-1)
    
    model_path = "sasrec_transformer.pt"
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path))
        print("Loaded pre-trained SASRec weights successfully.")
    else:
        print("[WARNING] Pre-trained weights not found. Using randomly initialized transformer.")
        
    model.eval()
    
    # 3. Generate representations
    embeddings_list = []
    user_ids_list = []
    
    with torch.no_grad():
        for batch in loader:
            inputs, _, lengths, user_ids = batch
            # Extract representation at last valid item
            # inputs shape is (batch, max_len-1)
            # lengths indicates sequence size of inputs
            embs = model.get_sequence_embedding(inputs, lengths)
            embeddings_list.append(embs.numpy())
            user_ids_list.append(user_ids.numpy())
            
    embeddings = np.concatenate(embeddings_list, axis=0)
    user_ids = np.concatenate(user_ids_list, axis=0)
    
    # Map user labels (LTV)
    ltv_df = pd.read_csv(users_path)
    ltv_map = dict(zip(ltv_df["user_id"], ltv_df["ltv"]))
    labels = np.array([ltv_map[uid] for uid in user_ids])
    
    # 4. Extract baseline aggregates
    raw_events_df = pd.read_csv(events_path)
    baselines = []
    for uid in user_ids:
        row = raw_events_df[raw_events_df["user_id"] == uid].iloc[0]
        seq = [int(x) for x in row["events"].split(",")]
        
        seq_len = len(seq)
        num_cards = sum(1 for e in seq if e == 4 or e == 5)
        num_payments = sum(1 for e in seq if e == 6)
        num_support = sum(1 for e in seq if e == 7)
        num_limits = sum(1 for e in seq if e == 8)
        
        baselines.append([seq_len, num_cards, num_payments, num_support, num_limits])
        
    baselines = np.array(baselines)
    
    return user_ids, baselines, embeddings, labels

def evaluate_ltv_forecasting():
    user_ids, baselines, embeddings, labels = extract_transformer_embeddings()
    
    # Train / Test split
    X_train_base, X_test_base, X_train_emb, X_test_emb, y_train, y_test = train_test_split(
        baselines, embeddings, labels, test_size=0.3, random_state=42
    )
    
    # Combined features
    X_train_comb = np.hstack([X_train_base, X_train_emb])
    X_test_comb = np.hstack([X_test_base, X_test_emb])
    
    print("\n" + "="*50)
    print("DOWNSTREAM EVALUATION: CUSTOMER LTV FORECASTING")
    print("="*50)
    
    # Regressor model: Random Forest
    def evaluate(name, X_tr, X_te):
        reg = RandomForestRegressor(n_estimators=100, random_state=42)
        reg.fit(X_tr, y_train)
        preds = reg.predict(X_te)
        r2 = r2_score(y_test, preds)
        mae = mean_absolute_error(y_test, preds)
        print(f"{name:<45} | R2: {r2:.4f} | MAE: ${mae:.2f}")
        return r2, mae
        
    r2_base, mae_base = evaluate("Baseline Features only", X_train_base, X_test_base)
    r2_emb, mae_emb = evaluate("Transformer embeddings only", X_train_emb, X_test_emb)
    r2_comb, mae_comb = evaluate("Combined Features (Base + Transformer)", X_train_comb, X_test_comb)
    print("="*50)
    
    # Save summary
    with open("ltv_evaluation_summary.txt", "w", encoding="utf-8") as f:
        f.write("Downstream Task: Customer LTV Forecasting\n")
        f.write(f"Baseline R2: {r2_base:.4f}, MAE: ${mae_base:.2f}\n")
        f.write(f"Transformer Embedding R2: {r2_emb:.4f}, MAE: ${mae_emb:.2f}\n")
        f.write(f"Combined Feature R2: {r2_comb:.4f}, MAE: ${mae_comb:.2f}\n")
        
    print("LTV evaluation summary saved to ltv_evaluation_summary.txt")

if __name__ == "__main__":
    evaluate_ltv_forecasting()
