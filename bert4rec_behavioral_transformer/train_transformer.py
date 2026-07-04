import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
import mlflow

from event_generator import generate_behavioral_data
from transformer_model import SASRecTransformer

class ClickstreamDataset(Dataset):
    """
    Dataset for sequential clickstream logs.
    Prepares inputs [e_1, ..., e_t-1] and targets [e_2, ..., e_t] for causal autoregressive training.
    """
    def __init__(self, csv_path, max_len=50):
        self.max_len = max_len
        df = pd.read_csv(csv_path)
        
        self.sequences = []
        self.lengths = []
        self.user_ids = []
        
        for _, row in df.iterrows():
            seq = [int(x) for x in row["events"].split(",")]
            self.user_ids.append(row["user_id"])
            self.lengths.append(min(len(seq), max_len))
            
            # Pad / truncate
            if len(seq) > max_len:
                seq = seq[:max_len]
            else:
                seq = seq + [0] * (max_len - len(seq))
                
            self.sequences.append(seq)
            
        self.sequences = np.array(self.sequences)
        self.lengths = np.array(self.lengths)
        self.user_ids = np.array(self.user_ids)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = self.sequences[idx]
        length = self.lengths[idx]
        user_id = self.user_ids[idx]
        
        # In autoregressive training:
        # Input seq is e_1 to e_{T-1}
        # Target seq is e_2 to e_T
        inputs = seq[:-1]
        targets = seq[1:]
        
        return (
            torch.tensor(inputs, dtype=torch.long),
            torch.tensor(targets, dtype=torch.long),
            torch.tensor(length - 1, dtype=torch.long), # Adjust length for input sequence length
            torch.tensor(user_id, dtype=torch.long)
        )

def train_transformer():
    data_dir = "data"
    events_path = os.path.join(data_dir, "clickstream_events.csv")
    if not os.path.exists(events_path):
        print("Data files not found. Generating behavioral sequence data...")
        generate_behavioral_data(num_users=300, output_dir=data_dir)

    # Hyperparameters
    epochs = 15
    batch_size = 32
    lr = 0.002
    max_len = 50
    d_model = 32
    nhead = 2
    num_layers = 2
    num_events = 9 # Events range from 1 to 9 (0 is padding)

    dataset = ClickstreamDataset(events_path, max_len=max_len)
    
    # Split into train/validation
    train_size = int(0.85 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    model = SASRecTransformer(num_events=num_events, d_model=d_model, nhead=nhead, num_layers=num_layers, max_len=max_len-1)
    
    # Use cross-entropy loss, ignoring the padding index 0 in the targets
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    mlflow.set_experiment("Clickstream_Transformer")
    
    with mlflow.start_run():
        mlflow.log_params({
            "epochs": epochs,
            "batch_size": batch_size,
            "lr": lr,
            "max_len": max_len,
            "d_model": d_model,
            "nhead": nhead,
            "num_layers": num_layers
        })
        
        print("Starting Autoregressive Transformer Training...")
        for epoch in range(epochs):
            model.train()
            train_loss = 0.0
            
            for batch in train_loader:
                inputs, targets, lengths, _ = batch
                
                optimizer.zero_grad()
                
                logits, _ = model(inputs) # (batch_size, seq_len, num_events+1)
                
                # Reshape for CrossEntropyLoss
                # logits: (batch_size * seq_len, num_events+1)
                # targets: (batch_size * seq_len)
                loss = criterion(logits.view(-1, num_events + 1), targets.view(-1))
                
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item() * inputs.size(0)
                
            train_loss /= len(train_dataset)
            
            # Validation loop
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for batch in val_loader:
                    inputs, targets, lengths, _ = batch
                    logits, _ = model(inputs)
                    loss = criterion(logits.view(-1, num_events + 1), targets.view(-1))
                    val_loss += loss.item() * inputs.size(0)
            val_loss /= len(val_dataset)
            
            print(f"Epoch {epoch+1:02d}/{epochs:02d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
            mlflow.log_metrics({"train_loss": train_loss, "val_loss": val_loss}, step=epoch)

        # Save model weights
        model_save_path = "sasrec_transformer.pt"
        torch.save(model.state_dict(), model_save_path)
        print(f"Transformer weights saved to {model_save_path}")
        mlflow.log_artifact(model_save_path)

if __name__ == "__main__":
    train_transformer()
