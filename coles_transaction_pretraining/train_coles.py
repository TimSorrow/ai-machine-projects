import os
import torch
from torch.utils.data import DataLoader, random_split
import mlflow

from data_generator import generate_synthetic_data
from dataset import CoLESDataset
from model import TransactionEncoder, CoLESModel, contrastive_loss

def train_coles():
    # 1. Ensure synthetic data exists
    data_dir = "data"
    tx_path = os.path.join(data_dir, "transactions.csv")
    if not os.path.exists(tx_path):
        print("Data files not found. Generating synthetic data...")
        generate_synthetic_data(num_users=300, output_dir=data_dir)

    # 2. Setup hyperparameters
    epochs = 15
    batch_size = 32
    lr = 0.001
    seq_len = 15
    embedding_dim = 64
    projection_dim = 32
    temperature = 0.15

    # 3. Load dataset
    dataset = CoLESDataset(tx_path, seq_len=seq_len)
    num_mccs = len(dataset.mcc_list)
    print(f"Dataset loaded. Number of distinct MCCs: {num_mccs}")

    # Split into train/validation
    train_size = int(0.85 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # 4. Instantiate models
    encoder = TransactionEncoder(num_mccs=num_mccs, embedding_dim=embedding_dim)
    model = CoLESModel(encoder, embedding_dim=embedding_dim, projection_dim=projection_dim)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # 5. MLflow Logging Setup
    mlflow.set_experiment("Sequence_CoLES_Pretraining")
    
    with mlflow.start_run():
        mlflow.log_params({
            "epochs": epochs,
            "batch_size": batch_size,
            "lr": lr,
            "seq_len": seq_len,
            "embedding_dim": embedding_dim,
            "projection_dim": projection_dim,
            "temperature": temperature
        })
        
        print("Starting CoLES Self-Supervised Pre-training...")
        for epoch in range(epochs):
            model.train()
            train_loss = 0.0
            
            for batch in train_loader:
                mccs1, amounts1, lens1, mccs2, amounts2, lens2 = batch
                
                optimizer.zero_grad()
                
                # Encode and project both sequence views
                z1 = model(mccs1, amounts1, lens1)
                z2 = model(mccs2, amounts2, lens2)
                
                # Compute contrastive loss
                loss = contrastive_loss(z1, z2, temperature=temperature)
                
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item() * mccs1.size(0)
                
            train_loss /= len(train_dataset)
            
            # Validation loop
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for batch in val_loader:
                    mccs1, amounts1, lens1, mccs2, amounts2, lens2 = batch
                    z1 = model(mccs1, amounts1, lens1)
                    z2 = model(mccs2, amounts2, lens2)
                    loss = contrastive_loss(z1, z2, temperature=temperature)
                    val_loss += loss.item() * mccs1.size(0)
            val_loss /= len(val_dataset)
            
            print(f"Epoch {epoch+1:02d}/{epochs:02d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
            mlflow.log_metrics({"train_loss": train_loss, "val_loss": val_loss}, step=epoch)

        # 6. Save model weight state
        model_save_path = "coles_encoder.pt"
        torch.save(encoder.state_dict(), model_save_path)
        print(f"Pre-trained encoder weights saved to {model_save_path}")
        mlflow.log_artifact(model_save_path)

if __name__ == "__main__":
    train_coles()
