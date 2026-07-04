import torch
import torch.nn as nn
import torch.nn.functional as F

class MultimodalRiskFusionModel(nn.Module):
    """
    A unified deep learning model that fuses:
    1. Sequential Modality: User's transaction sequence (GRU)
    2. Tabular Modality: Device/Network attributes (MLP)
    3. Graph Modality: Bipartite user-merchant graph embeddings (Embedding lookup & pooling)
    
    Target: Transaction Fraud Detection
    """
    def __init__(self, 
                 num_mccs, 
                 num_merchants, 
                 mcc_emb_dim=16, 
                 merchant_emb_dim=16, 
                 tabular_in_dim=6,
                 hidden_seq=32, 
                 hidden_tab=16, 
                 hidden_fusion=32):
        super().__init__()
        
        # 1. Sequential Modality components
        self.mcc_embed = nn.Embedding(num_embeddings=num_mccs + 1, embedding_dim=mcc_emb_dim, padding_idx=0)
        self.amount_proj = nn.Linear(1, mcc_emb_dim)
        self.seq_gru = nn.GRU(
            input_size=mcc_emb_dim * 2, 
            hidden_size=hidden_seq, 
            num_layers=1, 
            batch_first=True
        )
        
        # 2. Tabular Modality MLP
        self.tabular_mlp = nn.Sequential(
            nn.Linear(tabular_in_dim, hidden_tab * 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_tab * 2, hidden_tab),
            nn.ReLU()
        )
        
        # 3. Graph Modality (Simulating pre-learned node embeddings for merchants)
        # In a real system, these would be loaded from a graph database or GNN (e.g. DGL / PyG)
        self.merchant_graph_embeddings = nn.Embedding(
            num_embeddings=num_merchants + 1, 
            embedding_dim=merchant_emb_dim, 
            padding_idx=0
        )
        
        # 4. Fusion Layers
        fusion_input_dim = hidden_seq + hidden_tab + merchant_emb_dim
        
        self.fusion_mlp = nn.Sequential(
            nn.Linear(fusion_input_dim, hidden_fusion),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_fusion, 1) # Output single logit for binary classification
        )

    def forward(self, mcc_seq, amount_seq, seq_lengths, tabular_feats, merchant_seq):
        # mcc_seq: (batch_size, seq_len)
        # amount_seq: (batch_size, seq_len)
        # seq_lengths: (batch_size,)
        # tabular_feats: (batch_size, tabular_in_dim)
        # merchant_seq: (batch_size, seq_len)
        
        # --- Modality 1: Sequential Encoding ---
        mcc_embs = self.mcc_embed(mcc_seq) # (batch_size, seq_len, mcc_emb_dim)
        amount_expanded = amount_seq.unsqueeze(-1)
        amount_embs = F.relu(self.amount_proj(amount_expanded)) # (batch_size, seq_len, mcc_emb_dim)
        
        seq_features = torch.cat([mcc_embs, amount_embs], dim=-1) # (batch, seq_len, mcc_emb_dim * 2)
        
        packed_seq = nn.utils.rnn.pack_padded_sequence(
            seq_features, seq_lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        _, hidden = self.seq_gru(packed_seq)
        h_seq = hidden.squeeze(0) # (batch_size, hidden_seq)
        
        # --- Modality 2: Tabular Processing ---
        h_tab = self.tabular_mlp(tabular_feats) # (batch_size, hidden_tab)
        
        # --- Modality 3: Graph Embedding Processing ---
        # Look up simulated merchant embeddings in user-merchant interaction graph
        merch_embs = self.merchant_graph_embeddings(merchant_seq) # (batch_size, seq_len, merchant_emb_dim)
        # Average pooling over the sequence to represent overall merchant graph context of the user
        h_graph = torch.mean(merch_embs, dim=1) # (batch_size, merchant_emb_dim)
        
        # --- Modality Fusion ---
        fused_representation = torch.cat([h_seq, h_tab, h_graph], dim=-1) # (batch_size, fusion_input_dim)
        
        logits = self.fusion_mlp(fused_representation) # (batch_size, 1)
        return logits
