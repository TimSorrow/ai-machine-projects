import torch
import torch.nn as nn
import torch.nn.functional as F

class TransactionEncoder(nn.Module):
    """
    Sequence encoder for transactional event data.
    Embeds categorical features (MCC) and dense features (Amount), 
    and processes them using a GRU layer to yield a user representation embedding.
    """
    def __init__(self, num_mccs, mcc_emb_dim=16, amount_dim=1, hidden_dim=64, embedding_dim=64):
        super().__init__()
        # +1 to num_mccs for the 0-padding token
        self.mcc_embed = nn.Embedding(num_embeddings=num_mccs + 1, embedding_dim=mcc_emb_dim, padding_idx=0)
        self.amount_fc = nn.Linear(amount_dim, mcc_emb_dim) # Project amount to match MCC embedding size
        
        # Combined input size is mcc_emb_dim * 2
        input_size = mcc_emb_dim * 2
        
        # Sequence model
        self.gru = nn.GRU(input_size=input_size, hidden_size=hidden_dim, num_layers=1, batch_first=True)
        
        # Projection/embedding head
        self.fc = nn.Linear(hidden_dim, embedding_dim)

    def forward(self, mcc_seq, amount_seq, lengths):
        # mcc_seq shape: (batch_size, seq_len)
        # amount_seq shape: (batch_size, seq_len)
        # lengths shape: (batch_size,)
        
        # 1. Embed inputs
        mcc_embs = self.mcc_embed(mcc_seq) # (batch_size, seq_len, mcc_emb_dim)
        
        # Expand amount_seq to (batch_size, seq_len, 1) and project
        amount_expanded = amount_seq.unsqueeze(-1)
        amount_embs = F.relu(self.amount_fc(amount_expanded)) # (batch_size, seq_len, mcc_emb_dim)
        
        # 2. Concatenate features
        features = torch.cat([mcc_embs, amount_embs], dim=-1) # (batch_size, seq_len, mcc_emb_dim * 2)
        
        # 3. Recurrent processing
        # Use pack_padded_sequence to ignore padding elements during GRU recurrence
        packed_features = nn.utils.rnn.pack_padded_sequence(
            features, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        
        _, hidden = self.gru(packed_features) # hidden shape: (1, batch_size, hidden_dim)
        
        # 4. fixed-length embedding representation
        h = hidden.squeeze(0) # (batch_size, hidden_dim)
        embeddings = self.fc(h) # (batch_size, embedding_dim)
        
        return embeddings


class CoLESModel(nn.Module):
    """
    Wrapper for CoLES training. Includes the TransactionEncoder and a Contrastive Projection Head.
    """
    def __init__(self, encoder, embedding_dim=64, projection_dim=32):
        super().__init__()
        self.encoder = encoder
        
        # Non-linear projection head for contrastive learning
        self.projection_head = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, projection_dim)
        )

    def forward(self, mcc_seq, amount_seq, lengths):
        emb = self.encoder(mcc_seq, amount_seq, lengths)
        proj = self.projection_head(emb)
        # L2-normalize projection output for cosine similarity
        proj_norm = F.normalize(proj, p=2, dim=1)
        return proj_norm


def contrastive_loss(z1, z2, temperature=0.1):
    """
    Calculates contrastive loss (similar to NT-Xent / InfoNCE) between z1 and z2.
    z1 and z2 are L2-normalized embeddings of shape (batch_size, projection_dim).
    """
    batch_size = z1.shape[0]
    
    # Concatenate representations
    representations = torch.cat([z1, z2], dim=0) # (2 * batch_size, projection_dim)
    
    # Calculate similarity matrix
    similarity_matrix = torch.matmul(representations, representations.T) # (2 * batch_size, 2 * batch_size)
    
    # Scale by temperature
    similarity_matrix = similarity_matrix / temperature
    
    # Create targets for cross-entropy
    # For representation i, the positive match is at (i + batch_size) % (2 * batch_size)
    labels = torch.arange(batch_size, device=z1.device)
    labels = torch.cat([labels + batch_size, labels], dim=0)
    
    # Mask self-similarity (the diagonal)
    mask = torch.eye(2 * batch_size, device=z1.device).bool()
    similarity_matrix = similarity_matrix.masked_fill(mask, -9e15)
    
    loss = F.cross_entropy(similarity_matrix, labels)
    return loss
