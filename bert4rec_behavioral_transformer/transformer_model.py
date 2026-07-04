import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class SASRecTransformer(nn.Module):
    """
    SASRec (Self-Attention based Sequential Recommendation) style Transformer.
    Uses causal self-attention layers to model event sequences auto-regressively.
    Given [e_1, e_2, ..., e_t], it predicts [e_2, e_3, ..., e_{t+1}].
    """
    def __init__(self, num_events, d_model=32, nhead=2, num_layers=2, max_len=50, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        
        # Vocab size includes padding token (0)
        self.item_emb = nn.Embedding(num_embeddings=num_events + 1, embedding_dim=d_model, padding_idx=0)
        
        # Positional Embeddings
        self.pos_emb = nn.Embedding(num_embeddings=max_len, embedding_dim=d_model)
        
        self.emb_dropout = nn.Dropout(p=dropout)
        
        # Transformer layers
        # batch_first=True makes shape (batch_size, seq_len, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model * 4,
            dropout=dropout, activation='relu', batch_first=True, norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Logits head
        self.output_fc = nn.Linear(d_model, num_events + 1)
        
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.Linear):
            nn.init.xavier_normal_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0.0)

    def generate_causal_mask(self, sz, device):
        """Generates a lower-triangular causal mask to prevent attending to future tokens."""
        mask = torch.triu(torch.ones(sz, sz, device=device) * float('-inf'), diagonal=1)
        return mask

    def forward(self, seqs):
        # seqs shape: (batch_size, seq_len)
        batch_size, seq_len = seqs.size()
        device = seqs.device
        
        # 1. Look up embeddings
        emb = self.item_emb(seqs) # (batch_size, seq_len, d_model)
        
        # Scale embeddings by sqrt(d_model)
        emb = emb * math.sqrt(self.d_model)
        
        # 2. Add Positional Embeddings
        positions = torch.arange(seq_len, dtype=torch.long, device=device).unsqueeze(0).expand(batch_size, -1)
        pos_emb = self.pos_emb(positions)
        
        x = self.emb_dropout(emb + pos_emb)
        
        # 3. Create Causal Mask and Padding Key Mask
        causal_mask = self.generate_causal_mask(seq_len, device)
        
        # Key padding mask: True where value is 0 (padding token)
        key_padding_mask = (seqs == 0)
        
        # 4. Process sequence with Transformer
        # PyTorch Transformer Encoder handles causal masks and padding masks
        h = self.transformer(
            x, mask=causal_mask, is_causal=True, src_key_padding_mask=key_padding_mask
        ) # (batch_size, seq_len, d_model)
        
        # 5. Output logits
        logits = self.output_fc(h) # (batch_size, seq_len, num_events + 1)
        
        return logits, h
        
    def get_sequence_embedding(self, seqs, lengths):
        """
        Extracts the sequence-level representation.
        Takes the transformer hidden state at the last non-padded event index.
        """
        _, h = self.forward(seqs) # (batch_size, seq_len, d_model)
        
        batch_size = h.size(0)
        # Lengths are 1-indexed, convert to 0-indexed index of last item
        last_item_indices = (lengths - 1).clamp(min=0).unsqueeze(1).unsqueeze(2).expand(-1, -1, self.d_model)
        
        # Gather the last valid state
        seq_embs = torch.gather(h, dim=1, index=last_item_indices).squeeze(1) # (batch_size, d_model)
        return seq_embs
