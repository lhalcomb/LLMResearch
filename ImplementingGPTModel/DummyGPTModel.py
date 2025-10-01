import torch
import torch.nn as nn

"""
The DummyGPTModel class in this code defines a simplified version of a GPT-like
model using PyTorch's neural network module (nn.Module). The model architecture
in the DummyGPTModel class consists of token and positional embeddings, dropout,
a series of transformer blocks (DummyTransformerBlock), a final layer normalization
(DummyLayerNorm), and a linear output layer (out_head). The configuration is
passed in via a Python dictionary, for instance, the GPT_CONFIG_124M dictionary we
created earlier.
"""

class DummyGPTModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.drop_emb = nn.Dropout(cfg["drop_rate"])
        self.trf_blocks = nn.Sequential( #Uses a placeholder for TransformerBlock
        *[DummyTransformerBlock(cfg)
        for _ in range(cfg["n_layers"])]
        )
        self.final_norm = DummyLayerNorm(cfg["emb_dim"]) # Uses a placeholder for LayerNorm
        self.out_head = nn.Linear(
        cfg["emb_dim"], cfg["vocab_size"], bias=False
    )
    def forward(self, in_idx):
        batch_size, seq_len = in_idx.shape
        tok_embeds = self.tok_emb(in_idx)
        pos_embeds = self.pos_emb(
        torch.arange(seq_len, device=in_idx.device)
        )
        x = tok_embeds + pos_embeds
        x = self.drop_emb(x)
        x = self.trf_blocks(x)
        x = self.final_norm(x)
        logits = self.out_head(x)

        return logits

class DummyTransformerBlock(nn.Module): #A simple placeholder class that will be replaced by a real TransformerBlock later
    def __init__(self, cfg):
        super().__init__()
    def forward(self, x): #This just returns input
        return x
    
class DummyLayerNorm(nn.Module): #A simple placeholder class that will be replaced by a real LayerNorm later
    def __init__(self, normalized_shape, eps=1e-5): #The parameters here are just to mimic the LayerNorm interface.
        super().__init__()
    def forward(self, x):
        return x