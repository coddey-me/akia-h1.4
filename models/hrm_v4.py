# models/hrm_v4.py
import torch
import torch.nn as nn
from .conv_embed_v4 import ConvEmbedV4
from .reasoning import ReasoningModule

class HRM_V4(nn.Module):
    def __init__(self, embed_dim=256, high_hidden=512, low_hidden=512,
                 n_actions=4, vocab_size=60):
        super().__init__()
        self.encoder = ConvEmbedV4(embed_dim)
        self.reasoning = ReasoningModule(embed_dim, high_hidden)

        # Heads
        self.action_head = nn.Linear(embed_dim, n_actions)
        self.token_head = nn.Linear(embed_dim, vocab_size)

    def forward(self, maze_batch, seq_len=10, task='maze'):
        """
        maze_batch: (B,1,H,W)
        seq_len: sequence length
        task: 'maze' or 'text'
        """
        B = maze_batch.size(0)
        emb = self.encoder(maze_batch)           # (B, embed_dim)
        emb = self.reasoning(emb)               # (B, embed_dim)

        # Expand embedding over sequence
        emb_seq = emb.unsqueeze(1).repeat(1, seq_len, 1)  # (B, seq_len, embed_dim)

        if task == 'maze':
            logits = self.action_head(emb_seq)  # (B, seq_len, n_actions)
        else:
            logits = self.token_head(emb_seq)   # (B, seq_len, vocab_size)
        return logits
