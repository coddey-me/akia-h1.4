# models/reasoning.py
import torch
import torch.nn as nn

class ReasoningModule(nn.Module):
    """
    Minimal Hierarchical Reasoning module.
    Stage 1: simple MLP that combines maze + token embeddings.
    """
    def __init__(self, embed_dim=256, hidden_dim=512):
        super().__init__()
        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, embed_dim)

    def forward(self, emb):
        x = torch.relu(self.fc1(emb))
        x = self.fc2(x)
        return x
