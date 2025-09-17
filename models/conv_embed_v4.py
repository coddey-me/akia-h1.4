# models/conv_embed_v4.py
import torch
import torch.nn as nn

class ConvEmbedV4(nn.Module):
    """
    Maze encoder: takes 2D maze (B,1,H,W) -> embedding (B, embed_dim)
    """
    def __init__(self, embed_dim=256):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool2 = nn.MaxPool2d(2)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.pool3 = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(128, embed_dim)

    def forward(self, x):
        # x: (B,1,H,W)
        x = torch.relu(self.conv1(x))
        x = self.pool1(x)
        x = torch.relu(self.conv2(x))
        x = self.pool2(x)
        x = torch.relu(self.conv3(x))
        x = self.pool3(x)  # (B,128,1,1)
        x = x.view(x.size(0), -1)  # (B,128)
        x = self.fc(x)             # (B,embed_dim)
        return x
