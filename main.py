# main.py
import torch
import numpy as np
from models.hrm_v4 import HRM_V4
from models.tokenizer import CharTokenizer
from utils.collate_v4 import pad_action_sequences
from utils.visualization import plot_maze

# -----------------------------
# Config
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

maze_size = 50
batch_size = 2
embed_dim = 256
high_hidden = 512
low_hidden = 512
n_actions = 4

# -----------------------------
# Tokenizer
# -----------------------------
tokenizer = CharTokenizer()
vocab_size = tokenizer.vocab_size

# -----------------------------
# Initialize model
# -----------------------------
model = HRM_V4(embed_dim=embed_dim, high_hidden=high_hidden,
               low_hidden=low_hidden, n_actions=n_actions,
               vocab_size=vocab_size).to(device)
model.eval()
print("HRM_V4 initialized successfully.")

# -----------------------------
# Dummy maze data
# -----------------------------
dummy_mazes = torch.zeros((batch_size, 1, maze_size, maze_size), device=device)
dummy_paths = [torch.arange(maze_size) for _ in range(batch_size)]

# -----------------------------
# Dummy text data
# -----------------------------
seq_len = 10
dummy_text_input = torch.randint(1, vocab_size, (batch_size, seq_len), device=device)

# -----------------------------
# Forward pass: Maze
# -----------------------------
padded_paths, max_len = pad_action_sequences(dummy_paths)
with torch.no_grad():
    maze_logits = model(dummy_mazes, seq_len=max_len, task='maze')
print("Maze logits shape:", maze_logits.shape)

# -----------------------------
# Forward pass: Text
# -----------------------------
with torch.no_grad():
    text_logits = model(dummy_mazes, seq_len=seq_len, task='text')
print("Text logits shape:", text_logits.shape)

# -----------------------------
# Visualize a sample maze
# -----------------------------
sample_maze = np.zeros((maze_size, maze_size))
sample_path = [(i, i) for i in range(maze_size)]
plot_maze(sample_maze, sample_path)

print("Stage 1 run complete. Model and visualization working.")
