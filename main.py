# main.py
import os
import torch
import numpy as np

# -----------------------------
# Import modules safely
# -----------------------------
try:
    from models.hrm_v4 import HRM_V4
except ImportError:
    print("Warning: HRM_V4 model not found. Make sure 'models/hrm_v4.py' exists.")
    HRM_V4 = None

try:
    from models.tokenizer import CharTokenizer
except ImportError:
    print("Warning: Tokenizer not found. Make sure 'models/tokenizer.py' exists.")
    CharTokenizer = None

try:
    from utils.collate_v4 import pad_action_sequences
except ImportError:
    print("Warning: Collate utils not found. Make sure 'utils/collate_v4.py' exists.")
    pad_action_sequences = None

try:
    from utils.visualization import plot_maze
except ImportError:
    print("Warning: Visualization utils not found. Maze plotting will be skipped.")
    plot_maze = None

# -----------------------------
# Configuration
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

maze_size = 50
batch_size = 2
embed_dim = 256
high_hidden = 512
low_hidden = 512
n_actions = 4
seq_len = 10
stage = 1  # 1 = Stage 1 dummy run, 2 = Stage 2 training/inference

# -----------------------------
# Initialize tokenizer
# -----------------------------
if CharTokenizer:
    tokenizer = CharTokenizer()
    vocab_size = tokenizer.vocab_size
else:
    tokenizer = None
    vocab_size = 100  # fallback dummy size

# -----------------------------
# Initialize model
# -----------------------------
if HRM_V4:
    model = HRM_V4(embed_dim=embed_dim,
                   high_hidden=high_hidden,
                   low_hidden=low_hidden,
                   n_actions=n_actions,
                   vocab_size=vocab_size).to(device)
    model.eval()
    print("HRM_V4 initialized successfully.")
else:
    model = None
    print("Model initialization skipped due to missing HRM_V4.")

# -----------------------------
# Stage 1: Dummy run
# -----------------------------
if stage == 1:
    print("Running Stage 1: dummy forward pass")

    # Dummy maze
    dummy_mazes = torch.zeros((batch_size, 1, maze_size, maze_size), device=device)
    dummy_paths = [torch.arange(maze_size) for _ in range(batch_size)]

    # Dummy text
    dummy_text_input = torch.randint(1, vocab_size, (batch_size, seq_len), device=device)

    # Maze forward pass
    if model and pad_action_sequences:
        padded_paths, max_len = pad_action_sequences(dummy_paths)
        with torch.no_grad():
            maze_logits = model(dummy_mazes, seq_len=max_len, task='maze')
        print("Maze logits shape:", maze_logits.shape)
    else:
        print("Skipping maze forward pass due to missing components.")

    # Text forward pass
    if model:
        with torch.no_grad():
            text_logits = model(dummy_mazes, seq_len=seq_len, task='text')
        print("Text logits shape:", text_logits.shape)
    else:
        print("Skipping text forward pass due to missing model.")

    # Visualization
    if plot_maze:
        sample_maze = np.zeros((maze_size, maze_size))
        sample_path = [(i, i) for i in range(maze_size)]
        plot_maze(sample_maze, sample_path)
    else:
        print("Skipping maze visualization due to missing plot_maze function.")

# -----------------------------
# Stage 2: Training / Real Data
# -----------------------------
elif stage == 2:
    print("Stage 2: Training/inference with real data")

    # Check for datasets
    maze_file = "data/mazes.pt"
    code_file = "data/code.pt"
    convo_file = "data/convo.pt"

    for f in [maze_file, code_file, convo_file]:
        if not os.path.exists(f):
            print(f"Warning: Dataset {f} not found. Skipping.")

    # TODO: Add training loop here
    # Example placeholders:
    # - Load datasets
    # - Create DataLoaders
    # - Train HRM_V4
    # - Save checkpoints
    print("Training loop not yet implemented. Placeholder only.")

else:
    print("Invalid stage. Please set stage = 1 or stage = 2.")

print("main.py execution complete.")
