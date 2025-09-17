# utils/collate_v4.py
import torch

def pad_action_sequences(paths, pad_value=0):
    """
    Pads a list of 1D tensors (maze paths) to the same length.
    Returns (padded_tensor, max_len)
    """
    if len(paths) == 0:
        return torch.empty(0), 1

    lengths = [len(p) for p in paths]
    max_len = max(lengths)
    padded = torch.full((len(paths), max_len), pad_value, dtype=torch.long)
    for i, p in enumerate(paths):
        padded[i, :len(p)] = torch.tensor(p, dtype=torch.long)
    return padded, max_len
