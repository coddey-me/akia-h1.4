# models/tokenizer.py
class CharTokenizer:
    """
    Minimal character tokenizer for Stage 1 testing.
    """
    def __init__(self):
        self.chars = list("abcdefghijklmnopqrstuvwxyz0123456789.,:;!?()[]{} ")
        self.vocab_size = len(self.chars) + 1  # +1 for PAD
        self.pad_token = 0
        self.char2idx = {c: i+1 for i, c in enumerate(self.chars)}
        self.idx2char = {i+1: c for i, c in enumerate(self.chars)}

    def encode(self, text):
        return [self.char2idx.get(c, 0) for c in text]

    def decode(self, indices):
        return "".join([self.idx2char.get(i, "?") for i in indices])
