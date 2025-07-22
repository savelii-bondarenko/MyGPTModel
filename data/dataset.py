from tiktoken import get_encoding
from torch.utils.data import Dataset
import torch

def tokenize_text():
    TXT = "the-verdict.txt"
    tokenizer = get_encoding("gpt2")
    with open(TXT, "r", encoding="utf-8") as f:
        text = f.read()
        tokenized_text = tokenizer.encode(text)
    return tokenized_text

class TokenPair(Dataset):
    def __init__(self, tokens, chunk_size):
        self.chunk_size = chunk_size
        self.pairs = [
            (tokens[i:i+chunk_size], tokens[i+1:i+1+chunk_size])
            for i in range(len(tokens) - chunk_size - 1)
        ]

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        x, y = self.pairs[idx]
        return torch.tensor(x, dtype=torch.long), torch.tensor(y, dtype=torch.long)