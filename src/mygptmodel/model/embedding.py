import torch
import math

def build_embeddings(GPT_CONFIG_124M, DEVICE):
    token_emb = torch.nn.Embedding(
        num_embeddings=GPT_CONFIG_124M["vocab_size"],
        embedding_dim=GPT_CONFIG_124M["emb_dim"], device=DEVICE
    )

    max_len = GPT_CONFIG_124M["context_length"]
    d_model = GPT_CONFIG_124M["emb_dim"]

    pos = torch.arange(0, max_len, dtype=torch.float32, device=DEVICE).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32, device=DEVICE) * (-math.log(10000.0) / d_model))

    pos_emb = torch.zeros((max_len, d_model), dtype=torch.float32, device=DEVICE)
    pos_emb[:, 0::2] = torch.sin(pos * div_term)
    pos_emb[:, 1::2] = torch.cos(pos * div_term)
    pos_emb = pos_emb.unsqueeze(0)

    dropout = torch.nn.Dropout(GPT_CONFIG_124M["drop_rate"]).to(DEVICE)

    return token_emb, pos_emb, dropout
