import torch

def build_embeddings(GPT_CONFIG_124M, DEVICE):
    token_emb = torch.nn.Embedding(
        num_embeddings=GPT_CONFIG_124M["vocab_size"],
        embedding_dim=GPT_CONFIG_124M["emb_dim"]).to(DEVICE)
        
    pos_emb = torch.nn.Embedding(
        num_embeddings=GPT_CONFIG_124M["context_length"],
        embedding_dim=GPT_CONFIG_124M["emb_dim"]).to(DEVICE)
    
    dropout = torch.nn.Dropout(GPT_CONFIG_124M["drop_rate"]).to(DEVICE)
    return token_emb, pos_emb, dropout