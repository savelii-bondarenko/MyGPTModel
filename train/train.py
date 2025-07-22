import torch
from torch.utils.data import DataLoader
from config.gpt_config import GPT_CONFIG_124M
from model.transformer import Transformer
from model.embedding import build_embeddings
from data.dataset import TokenPair, tokenize_text
from train.utils import save_model
import tqdm

if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")

tokens = tokenize_text()
dataset = TokenPair(tokens, GPT_CONFIG_124M["context_length"])
dataloader = DataLoader(dataset, batch_size=1000, shuffle=True)
model = Transformer(**GPT_CONFIG_124M).to(DEVICE)
embedding, pos_embedding, dropout = build_embeddings(GPT_CONFIG_124M, DEVICE)

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
model.train()

for epoch in tqdm.tqdm(range(100)):
    for x_batch, y_batch in dataloader:
        x_batch, y_batch = x_batch.to(DEVICE), y_batch.to(DEVICE)

        token_embedding = embedding(x_batch)  
        positions = torch.arange(GPT_CONFIG_124M["context_length"]).unsqueeze(0).expand(x_batch.size(0), GPT_CONFIG_124M["context_length"]).to(DEVICE)
        pos_embeds = pos_embedding(positions)
        combined = dropout(token_embedding + pos_embeds)

        logits = model(combined) 
        logits = logits.transpose(1, 2)  

        loss = criterion(logits, y_batch)  
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    save_model(model, optimizer, epoch, f"checkpoints/gpt_epoch_{epoch}.pt")