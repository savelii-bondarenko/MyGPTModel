import torch
import tqdm
from torch.utils.data import DataLoader

from mygptmodel.config.gpt_config import GPT_CONFIG_124M, TRAIN_EPOCHS
from mygptmodel.data.dataset import TokenPair, tokenize_text
from mygptmodel.model.embedding import build_embeddings
from mygptmodel.model.transformer import Transformer
from mygptmodel.train.utils import save_model

if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")

print("Device: ", DEVICE)

tokens = tokenize_text()
dataset = TokenPair(tokens, GPT_CONFIG_124M["context_length"])
dataloader = DataLoader(dataset, batch_size=1000, shuffle=True)
model = Transformer(**GPT_CONFIG_124M).to(DEVICE)
embedding, pos_embedding, dropout = build_embeddings(GPT_CONFIG_124M, DEVICE)

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
model.train()

for epoch in tqdm.tqdm(range(TRAIN_EPOCHS)):
    for x_batch, y_batch in dataloader:
        x_batch, y_batch = x_batch.to(DEVICE), y_batch.to(DEVICE)

        token_embedding = embedding(x_batch)  
        pos_embeds = pos_embedding[:, :x_batch.size(1), :]
        combined = dropout(token_embedding + pos_embeds)

        logits = model(combined) 
        logits = logits.transpose(1, 2)  

        loss = criterion(logits, y_batch)  
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
save_model(model, optimizer, epoch, f"checkpoints/gpt_epoch_1.pt")