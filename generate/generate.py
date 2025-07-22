import torch
from model.transformer import Transformer
from config.gpt_config import GPT_CONFIG_124M
from model.embedding import build_embeddings
from tiktoken import get_encoding
from train.utils import load_model

if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")

model = Transformer(**GPT_CONFIG_124M).to(DEVICE)
load_model(model, None, "checkpoints/gpt_epoch_9.pt", map_location=DEVICE)
embedding, pos_embedding, dropout = build_embeddings(GPT_CONFIG_124M, DEVICE)
model.eval()

def generate_text(count_new_tokens: int, input: str, temperature:float):
    tokenizer = get_encoding("gpt2")
    tokens = tokenizer.encode(input)

    for _ in range(count_new_tokens):
        input_ids = torch.tensor(tokens[-GPT_CONFIG_124M["context_length"]:], dtype=torch.long).unsqueeze(0).to(DEVICE)  
        token_emb = embedding(input_ids) 
        positions = torch.arange(input_ids.size(1)).unsqueeze(0).to(DEVICE)
        pos_emb = pos_embedding(positions)
        combined = dropout(token_emb + pos_emb)

        with torch.no_grad():
            logits = model(combined)  

        next_token_logits = logits[0, -1] 
        scaled_logits = next_token_logits / temperature
        top_logits, top_pos = torch.topk(scaled_logits, 3)
        mask = torch.full_like(scaled_logits, False, dtype=torch.bool)
        mask[top_pos] = True
        new_logits = torch.where(mask, scaled_logits, torch.tensor(float("-inf")).to(DEVICE))

        topk_prob = torch.softmax(new_logits, dim=0)
        next_token_id = torch.multinomial(topk_prob, num_samples=1).item()

        tokens.append(next_token_id)
        
    output = tokenizer.decode(tokens)
    print(output)

if __name__ == '__main__':
    generate_text(count_new_tokens=32, input="Hello", temperature=0.8)
