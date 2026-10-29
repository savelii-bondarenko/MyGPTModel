import torch
import math

class Transformer(torch.nn.Module):
    def __init__(self, drop_rate, emb_dim, n_heads, vocab_size, **kwargs):
        super().__init__()
        self.dropout = torch.nn.Dropout(drop_rate)
        self.layerNorm1 = LayerNorm(emb_dim)
        self.layerNorm2 = LayerNorm(emb_dim)
        self.feedforward = FeedForward(emb_dim)
        self.attn = MultiHeadAttention(
            d_in=emb_dim,
            d_out=emb_dim,
            num_heads=n_heads,
        )
        self.linear_proj = torch.nn.Linear(emb_dim, vocab_size)  


    def forward(self, x):
        x_old = x
        x = self.layerNorm1(x)
        x = self.attn(x)
        x = self.dropout(x)
        x += x_old 

        x_old = x
        x = self.layerNorm2(x)
        x = self.feedforward(x)
        x = self.dropout(x)
        x += x_old 
        logits = self.linear_proj(x)
        return logits

class MultiHeadAttention(torch.nn.Module):
    def __init__(self, d_in, d_out, num_heads):
        super().__init__()

        self.num_heads = num_heads
        self.head_dim = d_out // num_heads

        assert d_out % num_heads == 0, "d_out must be divisible by num_heads"

        self.W_k = torch.nn.Parameter(torch.randn(d_in, d_out))
        self.W_q = torch.nn.Parameter(torch.randn(d_in, d_out))
        self.W_v = torch.nn.Parameter(torch.randn(d_in, d_out))


    def forward(self, x):
        batch_size, seq_len, _ = x.shape

        keys = x @ self.W_k
        values = x @ self.W_v
        queries = x @ self.W_q

        keys = keys.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        values = values.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        queries = queries.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        attn_scores = queries @ keys.transpose(-2, -1) 
        attn_scores = attn_scores / math.sqrt(self.head_dim)

        mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
        mask = mask.unsqueeze(0).unsqueeze(0).to(x.device)  
        attn_scores = attn_scores.masked_fill(mask, float('-inf'))

        attn_weights = torch.softmax(attn_scores, dim=-1)

        context = torch.matmul(attn_weights, values)  

        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)  

        return context

class LayerNorm(torch.nn.Module):
    def __init__(self, emb_dim, eps=1e-5):
        super().__init__()
        self.weights = torch.nn.Parameter(torch.ones(emb_dim))
        self.bias = torch.nn.Parameter(torch.zeros(emb_dim))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        variance = x.var(dim=-1, keepdim=True, unbiased=False)
        normed = (x - mean) / torch.sqrt(variance + self.eps)      
        return self.weights * normed + self.bias    

class FeedForward(torch.nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.input_layer = torch.nn.Linear(emb_dim, 4*emb_dim)
        self.gelu = GELU()
        self.output_layer = torch.nn.Linear(4*emb_dim, emb_dim)

    def forward(self, x):
        x = self.input_layer(x)
        x = self.gelu(x)
        return self.output_layer(x)

class GELU(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return 0.5*x*(1+torch.tanh(
            torch.sqrt(torch.tensor(2.0/torch.pi))*
            (x+0.044715*torch.pow(x,3))
        ))