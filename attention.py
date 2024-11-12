import torch
import torch.nn as nn

class Attention(nn.Module):
    # TODO multi head
    def __init__(self, embed_dim, num_heads=4):
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError(f"embed_dim {embed_dim} must be divisible by num_heads {num_heads}.")
        self.embed_dim = embed_dim
        self.head_dim = embed_dim // num_heads
        self.num_heads = num_heads
        #  takes in seq * embed_dim. outputs seq * embed_dim so needs embed_dim * embed_dim
        self.linear_q = nn.Linear(embed_dim, embed_dim)
        self.linear_k = nn.Linear(embed_dim, embed_dim)
        self.linear_v = nn.Linear(embed_dim, embed_dim)


    def forward(self, input):
        # these have dim seq_length, embed_dim
        q = self.linear_q(input)
        k = self.linear_k(input)
        v = self.linear_v(input)
        # heads should have shape: [num_heads, seq_length, head_dim]
        q = q.view(q.shape[0], self.num_heads, self.head_dim)
        k = k.view(k.shape[0], self.num_heads, self.head_dim)
        v = v.view(v.shape[0], self.num_heads, self.head_dim)
        # k will have dimensions seq * embed_dim
        attention_scores = q @ k.transpose(-2, -1) / self.embed_dim**0.5
        base = torch.full_like(attention_scores, float("-inf"))
        mask = torch.triu(base, diagonal=1)
        attention_scores = attention_scores + mask
        attention_weights = attention_scores.softmax(dim=-1)
        context_emb = attention_weights @ v
        # this ^ has dims seq, num_head, head_dim. We want seq, embed_dim
        context_emb = context_emb.view(context_emb.shape[0], -1)
        return context_emb