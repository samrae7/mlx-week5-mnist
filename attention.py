import torch
import torch.nn as nn

class Attention(nn.Module):
    # TODO multi head
    def __init__(self, embed_dim):
        super().__init__()
        self.embed_dim = embed_dim
        #  takes in seq * embed_dim. outputs seq * embed_dim so needs embed_dim * embed_dim
        self.linear_q = nn.Linear(embed_dim, embed_dim)
        self.linear_k = nn.Linear(embed_dim, embed_dim)
        self.linear_v = nn.Linear(embed_dim, embed_dim)


    def forward(self, input):
        q = self.linear_q(input)
        k = self.linear_k(input)
        v = self.linear_v(input)
        # k will have dimensions seq * embed_dim
        attention_scores = q @ k.transpose(-2, -1) / self.embed_dim**0.5
        base = torch.full_like(attention_scores, float("-inf"))
        mask = torch.triu(base, diagonal=1)
        attention_scores = attention_scores + mask
        attention_weights = attention_scores.softmax(dim=-1)
        context_emb = attention_weights @ v
        return context_emb