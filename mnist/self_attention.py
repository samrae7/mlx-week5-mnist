import torch.nn as nn


class SelfAttention(nn.module):
    def __init__(self, embed_dim, num_heads):

        if embed_dim % num_heads != 0:
            raise ValueError(f"embed_dim {embed_dim} must be divisible by num_heads {num_heads}.")
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.linear_q = nn.Linear(embed_dim, embed_dim)
        self.linear_k = nn.Linear(embed_dim, embed_dim)
        self.linear_v = nn.Linear(embed_dim, embed_dim)
        self.softmax = nn.Softmax(dim=-1)
        self.linear_out = nn.Linear(embed_dim, embed_dim) 

    def forward(self, input):
        q = self.linear_q(input)
        k = self.linear_k(input)
        v = self.linear_v(input)
        # heads should have shape: [batch_size, num_heads, seq_length, head_dim]
        q = q.view(q.shape[0], q.shape[1], self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(k.shape[0], k.shape[1], self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(v.shape[0], v.shape[1], self.num_heads, self.head_dim).transpose(1, 2)
        transposed_k = k.transpose(-2,-1)
        scaling_factor = self.head_dim**0.5
        attention_scores = (q @ transposed_k) / scaling_factor
        attention_weights = self.softmax(attention_scores)
        context_emb = attention_weights @ v
        # this has shape batch_size, num_heads, seq_length, head_dim,
        # so we need to do something to get batch_size, seq_length, emb_dim ie. 2,4,12
        context_emb = context_emb.transpose(1, 2).contiguous()  # [batch_size, seq_length, num_heads, head_dim]
        context_emb = context_emb.view(context_emb.shape[0], context_emb.shape[1], -1)  # [batch_size, seq_length, embed_dim]
        context_emb = self.linear_out(context_emb)
        return context_emb
    