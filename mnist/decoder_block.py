import torch.nn as nn

from cross_attention import CrossAttention
from feed_forward import FeedForward
from self_attention import SelfAttention

class DecoderBlock(nn.Module):
    def __init__(self, embed_dim, num_layers, encoder_embed_dim):
        super().__init__()
        self.num_layers = num_layers

        self.masked_self_attn = SelfAttention(embed_dim=embed_dim, num_heads=4, masked=True)
        self.cross_attn = CrossAttention(encoder_embed_dim=encoder_embed_dim, decoder_embed_dim=embed_dim, d_k=64)
        self.ff = FeedForward(embed_dim)

        self.self_attn_norm = nn.LayerNorm(normalized_shape=(embed_dim,))
        self.cross_attn_norm = nn.LayerNorm(normalized_shape=(embed_dim,))
        self.ff_norm = nn.LayerNorm(normalized_shape=(embed_dim,))

    def forward(self, decoder_io, encoder_out):
        self_attn_embs = self.masked_self_attn(decoder_io)
        self_attn_embs = self.self_attn_norm(self_attn_embs + decoder_io)

        cross_attn_embs = self.cross_attn(encoder_out, self_attn_embs)
        cross_attn_embs = self.cross_attn_norm(cross_attn_embs + self_attn_embs)

        out = self.ff(cross_attn_embs)
        out = self.ff_norm(out + cross_attn_embs)
        return out