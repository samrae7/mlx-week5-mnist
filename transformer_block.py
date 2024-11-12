import torch.nn as nn

from attention import Attention

class TransformerBlock(nn.Module):
  def __init__(self, embed_dim):
    super().__init__()
    self.attention = Attention(embed_dim)

    self.attn_norm = nn.LayerNorm(normalized_shape=(embed_dim,))
    self.output_norm = nn.LayerNorm(normalized_shape=(embed_dim,))
    # Feed-forward network with two linear layers and ReLU activation
    self.ffn_layer1 = nn.Linear(embed_dim, embed_dim * 4)
    self.relu = nn.ReLU()
    self.ffn_layer2 = nn.Linear(embed_dim * 4, embed_dim)


  def forward(self, input_embs):
    # attention with residual connection
    context_embs = self.attention(input_embs)
    context_embs = context_embs + input_embs
    context_embs_norm = self.attn_norm(context_embs)

    # Feed-forward layer with residual connection
    ffn_output = self.ffn_layer1(context_embs_norm)
    ffn_output = self.relu(ffn_output)
    ffn_output = self.ffn_layer2(ffn_output)
    ffn_output = ffn_output + context_embs_norm

    output_norm = self.output_norm(ffn_output)
    return output_norm