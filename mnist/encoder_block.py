import torch.nn as nn

from mnist.self_attention import SelfAttention

class EncoderBlock(nn.Module):
  def __init__(self, embed_dim, num_layers=4):
    super().__init__()
    self.num_layers = num_layers
    self.attention = SelfAttention(embed_dim, 2)

    self.attn_norm = nn.LayerNorm(normalized_shape=(embed_dim,))
    self.output_norm = nn.LayerNorm(normalized_shape=(embed_dim,))
    # Feed-forward network with two linear layers and ReLU activation
    self.ffn_layer1 = nn.Linear(embed_dim, embed_dim * 4)
    self.relu = nn.ReLU()
    self.ffn_layer2 = nn.Linear(embed_dim * 4, embed_dim)


  def forward(self, input_embs):
    out = input_embs
    for i in range(self.num_layers):
      # attention with residual connection
      context_embs = self.attention(input_embs)
      context_embs = context_embs + input_embs
      context_embs_norm = self.attn_norm(context_embs)

      # Feed-forward layer with residual connection
      ffn_output = self.ffn_layer1(context_embs_norm)
      ffn_output = self.relu(ffn_output)
      ffn_output = self.ffn_layer2(ffn_output)
      ffn_output = ffn_output + context_embs_norm

      out = self.output_norm(ffn_output)
    return out
