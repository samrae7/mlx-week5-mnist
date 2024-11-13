import torch.nn as nn

class FeedForward(nn.Module):
    def __init__(self, embed_dim):
        super().__init__(self)
        self.relu = nn.ReLU()
        self.layer1 = nn.Linear(embed_dim, embed_dim * 4)
        self.layer2 = nn.Linear(embed_dim * 4, embed_dim)
        self.norm = nn.LayerNorm(normalized_shape=(embed_dim,))


    def forward(self, input):
        ffn_output = self.ffn_layer1(input)
        ffn_output = self.relu(ffn_output)
        ffn_output = self.layer2(ffn_output)
        ffn_output = ffn_output + input