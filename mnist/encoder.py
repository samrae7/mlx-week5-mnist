import torch.nn as nn

from mnist.encoder_block import EncoderBlock

class Encoder(nn.Module):

    def __init__(self, embed_dim, num_layers):
        super.__init__(self)
        self.layers = nn.ModuleList([EncoderBlock(embed_dim, num_layers) for i in range(num_layers)])
    
    def forward(self, input):
        io = input
        for layer in self.layers:
            io = layer(io)
        return io
