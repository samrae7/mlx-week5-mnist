import torch.nn as nn

from mnist.decoder_block import DecoderBlock

class Decoder(nn.Module):

    def __init__(self, embed_dim, num_layers=4):
        super.__init__(self)
        self.layers = nn.ModuleList([DecoderBlock(embed_dim, num_layers) for i in range(num_layers)])
    
    def forward(self, input):
        io = input
        for layer in self.layers:
            io = layer(io)
        return io
