import torch.nn as nn

from decoder_block import DecoderBlock

class MNISTDecoder(nn.Module):

    def __init__(self, embed_dim, encoder_embed_dim, num_layers=4):
        super().__init__()
        self.layers = nn.ModuleList([DecoderBlock(embed_dim, num_layers, encoder_embed_dim=encoder_embed_dim) for i in range(num_layers)])
    
    def forward(self, decoder_input, enc_out):
        io = decoder_input
        for layer in self.layers:
            io = layer(io, enc_out)
        return io
