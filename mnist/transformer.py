import torch
import torch.nn as nn

from mnist_decoder import MNISTDecoder
from encoder import Encoder

encoder_embed_dim = 64
decoder_embed_dim = 64
num_encoder_blocks = 4
num_decoder_blocks = 4
# 0-9 + start + end tokens
vocab_size = 12

class Transformer(nn.Module):
    def __init__(self, vocab_size = vocab_size, max_seq_length = 10):
        super().__init__()
        # TODO dont hardcode 784
        self.encoder_projection = nn.Linear(784, encoder_embed_dim)
        # vocab size matches ?in this case yes because bothe represent digits 0 - 9
        self.encoder_emb = nn.Embedding(num_embeddings=vocab_size, embedding_dim=encoder_embed_dim)        
        self.decoder_emb = nn.Embedding(num_embeddings=vocab_size, embedding_dim=decoder_embed_dim)        
        self.encoder = Encoder(embed_dim=encoder_embed_dim, num_layers=num_encoder_blocks)
        self.decoder = MNISTDecoder(embed_dim=decoder_embed_dim, encoder_embed_dim=encoder_embed_dim, num_layers=num_decoder_blocks)
        self.final_layer = nn.Linear(decoder_embed_dim, vocab_size)
        self.enc_pos_encodings = self.create_positional_encoding(max_seq_length, encoder_embed_dim)
        self.dec_pos_encodings = self.create_positional_encoding(max_seq_length, decoder_embed_dim)
    
    def forward(self, encoder_input, decoder_input):
        enc_embs = self.encoder_projection(encoder_input)
        # enc_embs = self.encoder_emb(enc_embs)
        #  change when batching
        
        # add positional encodings
        seq_length = enc_embs.size(1)
        enc_pos_encodings = self.enc_pos_encodings[:seq_length, :].to(enc_embs.device)
        enc_embs += enc_pos_encodings
        encoder_out = self.encoder(enc_embs)
        
       
        dec_embs = self.decoder_emb(decoder_input)
        dec_seq_length = dec_embs.size(1)
        dec_pos_encodings = self.dec_pos_encodings[:dec_seq_length, :].to(dec_embs.device)
        dec_embs += dec_pos_encodings
        decoder_out = self.decoder(dec_embs, encoder_out)
        return self.final_layer(decoder_out)
    
    def create_positional_encoding(self, max_length, embed_dim):
        # Create positional encodings
        position = torch.arange(max_length, dtype=torch.float).unsqueeze(1)  # Shape: (max_length, 1)
        dim = torch.arange(embed_dim, dtype=torch.float).unsqueeze(0)  # Shape: (1, embed_dim)
        
        # Compute the positional encoding using sine and cosine functions
        angles = position / (10000 ** (dim / embed_dim))
        pos_encoding = torch.zeros(max_length, embed_dim)
        pos_encoding[:, 0::2] = torch.sin(angles[:, 0::2])  # Apply sine to even indices
        pos_encoding[:, 1::2] = torch.cos(angles[:, 1::2])  # Apply cosine to odd indices

        return pos_encoding