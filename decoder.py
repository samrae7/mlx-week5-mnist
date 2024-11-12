import torch
import torch.nn as nn

from transformer_block import TransformerBlock

tokens = {"<s>": 0, "a": 1, "b": 2, "c": 3, '<e>': 4}
chars = {value: key for key, value in tokens.items()}

def tokenise(char):
    return tokens[char]

class Decoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_layers):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, embed_dim)
        self.num_layers = num_layers
        # transformerBlock
        # this should take in seq_length * emb_dim, and output the same

        self.block = TransformerBlock(embed_dim)
        # should take in seq_length * emb_dim and output seq_length * vocab size,
        # so it should be emb_dim * vocab_size
        self.linear = nn.Linear(embed_dim, vocab_size)
        self.pos_encodings = self.create_positional_encoding(100, embed_dim)

    
    def forward(self, inputs):
        input_embs = self.emb(inputs)
        seq_length = input_embs.size(0)
        pos_encodings = self.pos_encodings[:seq_length, :].to(input_embs.device)
        input_embs += pos_encodings
        for i in range(self.num_layers):
            out = self.block(input_embs)
        out = self.linear(out)
        # removed softmax because CEL does that
        # out = nn.functional.softmax(out, dim=-1)
        # TODO? do loss here?
        return out
    
    #  TODO: This is from chatGPT - revise and understand
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

if __name__ == "__main__":
    corpus = "aabbcc"
    input = "bb"
    tokenised_input = [tokenise(char) for char in input]
    input_tensor = torch.tensor(tokenised_input)
    decoder = Decoder(3,8, num_layers=4)
    probs = decoder(input_tensor)
    predicted_tokens = torch.argmax(probs, dim=-1)
    print(predicted_tokens)
    items = [token.item() for token in predicted_tokens.squeeze()]
    output_tokens = [chars[token] for token in items]
    print(output_tokens)
