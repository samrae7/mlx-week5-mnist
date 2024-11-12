import torch
import torch.nn as nn

from transformer_block import TransformerBlock

tokens = {"<s>": 0, "a": 1, "b": 2, "c": 3}
chars = {value: key for key, value in tokens.items()}

def tokenise(char):
    return tokens[char]

class Decoder(nn.Module):
    def __init__(self, vocab_size, embed_dim):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, embed_dim)
        # transformerBlock
        # this should take in seq_length * emb_dim, and output the same
        self.block = TransformerBlock(embed_dim)
        # should take in seq_length * emb_dim and output seq_length * vocab size,
        # so it should be emb_dim * vocab_size
        self.linear = nn.Linear(embed_dim, vocab_size)
    
    def forward(self, inputs):
        input_embs = self.emb(inputs)
        out = self.block(input_embs)
        out = self.linear(out)
        out = nn.functional.softmax(out, dim=-1)
        # TODO? do loss here?
        return out

if __name__ == "__main__":
    corpus = "aabbcc"
    input = "bb"
    tokenised_input = [tokenise(char) for char in input]
    input_tensor = torch.tensor(tokenised_input)
    decoder = Decoder(3,8)
    probs = decoder(input_tensor)
    predicted_tokens = torch.argmax(probs, dim=-1)
    print(predicted_tokens)
    items = [token.item() for token in predicted_tokens.squeeze()]
    output_tokens = [chars[token] for token in items]
    print(output_tokens)
