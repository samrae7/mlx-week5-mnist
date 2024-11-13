import torch.nn as nn

class CrossAttention(nn.Module):
    def __init__(self, encoder_embed_dim, decoder_embed_dim, d_k):
        super().__init__()
        self.d_k = d_k
        self.linear_v = nn.Linear(encoder_embed_dim, decoder_embed_dim)
        self.linear_k = nn.Linear(encoder_embed_dim, d_k)
        self.linear_q = nn.Linear(decoder_embed_dim, d_k)

    def forward(self, encoder_input, decoder_input):
        # make k, q, v
            # get k and v from encoder input
            #  get q from decoder input. Dimensions should be (input_seq_length, decoder_input_embedding  
        v = self.linear_v(encoder_input)
        k = self.linear_k(encoder_input)
        q = self.linear_q(decoder_input)
        scaling_factor = self.d_k**0.5
        attention_scores = q @ k.transpose(-2,-1) / scaling_factor
        attention_weights = nn.functional.softmax(attention_scores, dim=-1)
        
        #  softmax here
        context_embs = attention_weights @ v
        return context_embs