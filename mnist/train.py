import torch
from torch.utils.data import Dataset, DataLoader

from dataset import CombinedMNIST
from transformer import Transformer

dataset = CombinedMNIST()
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

model = Transformer()

def test_pass_through(model):
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    encoder_batch, decoder_batch = next(iter(dataloader))
    encoder_batch = encoder_batch.to(device)
    decoder_batch = decoder_batch.to(device)

    with torch.no_grad():
        output = model(encoder_batch, decoder_batch)

    print("Batch shapes:")
    print(f"Encoder input: {encoder_batch.shape}")  # Should be [batch_size, sequence_length, 28 x 28]
    print(f"Decoder input: {decoder_batch.shape}")  # Should be [batch_size, seq_length]
    print(f"Output: {output.shape}")  # Should be [batch_size, seq_length, vocab_size]

    print("\nFirst sample in batch:")
    print(f"Encoder input: {encoder_batch[0]}")
    print(f"Decoder input: {decoder_batch[0]}")
    print(f"Output probabilities for first position: {torch.softmax(output[0][0], dim=0)}")

test_pass_through(model)

# def train(model):
#     model.train
#     for 