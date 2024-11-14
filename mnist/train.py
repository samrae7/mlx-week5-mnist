import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import wandb

from dataset import CombinedMNIST
from transformer import Transformer

dataset = CombinedMNIST()
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# 0-9 + start token
vocab_size = 11
model = Transformer(vocab_size=vocab_size)

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

# test_pass_through(model)

def train(model, num_epochs):
    criterion = nn.CrossEntropyLoss(size_average=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.train()
    optim = torch.optim.Adam(model.parameters(), lr=0.001)
    wandb.init('MNIST')
    for epoch in range(num_epochs):
        epoch_loss = 0
        num_batches = 0
        for enc_in, dec_in, target in tqdm(dataloader, total=len(dataloader)):
            optim.zero_grad()
            logits = model(enc_in, dec_in)
            # convert to [batch_size * seq_length, vocab_size]
            logits = logits.view(-1, vocab_size)
            # change from [batch_size, seq_length] to [batch_size * seq_length]
            target = target.view(-1)
            loss = criterion(logits, target)
            wandb.log({'batch_loss': loss})
            loss.backward()
            optim.step()
            epoch_loss += loss.item()
            num_batches += 1
        avg_epoch_loss = epoch_loss / num_batches
        print(f"Epoch: {epoch + 1}, avg_loss:{avg_epoch_loss:.4f}")
        wandb.log({'epoch': epoch, 'avg_loss': avg_epoch_loss})
    wandb.finish()
    

train(model, 2)
