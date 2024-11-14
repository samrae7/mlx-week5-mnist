import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import wandb

from dataset import START_TOKEN, CombinedMNIST
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

def train(model, num_epochs, save_path='model_weights.pth'):
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
    torch.save(model.state_dict(), save_path)
    print(f"Model weights saved to {save_path}")
    wandb.finish()

def validate(model, model_path='model_weights.pth'):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    start_tensor = torch.tensor([[START_TOKEN]]).to(device)

    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.to(device)
    model.eval()

    encoder_batch, decoder_batch, label_batch = next(iter(dataloader))
    encoder_batch = encoder_batch.to(device)
    decoder_batch = decoder_batch.to(device)
    label_batch = label_batch.to(device)

    label = label_batch[:1, :]
    predicted_sequence = [START_TOKEN]
    decoder_input = start_tensor
    # get batch of 1
    enc_in = encoder_batch[:1, :]
    with torch.no_grad():
        # logits = model(enc_in, start_tensor)
        for _ in range(encoder_batch.size(1)):
            logits = model(enc_in, decoder_input)
            # only want probs from most recent logit in sequence
            probs = nn.functional.softmax(logits[:, -1], dim=-1)  # Get last token predictions
            predicted = torch.argmax(probs, dim=-1)
            predicted_sequence.append(predicted.item())

            decoder_input = torch.tensor([predicted_sequence]).to(device)

    result = ''.join(map(str, predicted_sequence[1:]))  # Exclude start token
    return label, result
    

# train(model, 2)
label, result = validate(model)
print(f"label: {label}, result: {result}")
