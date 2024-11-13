import torch
from torch.utils.data import Dataset, DataLoader

from transformer import Transformer

# Create a simple dataset
class SimpleDigitDataset(Dataset):
    def __init__(self, num_samples=100, seq_length=10):
        # Create random sequences of digits
        self.encoder_inputs = torch.randint(0, 10, (num_samples, seq_length))
        self.decoder_inputs = torch.randint(0, 10, (num_samples, seq_length))
        
    def __len__(self):
        return len(self.encoder_inputs)
    
    def __getitem__(self, idx):
        return self.encoder_inputs[idx], self.decoder_inputs[idx]

# Create dataset and dataloader
dataset = SimpleDigitDataset(num_samples=100, seq_length=10)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# Initialize model
model = Transformer(max_seq_length=10)
model.eval()

# Move to device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Get one batch
encoder_batch, decoder_batch = next(iter(dataloader))
encoder_batch = encoder_batch.to(device)
decoder_batch = decoder_batch.to(device)

# Forward pass
with torch.no_grad():
    output = model(encoder_batch, decoder_batch)

print("Batch shapes:")
print(f"Encoder input: {encoder_batch.shape}")  # Should be [32, 10]
print(f"Decoder input: {decoder_batch.shape}")  # Should be [32, 10]
print(f"Output: {output.shape}")  # Should be [32, 10, vocab_size]

# Print first sample in batch
print("\nFirst sample in batch:")
print(f"Encoder input: {encoder_batch[0]}")
print(f"Decoder input: {decoder_batch[0]}")
print(f"Output probabilities for first position: {torch.softmax(output[0][0], dim=0)}")