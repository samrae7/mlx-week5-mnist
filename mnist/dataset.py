import torch
import torchvision
import random
from PIL import Image

START_TOKEN = 10
END_TOKEN = 11

class CombinedMNIST(torch.utils.data.Dataset):
    def __init__(self):
        super().__init__()
        # Transform to convert MNIST images to tensors
        self.tf = torchvision.transforms.ToTensor()
        # Load MNIST dataset
        self.ds = torchvision.datasets.MNIST(root='.', train=True, download=True)
        # Store length of dataset
        self.ln = len(self.ds)
        self.sample_indices = [random.sample(range(self.ln), 4) for _ in range(self.ln)]
    
    def __len__(self):
        return self.ln
    
    def __getitem__(self, idx):
        indices =  self.sample_indices[idx]
        quadrant_tensors = []
        labels = []
        
        for i in indices:
            x, y = self.ds[i]
            # Convert PIL image to tensor and flatten
            x_tensor = self.tf(x).view(-1)  # Will be shape [784] (28*28)
            quadrant_tensors.append(x_tensor)
            labels.append(y)
        
        # Stack the quadrant tensors into a single tensor
        encoder_in = torch.stack(quadrant_tensors)  # Shape will be [4, 784]
        labels = torch.tensor(labels)  # Shape will be [4]
        start_token = torch.tensor([START_TOKEN])
        end_token = torch.tensor([END_TOKEN])
        decoder_in = torch.cat((start_token, labels))
        target = torch.cat((labels, end_token))
        
        return encoder_in, decoder_in, target

# Test it
ds = CombinedMNIST()
quadrants, enc_in, labels = ds[0]
print("Quadrants shape:", quadrants.shape)  # Should be [4, 784]
print("Labels shape:", labels.shape)      # Should be [4]
