import torch
import torchvision
import random
from PIL import Image

class CombinedMNIST(torch.utils.data.Dataset):
    def __init__(self):
        super().__init__()
        # Transform to convert MNIST images to tensors
        self.tf = torchvision.transforms.ToTensor()
        # Load MNIST dataset
        self.ds = torchvision.datasets.MNIST(root='.', train=True, download=True)
        # Store length of dataset
        self.ln = len(self.ds)
    
    def __len__(self):
        return self.ln
    
    def __getitem__(self, idx):
        idx = random.sample(range(self.ln), 4)
        quadrant_tensors = []
        labels = []
        
        for i in idx:
            x, y = self.ds[i]
            # Convert PIL image to tensor and flatten
            x_tensor = self.tf(x).view(-1)  # Will be shape [784] (28*28)
            quadrant_tensors.append(x_tensor)
            labels.append(y)
        
        # Stack the quadrant tensors into a single tensor
        quadrants = torch.stack(quadrant_tensors)  # Shape will be [4, 784]
        labels = torch.tensor(labels)  # Shape will be [4]
        
        return quadrants, labels

# Test it
ds = CombinedMNIST()
quadrants, labels = ds[0]
print("Quadrants shape:", quadrants.shape)  # Should be [4, 784]
print("Labels shape:", labels.shape)      # Should be [4]
