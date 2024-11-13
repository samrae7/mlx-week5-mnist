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
        # Randomly sample 4 indices from the dataset
        idx = random.sample(range(self.ln), 4)
        store = []
        label = []
        
        # Get images and labels for each index
        for i in idx:
            x, y = self.ds[i]
            store.append(x)
            label.append(y)
        
        # Create new 56x56 image
        img = Image.new('L', (56, 56))
        # Paste images in quadrants
        img.paste(store[0], (0, 0))
        img.paste(store[1], (28, 0))
        img.paste(store[2], (0, 28))
        img.paste(store[3], (28, 28))
        
        # Convert the combined image to a tensor
        img_tensor = self.tf(img)
        
        return img_tensor, torch.tensor(label)
    
ds = CombinedMNIST()
img, label = ds[0]  # Get first item
print(label)
