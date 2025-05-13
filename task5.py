# Feature analysis using t-SNE visualization
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE
import random

def set_seed(seed):
    """Set all random seeds for reproducibility"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)

# Set seed for reproducibility
set_seed(42)

# Setup device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyperparameters
input_size = 784
hidden_size = 500
num_classes = 10
batch_size = 100
learning_rate = 0.001
num_epochs = 5

# MNIST dataset
train_dataset = torchvision.datasets.MNIST(root='./data/',
                                          train=True,
                                          transform=transforms.ToTensor(),
                                          download=True)

# Data loader
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                          batch_size=batch_size,
                                          shuffle=True)

# Fully connected neural network with feature extraction capability
class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size) 
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)  
    
    def forward(self, x):
        out = self.fc1(x)
        hidden_features = self.relu(out)
        out = self.fc2(hidden_features)
        return out
    
    def get_hidden_features(self, x):
        """Extract hidden features z_i = σ(W^(1)T x_i + b^(1))"""
        out = self.fc1(x)
        hidden_features = self.relu(out)
        return hidden_features

# Initialize model
model = NeuralNet(input_size, hidden_size, num_classes).to(device)

# Initialize weights deterministically
def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=1.0)
        torch.nn.init.zeros_(m.bias)

model.apply(init_weights)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Train the model
print("Training the model...")
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        # Reshape images to (batch_size, input_size)
        images = images.reshape(-1, input_size).to(device)
        labels = labels.to(device)
        
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

print("Training complete!")

# Function to extract features and raw inputs
def extract_features(model, data_loader, num_samples=6000):
    """Extract hidden features and raw inputs for visualization"""
    model.eval()
    hidden_features = []
    raw_inputs = []
    labels_list = []
    
    with torch.no_grad():
        for images, labels in data_loader:
            # Only take a subset of samples to avoid memory issues with t-SNE
            if len(hidden_features) * batch_size >= num_samples:
                break
                
            # Reshape images and move to device
            flat_images = images.reshape(-1, input_size).to(device)
            
            # Get hidden features
            features = model.get_hidden_features(flat_images)
            
            # Store data
            hidden_features.append(features.cpu().numpy())
            raw_inputs.append(flat_images.cpu().numpy())
            labels_list.append(labels.numpy())
    
    # Concatenate batches
    hidden_features = np.vstack(hidden_features)[:num_samples]
    raw_inputs = np.vstack(raw_inputs)[:num_samples]
    labels = np.concatenate(labels_list)[:num_samples]
    
    return hidden_features, raw_inputs, labels

# Extract features from a subset of the training data for visualization
print("Extracting features...")
hidden_features, raw_inputs, labels = extract_features(model, train_loader, num_samples=2000)

# Apply t-SNE to hidden features
print("Applying t-SNE to hidden features...")
tsne_hidden = TSNE(n_components=2, random_state=42, perplexity=30)
hidden_2d = tsne_hidden.fit_transform(hidden_features)

# Apply t-SNE to raw inputs
print("Applying t-SNE to raw inputs...")
tsne_raw = TSNE(n_components=2, random_state=42, perplexity=30)
raw_2d = tsne_raw.fit_transform(raw_inputs)

# Plot t-SNE visualizations
plt.figure(figsize=(20, 10))

# Define colors for each digit
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
          '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

# Plot hidden features
plt.subplot(1, 2, 1)
for i in range(10):  # For each digit
    idx = labels == i
    plt.scatter(hidden_2d[idx, 0], hidden_2d[idx, 1], c=colors[i], label=str(i), alpha=0.6)

plt.title('t-SNE of Hidden Features (z_i)')
plt.legend()
plt.grid(True)

# Plot raw inputs
plt.subplot(1, 2, 2)
for i in range(10):  # For each digit
    idx = labels == i
    plt.scatter(raw_2d[idx, 0], raw_2d[idx, 1], c=colors[i], label=str(i), alpha=0.6)

plt.title('t-SNE of Raw Inputs (x_i)')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig('tsne_comparison.png', dpi=300)
plt.show()

# Analysis of the results
print("\nAnalysis of t-SNE Visualizations:")
print("-" * 50)
print("Hidden Features (z_i):")
print("- These represent the activations after the first layer and ReLU.")
print("- They show how the model has transformed the input data.")
print("- The separation between digit clusters indicates what the model has learned.")
print("\nRaw Inputs (x_i):")
print("- These are the flattened pixel values of the original images.")
print("- They show the natural structure and similarity between different digits.")
print("\nComparison:")
print("- The hidden features visualization shows how the model has learned to separate")
print("  different digits in the feature space.")
print("- Digits that are visually similar (like 3 and 8, or 4 and 9) might be")
print("  closer in the raw input space but better separated in the hidden feature space.")
print("- The clustering pattern differences reveal what transformations the model")
print("  has learned to make classification easier.")
print("-" * 50)