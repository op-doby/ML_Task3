# based on task2.py with modifications for validation set
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import random
from torch.utils.data import random_split, Subset

def set_seed(seed):
    """Set all random seeds for reproducibility"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)

def train_model(seed):
    """Train the model with a specific seed and return errors"""
    # Set the seed for reproducibility
    set_seed(seed)
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Hyperparameters
    input_size = 784
    hidden_size = 500
    num_classes = 10
    num_epochs = 5
    batch_size = 100
    learning_rate = 0.001
    
    # MNIST dataset
    full_train_dataset = torchvision.datasets.MNIST(root='./data/',
                                                   train=True,
                                                   transform=transforms.ToTensor(),
                                                   download=True)
    
    test_dataset = torchvision.datasets.MNIST(root='./data/',
                                              train=False,
                                              transform=transforms.ToTensor())
    
    # Split the training set into train and validation
    # Create a generator with the seed for reproducible splits
    g = torch.Generator()
    g.manual_seed(seed)
    
    # Get 10,000 random indices for validation set
    train_size = len(full_train_dataset) - 10000  # 50,000 for training
    val_size = 10000  # 10,000 for validation
    
    # Create train/validation split
    train_dataset, val_dataset = random_split(
        full_train_dataset, 
        [train_size, val_size], 
        generator=g
    )
    
    # Data loaders with fixed seed for shuffling
    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=g
    )
    
    val_loader = torch.utils.data.DataLoader(
        dataset=val_dataset,
        batch_size=batch_size,
        shuffle=False
    )
    
    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        shuffle=False
    )
    
    # Fully connected neural network
    class NeuralNet(nn.Module):
        def __init__(self, input_size, hidden_size, num_classes):
            super(NeuralNet, self).__init__()
            self.fc1 = nn.Linear(input_size, hidden_size) 
            self.relu = nn.ReLU()
            self.fc2 = nn.Linear(hidden_size, num_classes)  
    
        def forward(self, x):
            out = self.fc1(x)
            out = self.relu(out)
            out = self.fc2(out)
            return out
    
    # Initialize model with the seed
    model = NeuralNet(input_size, hidden_size, num_classes).to(device)
    
    # Initialize weights deterministically
    def init_weights(m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight, gain=1.0)
            torch.nn.init.zeros_(m.bias)
    
    model.apply(init_weights)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    # Lists to store errors
    train_errors = []
    val_errors = []
    test_errors = []
    
    # Variables to track best validation error
    best_val_error = float('inf')
    best_test_error = None
    best_epoch = -1
    
    # Training loop
    for epoch in range(num_epochs):
        # Train the model
        model.train()
        total_train_loss = 0
        for images, labels in train_loader:
            images = images.reshape(-1, input_size).to(device)
            labels = labels.to(device)
    
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
    
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    
            total_train_loss += loss.item() * labels.size(0)
    
        etr = total_train_loss / len(train_loader.dataset)
        train_errors.append(etr)
    
        # Evaluate on validation set
        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.reshape(-1, input_size).to(device)
                labels = labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                total_val_loss += loss.item() * labels.size(0)
    
        eva = total_val_loss / len(val_loader.dataset)
        val_errors.append(eva)
    
        # Evaluate on test set
        total_test_loss = 0
        with torch.no_grad():
            for images, labels in test_loader:
                images = images.reshape(-1, input_size).to(device)
                labels = labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                total_test_loss += loss.item() * labels.size(0)
    
        ete = total_test_loss / len(test_loader.dataset)
        test_errors.append(ete)
    
        # Check if this is the best validation error
        if eva < best_val_error:
            best_val_error = eva
            best_test_error = ete
            best_epoch = epoch + 1
    
        print(f'Seed {seed}, Epoch [{epoch + 1}/{num_epochs}], Train Error: {etr:.4f}, '
              f'Validation Error: {eva:.4f}, Test Error: {ete:.4f}')
    
    print(f'Seed {seed}, Best Validation Error: {best_val_error:.4f} (Epoch {best_epoch}), '
          f'Corresponding Test Error: {best_test_error:.4f}')
    
    return {
        'train_errors': train_errors,
        'val_errors': val_errors,
        'test_errors': test_errors,
        'best_val_error': best_val_error,
        'best_test_error': best_test_error,
        'best_epoch': best_epoch
    }

# Run the model with 5 different seeds
seeds = [42, 123, 456, 789, 1024]
results = []

for seed in seeds:
    print(f"\nRunning with seed {seed}")
    result = train_model(seed)
    results.append(result)

# Calculate statistics for best validation errors and corresponding test errors
best_val_errors = [result['best_val_error'] for result in results]
corresponding_test_errors = [result['best_test_error'] for result in results]

mean_val_error = np.mean(best_val_errors)
std_val_error = np.std(best_val_errors)
mean_test_error = np.mean(corresponding_test_errors)
std_test_error = np.std(corresponding_test_errors)

print("\nResults Summary:")
print(f"Mean of best validation errors: {mean_val_error:.4f}")
print(f"Standard deviation of best validation errors: {std_val_error:.4f}")
print(f"Mean of corresponding test errors: {mean_test_error:.4f}")
print(f"Standard deviation of corresponding test errors: {std_test_error:.4f}")

# Plot validation and test errors for all seeds
plt.figure(figsize=(15, 10))

# Plot validation errors
plt.subplot(2, 1, 1)
for i, result in enumerate(results):
    plt.plot(range(1, len(result['val_errors']) + 1), result['val_errors'], 
             marker='o', label=f'Seed {seeds[i]}')
    # Mark the best validation error point
    best_epoch = result['best_epoch']
    plt.scatter(best_epoch, result['best_val_error'], 
                marker='*', s=200, c='red', 
                label=f'Best for seed {seeds[i]}' if i == 0 else "")

plt.xlabel('Epoch')
plt.ylabel('Validation Error')
plt.title('Validation Errors During Training with Different Seeds')
plt.legend()
plt.grid(True)

# Plot test errors
plt.subplot(2, 1, 2)
for i, result in enumerate(results):
    plt.plot(range(1, len(result['test_errors']) + 1), result['test_errors'], 
             marker='o', label=f'Seed {seeds[i]}')
    # Mark the test error corresponding to best validation error
    best_epoch = result['best_epoch']
    plt.scatter(best_epoch, result['best_test_error'], 
                marker='*', s=200, c='red', 
                label=f'Corresponding to best val for seed {seeds[i]}' if i == 0 else "")

plt.xlabel('Epoch')
plt.ylabel('Test Error')
plt.title('Test Errors During Training with Different Seeds')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig('validation_test_errors.png')
plt.show()

# Create a table of best validation errors and corresponding test errors
print("\nBest Validation Errors and Corresponding Test Errors:")
print("-" * 70)
print(f"{'Seed':<10} {'Best Val Error':<20} {'Best Epoch':<15} {'Corresponding Test Error':<25}")
print("-" * 70)
for i, result in enumerate(results):
    print(f"{seeds[i]:<10} {result['best_val_error']:<20.4f} {result['best_epoch']:<15} {result['best_test_error']:<25.4f}")
print("-" * 70)