# based on task1.py with modifications for seed control
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import random

def set_seed(seed):
    """Set all random seeds for reproducibility"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)

def train_model(seed):
    """Train the model with a specific seed and return test errors"""
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
    train_dataset = torchvision.datasets.MNIST(root='./data/',
                                               train=True,
                                               transform=transforms.ToTensor(),
                                               download=True)
    
    test_dataset = torchvision.datasets.MNIST(root='./data/',
                                              train=False,
                                              transform=transforms.ToTensor())
    
    # Data loaders with fixed seed for shuffling
    g = torch.Generator()
    g.manual_seed(seed)
    
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               generator=g)
    
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=batch_size,
                                              shuffle=False)
    
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
    test_errors = []
    
    # Training loop
    for epoch in range(num_epochs):
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
    
        print(f'Seed {seed}, Epoch [{epoch+1}/{num_epochs}], Train Error: {etr:.4f}, Test Error: {ete:.4f}')
    
    # Report the final test error
    final_test_error = test_errors[-1]
    print(f'Seed {seed}, Final Test Error: {final_test_error:.4f}')
    
    return test_errors, final_test_error

# Run the model with 5 different seeds
seeds = [42, 123, 456, 789, 1024]
all_test_errors = []
final_test_errors = []

for seed in seeds:
    print(f"\nRunning with seed {seed}")
    test_errors, final_error = train_model(seed)
    all_test_errors.append(test_errors)
    final_test_errors.append(final_error)

# Calculate statistics
mean_final_error = np.mean(final_test_errors)
std_final_error = np.std(final_test_errors)

print("\nResults Summary:")
print(f"Mean of final test errors: {mean_final_error:.4f}")
print(f"Standard deviation of final test errors: {std_final_error:.4f}")

# Plot test errors for all seeds
plt.figure(figsize=(10, 6))
for i, errors in enumerate(all_test_errors):
    plt.plot(range(1, len(errors) + 1), errors, marker='o', label=f'Seed {seeds[i]}')

plt.xlabel('Epoch')
plt.ylabel('Test Error')
plt.title('Test Errors During Training with Different Seeds')
plt.legend()
plt.grid(True)
plt.savefig('test_errors_different_seeds.png')
plt.show()

# Assess robustness
cv = std_final_error / mean_final_error  # Coefficient of variation
print(f"\nCoefficient of variation: {cv:.4f}")

if cv < 0.05:
    robustness = "The model is very robust to the choice of seed (CV < 5%)"
elif cv < 0.10:
    robustness = "The model is reasonably robust to the choice of seed (CV < 10%)"
else:
    robustness = "The model shows significant variability with different seeds (CV > 10%)"

print(robustness)