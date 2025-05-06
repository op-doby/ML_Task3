import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE
import random
import os

# Function to set random seeds for reproducibility
def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Define the neural network model
class SimpleNN(nn.Module):
    def __init__(self, input_size=784, hidden_size=128, output_size=10):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.hidden_size = hidden_size
    
    def forward(self, x):
        x = x.view(-1, 784)  # Flatten the input
        z = F.relu(self.fc1(x))  # Hidden features
        out = self.fc2(z)
        return out, z
    
    def get_hidden_features(self, x):
        x = x.view(-1, 784)
        z = F.relu(self.fc1(x))
        return z

# Function to train the model
def train(model, train_loader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for inputs, targets in train_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        
        optimizer.zero_grad()
        outputs, _ = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
    
    train_loss = running_loss / len(train_loader)
    train_acc = 100. * correct / total
    train_error = 1 - (correct / total)
    
    return train_loss, train_acc, train_error

# Function to evaluate the model
def evaluate(model, data_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_targets = []
    misclassified_images = []
    misclassified_preds = []
    misclassified_targets = []
    
    with torch.no_grad():
        for inputs, targets in data_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs, _ = model(inputs)
            loss = criterion(outputs, targets)
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
            
            # Store misclassified images
            mask = ~predicted.eq(targets)
            if mask.any():
                misclassified_images.append(inputs[mask].cpu())
                misclassified_preds.append(predicted[mask].cpu())
                misclassified_targets.append(targets[mask].cpu())
    
    eval_loss = running_loss / len(data_loader)
    eval_acc = 100. * correct / total
    eval_error = 1 - (correct / total)
    
    misclassified = {
        'images': misclassified_images,
        'preds': misclassified_preds,
        'targets': misclassified_targets
    }
    
    return eval_loss, eval_acc, eval_error, misclassified

# Function to plot training and testing errors
def plot_errors(train_errors, test_errors, title="Training and Testing Errors"):
    plt.figure(figsize=(10, 6))
    plt.plot(train_errors, label='Train Error')
    plt.plot(test_errors, label='Test Error')
    plt.xlabel('Epochs')
    plt.ylabel('Error')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    return plt.gcf()

# Function to plot misclassified images
def plot_misclassified(misclassified, num_images=10):
    images = torch.cat(misclassified['images'])
    preds = torch.cat(misclassified['preds'])
    targets = torch.cat(misclassified['targets'])
    
    n = min(num_images, len(images))
    fig, axes = plt.subplots(2, 5, figsize=(12, 5))
    axes = axes.flatten()
    
    for i in range(n):
        axes[i].imshow(images[i].squeeze(), cmap='gray')
        axes[i].set_title(f'Pred: {preds[i]}, True: {targets[i]}')
        axes[i].axis('off')
    
    plt.tight_layout()
    return fig

# Function to extract features and visualize with t-SNE
def visualize_tsne(model, data_loader, device, is_features=True):
    model.eval()
    features = []
    labels = []
    
    with torch.no_grad():
        for inputs, targets in data_loader:
            inputs = inputs.to(device)
            if is_features:
                # Get hidden features
                batch_features = model.get_hidden_features(inputs).cpu().numpy()
            else:
                # Use raw inputs
                batch_features = inputs.view(-1, 784).cpu().numpy()
            
            features.append(batch_features)
            labels.append(targets.numpy())
    
    features = np.vstack(features)
    labels = np.concatenate(labels)
    
    # Use a subset for t-SNE to make it faster
    subset_size = min(5000, len(features))
    indices = np.random.choice(len(features), subset_size, replace=False)
    subset_features = features[indices]
    subset_labels = labels[indices]
    
    # Apply t-SNE
    tsne = TSNE(n_components=2, random_state=42)
    tsne_results = tsne.fit_transform(subset_features)
    
    # Plot t-SNE results
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(tsne_results[:, 0], tsne_results[:, 1], c=subset_labels, 
                         cmap='tab10', alpha=0.6, s=10)
    plt.colorbar(scatter, ticks=range(10))
    if is_features:
        plt.title('t-SNE Visualization of Hidden Features')
    else:
        plt.title('t-SNE Visualization of Raw Inputs')
    plt.xlabel('t-SNE Dimension 1')
    plt.ylabel('t-SNE Dimension 2')
    
    return plt.gcf()

# Create directory for saving results
os.makedirs('results', exist_ok=True)

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Define transformations
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# Task 1: MNIST Classification
def task1():
    print("\nTask 1: MNIST Classification")
    
    # Set seed for reproducibility
    set_seed(42)
    
    # Load MNIST dataset
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('./data', train=False, download=True, transform=transform)
    
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)
    
    # Initialize model, loss function, and optimizer
    model = SimpleNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Training loop
    num_epochs = 10
    train_errors = []
    test_errors = []
    
    for epoch in range(num_epochs):
        train_loss, train_acc, train_error = train(model, train_loader, optimizer, criterion, device)
        test_loss, test_acc, test_error, misclassified = evaluate(model, test_loader, criterion, device)
        
        train_errors.append(train_error)
        test_errors.append(test_error)
        
        print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, '
              f'Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%')
    
    # Plot training and testing errors
    error_fig = plot_errors(train_errors, test_errors)
    error_fig.savefig('results/task1_errors.png')
    
    # Plot misclassified images
    if misclassified['images']:
        misc_fig = plot_misclassified(misclassified)
        misc_fig.savefig('results/task1_misclassified.png')
    
    print(f"Final test error: {test_error:.4f}")
    return model, test_error

# Task 2: Mitigate Pseudorandomness
def task2():
    print("\nTask 2: Mitigate Pseudorandomness")
    
    seeds = [42, 123, 456, 789, 101]
    num_epochs = 10
    all_test_errors = []
    
    plt.figure(figsize=(10, 6))
    
    for i, seed in enumerate(seeds):
        print(f"Running with seed {seed}")
        set_seed(seed)
        
        # Load MNIST dataset
        train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
        test_dataset = datasets.MNIST('./data', train=False, download=True, transform=transform)
        
        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)
        
        # Initialize model, loss function, and optimizer
        model = SimpleNN().to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        
        # Training loop
        test_errors = []
        
        for epoch in range(num_epochs):
            train_loss, train_acc, train_error = train(model, train_loader, optimizer, criterion, device)
            test_loss, test_acc, test_error, _ = evaluate(model, test_loader, criterion, device)
            test_errors.append(test_error)
            
            print(f'Seed {seed}, Epoch {epoch+1}/{num_epochs}, Test Error: {test_error:.4f}')
        
        plt.plot(test_errors, label=f'Seed {seed}')
        all_test_errors.append(test_errors[-1])
    
    plt.xlabel('Epochs')
    plt.ylabel('Test Error')
    plt.title('Test Errors for Different Seeds')
    plt.legend()
    plt.grid(True)
    plt.savefig('results/task2_seed_errors.png')
    
    mean_error = np.mean(all_test_errors)
    std_error = np.std(all_test_errors)
    
    print(f"Mean final test error: {mean_error:.4f}")
    print(f"Standard deviation of final test error: {std_error:.4f}")
    
    return mean_error, std_error

# Task 3: Validation Dataset
def task3():
    print("\nTask 3: Validation Dataset")
    
    seeds = [42, 123, 456, 789, 101]
    num_epochs = 10
    results = []
    
    for i, seed in enumerate(seeds):
        print(f"Running with seed {seed}")
        set_seed(seed)
        
        # Load MNIST dataset
        full_train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
        test_dataset = datasets.MNIST('./data', train=False, download=True, transform=transform)
        
        # Split into train and validation
        train_size = len(full_train_dataset) - 10000
        val_size = 10000
        train_dataset, val_dataset = random_split(full_train_dataset, [train_size, val_size])
        
        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=1000, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)
        
        # Initialize model, loss function, and optimizer
        model = SimpleNN().to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        
        # Training loop
        val_errors = []
        test_errors = []
        
        for epoch in range(num_epochs):
            train_loss, train_acc, train_error = train(model, train_loader, optimizer, criterion, device)
            val_loss, val_acc, val_error, _ = evaluate(model, val_loader, criterion, device)
            test_loss, test_acc, test_error, _ = evaluate(model, test_loader, criterion, device)
            
            val_errors.append(val_error)
            test_errors.append(test_error)
            
            print(f'Seed {seed}, Epoch {epoch+1}/{num_epochs}, Val Error: {val_error:.4f}, Test Error: {test_error:.4f}')
        
        # Find the epoch with minimum validation error
        min_val_idx = np.argmin(val_errors)
        min_val_error = val_errors[min_val_idx]
        corresponding_test_error = test_errors[min_val_idx]
        
        print(f"Seed {seed}: Min val error: {min_val_error:.4f} at epoch {min_val_idx+1}, "
              f"Corresponding test error: {corresponding_test_error:.4f}")
        
        results.append({
            'seed': seed,
            'min_val_error': min_val_error,
            'corresponding_test_error': corresponding_test_error,
            'epoch': min_val_idx + 1
        })
    
    # Print summary
    print("\nSummary of results:")
    for res in results:
        print(f"Seed {res['seed']}: Min val error: {res['min_val_error']:.4f} at epoch {res['epoch']}, "
              f"Corresponding test error: {res['corresponding_test_error']:.4f}")
    
    return results

# Task 4: Grid Search
def task4():
    print("\nTask 4: Grid Search")
    
    # Define hyperparameter grid
    hidden_sizes = [64, 128, 256]
    batch_sizes = [32, 64, 128]
    learning_rates = [0.0001, 0.001, 0.01]
    
    # Set seed for reproducibility
    set_seed(42)
    
    # Load MNIST dataset
    full_train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('./data', train=False, download=True, transform=transform)
    
    # Split into train and validation
    train_size = len(full_train_dataset) - 10000
    val_size = 10000
    train_dataset, val_dataset = random_split(full_train_dataset, [train_size, val_size])
    
    test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)
    
    # Grid search
    results = []
    best_val_error = float('inf')
    best_params = {}
    
    for hidden_size in hidden_sizes:
        for batch_size in batch_sizes:
            for lr in learning_rates:
                print(f"\nTraining with hidden_size={hidden_size}, batch_size={batch_size}, lr={lr}")
                
                train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
                val_loader = DataLoader(val_dataset, batch_size=1000, shuffle=False)
                
                # Initialize model, loss function, and optimizer
                model = SimpleNN(hidden_size=hidden_size).to(device)
                criterion = nn.CrossEntropyLoss()
                optimizer = optim.Adam(model.parameters(), lr=lr)
                
                # Training loop
                num_epochs = 5  # Reduced epochs for grid search
                val_errors = []
                test_errors = []
                
                for epoch in range(num_epochs):
                    train_loss, train_acc, train_error = train(model, train_loader, optimizer, criterion, device)
                    val_loss, val_acc, val_error, _ = evaluate(model, val_loader, criterion, device)
                    test_loss, test_acc, test_error, _ = evaluate(model, test_loader, criterion, device)
                    
                    val_errors.append(val_error)
                    test_errors.append(test_error)
                
                # Find the epoch with minimum validation error
                min_val_idx = np.argmin(val_errors)
                min_val_error = val_errors[min_val_idx]
                corresponding_test_error = test_errors[min_val_idx]
                
                result = {
                    'hidden_size': hidden_size,
                    'batch_size': batch_size,
                    'learning_rate': lr,
                    'min_val_error': min_val_error,
                    'corresponding_test_error': corresponding_test_error,
                    'epoch': min_val_idx + 1
                }
                
                results.append(result)
                
                print(f"Min val error: {min_val_error:.4f} at epoch {min_val_idx+1}, "
                      f"Corresponding test error: {corresponding_test_error:.4f}")
                
                if min_val_error < best_val_error:
                    best_val_error = min_val_error
                    best_params = {
                        'hidden_size': hidden_size,
                        'batch_size': batch_size,
                        'learning_rate': lr
                    }
    
    # Print results table
    print("\nGrid Search Results:")
    print("hidden_size | batch_size | learning_rate | val_error | test_error")
    print("-" * 70)
    
    for res in results:
        print(f"{res['hidden_size']:11d} | {res['batch_size']:10d} | {res['learning_rate']:.6f} | {res['min_val_error']:.6f} | {res['corresponding_test_error']:.6f}")
    
    print(f"\nBest parameters: hidden_size={best_params['hidden_size']}, "
          f"batch_size={best_params['batch_size']}, learning_rate={best_params['learning_rate']}")
    
    return results, best_params

# Task 5: Feature Analysis
def task5():
    print("\nTask 5: Feature Analysis")
    
    # Set seed for reproducibility
    set_seed(42)
    
    # Load MNIST dataset
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=1000, shuffle=False)
    
    # Initialize model with best parameters from Task 4
    model = SimpleNN(hidden_size=128).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Train the model
    num_epochs = 5
    for epoch in range(num_epochs):
        train_loss, train_acc, train_error = train(model, train_loader, optimizer, criterion, device)
        print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
    
    # Visualize hidden features using t-SNE
    hidden_features_fig = visualize_tsne(model, train_loader, device, is_features=True)
    hidden_features_fig.savefig('results/task5_hidden_features_tsne.png')
    
    # Visualize raw inputs using t-SNE
    raw_inputs_fig = visualize_tsne(model, train_loader, device, is_features=False)
    raw_inputs_fig.savefig('results/task5_raw_inputs_tsne.png')
    
    return model

if __name__ == "__main__":
    # Execute all tasks
    model1, test_error1 = task1()
    mean_error, std_error = task2()
    task3_results = task3()
    task4_results, best_params = task4()
    model5 = task5()
    
    print("\nAll tasks completed successfully!")
