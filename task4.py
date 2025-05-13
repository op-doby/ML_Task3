# Grid search for hyperparameter optimization
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import random
from torch.utils.data import random_split

def set_seed(seed):
    """Set all random seeds for reproducibility"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)

def train_model(hidden_size, batch_size, learning_rate, seed=42):
    """Train the model with specific hyperparameters and return errors"""
    # Set the seed for reproducibility
    set_seed(seed)
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Fixed hyperparameters
    input_size = 784
    num_classes = 10
    num_epochs = 5
    
    # MNIST dataset
    full_train_dataset = torchvision.datasets.MNIST(root='./data/',
                                                   train=True,
                                                   transform=transforms.ToTensor(),
                                                   download=True)
    
    test_dataset = torchvision.datasets.MNIST(root='./data/',
                                              train=False,
                                              transform=transforms.ToTensor())
    
    # Split the training set into train and validation
    g = torch.Generator()
    g.manual_seed(seed)
    
    train_size = len(full_train_dataset) - 10000  # 50,000 for training
    val_size = 10000  # 10,000 for validation
    
    train_dataset, val_dataset = random_split(
        full_train_dataset, 
        [train_size, val_size], 
        generator=g
    )
    
    # Data loaders
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
    
    # Initialize model
    model = NeuralNet(input_size, hidden_size, num_classes).to(device)
    
    # Initialize weights deterministically
    def init_weights(m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight, gain=1.0)
            torch.nn.init.zeros_(m.bias)
    
    model.apply(init_weights)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
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
    
        # Check if this is the best validation error
        if eva < best_val_error:
            best_val_error = eva
            best_test_error = ete
            best_epoch = epoch
    
        print(f'Hyperparams [HS={hidden_size}, BS={batch_size}, LR={learning_rate:.6f}], '
              f'Epoch [{epoch+1}/{num_epochs}], Train: {etr:.4f}, Val: {eva:.4f}, Test: {ete:.4f}')
    
    print(f'Best Val Error: {best_val_error:.4f} (Epoch {best_epoch+1}), '
          f'Corresponding Test Error: {best_test_error:.4f}')
    
    return {
        'hidden_size': hidden_size,
        'batch_size': batch_size,
        'learning_rate': learning_rate,
        'best_val_error': best_val_error,
        'best_test_error': best_test_error,
        'best_epoch': best_epoch + 1
    }

def grid_search():
    """Perform grid search over hyperparameters"""
    # Define hyperparameter grid
    hidden_sizes = [100, 300, 500]
    batch_sizes = [50, 100]
    learning_rates = [0.01, 0.001, 0.0001]
    
    # Store results
    results = []
    
    # Total number of combinations
    total_combinations = len(hidden_sizes) * len(batch_sizes) * len(learning_rates)
    print(f"Starting grid search with {total_combinations} combinations...")
    
    # Perform grid search
    for hidden_size in hidden_sizes:
        for batch_size in batch_sizes:
            for learning_rate in learning_rates:
                print(f"\nTraining with Hidden Size={hidden_size}, Batch Size={batch_size}, Learning Rate={learning_rate}")
                
                # Train model with current hyperparameters
                result = train_model(hidden_size, batch_size, learning_rate)
                results.append(result)
    
    # Create a table of results
    print("\n" + "="*80)
    print("Grid Search Results:")
    print("="*80)
    print(f"{'Hidden Size':<15} {'Batch Size':<15} {'Learning Rate':<15} {'Best Val Error':<20} {'Best Test Error':<20} {'Best Epoch':<10}")
    print("-"*80)
    
    # Find the best combination
    best_val_error = float('inf')
    best_result = None
    
    for result in results:
        print(f"{result['hidden_size']:<15} {result['batch_size']:<15} {result['learning_rate']:<15.6f} "
              f"{result['best_val_error']:<20.4f} {result['best_test_error']:<20.4f} {result['best_epoch']:<10}")
        
        if result['best_val_error'] < best_val_error:
            best_val_error = result['best_val_error']
            best_result = result
    
    print("\nBest Hyperparameters:")
    print(f"Hidden Size: {best_result['hidden_size']}")
    print(f"Batch Size: {best_result['batch_size']}")
    print(f"Learning Rate: {best_result['learning_rate']}")
    print(f"Best Validation Error: {best_result['best_val_error']:.4f}")
    print(f"Corresponding Test Error: {best_result['best_test_error']:.4f}")
    print(f"Best Epoch: {best_result['best_epoch']}")
    
    # Create a simple visualization
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Prepare data for plotting
    x_labels = []
    test_errors = []
    val_errors = []
    
    for result in results:
        label = f"HS={result['hidden_size']},BS={result['batch_size']},LR={result['learning_rate']:.4f}"
        x_labels.append(label)
        test_errors.append(result['best_test_error'])
        val_errors.append(result['best_val_error'])
    
    # Find the best combination index
    best_idx = val_errors.index(min(val_errors))
    
    # Plot bars
    x = np.arange(len(x_labels))
    width = 0.35
    
    ax.bar(x - width/2, val_errors, width, label='Validation Error')
    ax.bar(x + width/2, test_errors, width, label='Test Error')
    
    # Highlight the best combination
    ax.bar(best_idx - width/2, val_errors[best_idx], width, color='green', label='Best Validation Error')
    ax.bar(best_idx + width/2, test_errors[best_idx], width, color='lightgreen', label='Corresponding Test Error')
    
    # Add labels and legend
    ax.set_ylabel('Error')
    ax.set_title('Validation and Test Errors for Different Hyperparameter Combinations')
    ax.set_xticks([])  # Hide x-labels as they would be too crowded
    ax.legend()
    
    plt.tight_layout()
    plt.savefig('grid_search_results.png')
    plt.show()
    
    return results

if __name__ == "__main__":
    results = grid_search()
    print("\nGrid search completed. Results visualization saved as 'grid_search_results.png'")