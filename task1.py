# based on https://github.com/milindmalshe/Fully-Connected-Neural-Network-PyTorch/blob/master/FCN_MNIST_Classification_PyTorch.py
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

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

# Data loaders
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True)

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

model = NeuralNet(input_size, hidden_size, num_classes).to(device)
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

	# Activating forward on the images
        outputs = model(images)
        loss = criterion(outputs, labels)

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

    print(f'Epoch [{epoch+1}/{num_epochs}], Train Error: {etr:.4f}, Test Error: {ete:.4f}')

# Report the final test error from the last epoch
final_test_error = test_errors[-1]  # Get the last test error from the list
print(f'Final Test Error (Cross-Entropy Loss): {final_test_error:.4f}')

# Final test misclassified examples
misclassified = []

with torch.no_grad():
    for images, labels in test_loader:
        flat_images = images.reshape(-1, input_size).to(device)
        labels = labels.to(device)
        outputs = model(flat_images)
        _, predicted = torch.max(outputs.data, 1)

        # Save some misclassified examples
        for i in range(images.size(0)):
            if predicted[i] != labels[i] and len(misclassified) < 10:
                misclassified.append((images[i].cpu(), predicted[i].cpu(), labels[i].cpu()))

# Plot error graphs
plt.figure()
plt.plot(train_errors, label='Train Error')
plt.plot(test_errors, label='Test Error')
plt.xlabel('Epoch')
plt.ylabel('Error')
plt.title('Train vs Test Error')
plt.legend()
plt.grid(True)
plt.show()

# Show misclassified images
fig, axs = plt.subplots(2, 5, figsize=(10, 5))
fig.suptitle('Misclassified Images')
for idx, (img, pred, true) in enumerate(misclassified):
    row, col = divmod(idx, 5)
    axs[row, col].imshow(img.squeeze(), cmap='gray')
    axs[row, col].set_title(f'True: {true}, Pred: {pred}')
    axs[row, col].axis('off')
plt.tight_layout()
plt.show()
