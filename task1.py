# based on https://github.com/milindmalshe/Fully-Connected-Neural-Network-PyTorch/blob/master/FCN_MNIST_Classification_PyTorch.py
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import os

# Create results directory if it doesn't exist
os.makedirs('results', exist_ok=True)

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

# Data loader
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
										   batch_size=batch_size,
										   shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
										  batch_size=batch_size,
										  shuffle=False)

# Uncomment to visualize some training images
# images, labels = next(iter(train_loader))
# fig, axs = plt.subplots(2, 5)
# for ii in range(2):
# 	for jj in range(5):
# 		idx = 5 * ii + jj
# 		axs[ii, jj].imshow(images[idx].squeeze())
# 		axs[ii, jj].set_title(labels[idx].item())
# 		axs[ii, jj].axis('off')
# plt.show()


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

# Function to compute error rate
def compute_error(loader, model):
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in loader:
            images = images.reshape(-1, input_size).to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        error_rate = 1 - (correct / total)
        return error_rate

# Lists to store errors
train_errors = []
test_errors = []

# train the model
total_step = len(train_loader)
for epoch in range(num_epochs):
	# Training
	model.train()
	for i, (images, labels) in enumerate(train_loader):
		images = images.reshape(-1, input_size).to(device)
		labels = labels.to(device)

		outputs = model(images)
		loss = criterion(outputs, labels)

		optimizer.zero_grad()
		loss.backward()
		optimizer.step()

		if (i+1) % 100 == 0:
			print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
				.format(epoch+1, num_epochs, i+1, total_step, loss.item()))
	
	# Compute errors after each epoch
	model.eval()
	train_error = compute_error(train_loader, model)
	test_error = compute_error(test_loader, model)
	
	train_errors.append(train_error)
	test_errors.append(test_error)
	
	print(f'Epoch [{epoch+1}/{num_epochs}], Train Error: {train_error:.4f}, Test Error: {test_error:.4f}')

# Final test error
final_test_error = test_errors[-1]
print(f'Final test error: {final_test_error:.4f}')

# Plot training and test errors
plt.figure(figsize=(10, 6))
plt.plot(range(1, num_epochs + 1), train_errors, 'b-', label='Training Error')
plt.plot(range(1, num_epochs + 1), test_errors, 'r-', label='Test Error')
plt.title('Training and Test Error over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Error Rate')
plt.legend()
plt.grid(True)
plt.savefig('results/task1_error_plot.png')
plt.show()

# Find and plot misclassified images
def get_misclassified_samples(model, loader, num_samples=10):
    model.eval()
    misclassified_images = []
    misclassified_labels = []
    misclassified_preds = []
    
    with torch.no_grad():
        for images, labels in loader:
            images_flat = images.reshape(-1, input_size).to(device)
            labels = labels.to(device)
            
            outputs = model(images_flat)
            _, preds = torch.max(outputs, 1)
            
            # Find misclassified samples
            mask = preds != labels
            misclassified_idx = mask.nonzero(as_tuple=True)[0]
            
            for idx in misclassified_idx:
                if len(misclassified_images) < num_samples:
                    misclassified_images.append(images[idx].cpu().numpy())
                    misclassified_labels.append(labels[idx].item())
                    misclassified_preds.append(preds[idx].item())
                else:
                    break
            
            if len(misclassified_images) >= num_samples:
                break
    
    return misclassified_images, misclassified_labels, misclassified_preds

# Get misclassified samples
misclassified_images, misclassified_labels, misclassified_preds = get_misclassified_samples(model, test_loader)

# Plot misclassified images
fig, axes = plt.subplots(2, 5, figsize=(12, 6))
axes = axes.flatten()

for i, (img, true_label, pred_label) in enumerate(zip(misclassified_images, misclassified_labels, misclassified_preds)):
    axes[i].imshow(img.squeeze(), cmap='gray')
    axes[i].set_title(f'True: {true_label}, Pred: {pred_label}')
    axes[i].axis('off')

plt.tight_layout()
plt.savefig('results/task1_misclassified.png')
plt.show()

# Save the model checkpoint
torch.save(model.state_dict(), 'results/task1_model.ckpt')
