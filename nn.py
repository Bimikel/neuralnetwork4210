import torch
import torch.nn as nn
import torch.optim as optim

# Define the neural network
class SimpleNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)  # Fully connected layer 1
        self.relu = nn.ReLU()                         # Activation function
        self.fc2 = nn.Linear(hidden_size, output_size)  # Fully connected layer 2

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

from torch.utils.data import DataLoader, TensorDataset

# Example dataset
X = torch.randn(100, 3)  # 100 samples, 3 features
y = torch.randint(0, 2, (100,))  # Binary labels

# Convert to dataset and loader
dataset = TensorDataset(X, y)
dataloader = DataLoader(dataset, batch_size=10, shuffle=True)

input_size = 3
hidden_size = 5
output_size = 2

model = SimpleNN(input_size, hidden_size, output_size)
criterion = nn.CrossEntropyLoss()  # Suitable for classification tasks
optimizer = optim.SGD(model.parameters(), lr=0.01)  # Stochastic Gradient Descent

num_epochs = 20

for epoch in range(num_epochs):
    for batch_X, batch_y in dataloader:
        # Forward pass
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# Example evaluation
with torch.no_grad():
    test_X = torch.randn(10, 3)  # Test input
    test_outputs = model(test_X)
    predictions = torch.argmax(test_outputs, dim=1)
    print("Predictions:", predictions)

