# Lab 6: Neural Network using PyTorch

import torch
import torch.nn as nn
import torch.optim as optim

# Training Data
X = torch.tensor([[1.0], [2.0], [3.0], [4.0]])
Y = torch.tensor([[2.0], [4.0], [6.0], [8.0]])

# Define Neural Network
class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.fc1 = nn.Linear(1, 5)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(5, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

# Create Model
model = NeuralNetwork()

# Loss Function and Optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Training
for epoch in range(1000):
    output = model(X)
    loss = criterion(output, Y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# Display Results
print("Actual Values:")
print(Y)

print("Predicted Values:")
print(model(X).detach())

print("Final Loss:", loss.item())
print("Neural Network Training Completed!")
