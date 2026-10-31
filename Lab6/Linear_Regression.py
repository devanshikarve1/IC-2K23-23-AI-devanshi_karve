# Lab 6: Linear Regression using PyTorch

import torch
import torch.nn as nn
import torch.optim as optim

# Training Data
X = torch.tensor([[1.0], [2.0], [3.0], [4.0], [5.0]])
Y = torch.tensor([[2.0], [4.0], [6.0], [8.0], [10.0]])

# Create Linear Regression Model
model = nn.Linear(1, 1)

# Loss Function and Optimizer
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# Model Training
for epoch in range(1000):
    predictions = model(X)
    loss = criterion(predictions, Y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# Display Results
print("Actual Values:")
print(Y)

print("Predicted Values:")
print(model(X).detach())

print("Final Loss:", loss.item())

print("Linear Regression Completed Successfully!")
