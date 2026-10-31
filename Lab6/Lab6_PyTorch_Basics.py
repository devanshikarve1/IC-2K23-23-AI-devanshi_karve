# Lab 6: Deep Learning Fundamentals using PyTorch

# 1. Import PyTorch
import torch

print("PyTorch Version:", torch.__version__)

# 2. Tensor Operations
a = torch.tensor([1, 2, 3])
b = torch.tensor([4, 5, 6])

print("Tensor A:", a)
print("Tensor B:", b)
print("Addition:", a + b)
print("Multiplication:", a * b)

# 3. Automatic Differentiation
x = torch.tensor(2.0, requires_grad=True)
y = x ** 2 + 3 * x + 1

y.backward()

print("Value of y:", y.item())
print("Gradient:", x.grad.item())

# 4. Linear Regression using PyTorch
X = torch.tensor([[1.0], [2.0], [3.0], [4.0]])
Y = torch.tensor([[2.0], [4.0], [6.0], [8.0]])

model = torch.nn.Linear(1, 1)

loss_function = torch.nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

for epoch in range(500):
    prediction = model(X)
    loss = loss_function(prediction, Y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

print("Linear Regression Predictions:")
print(model(X).detach())

# 5. Neural Network Basics
class NeuralNetwork(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = torch.nn.Linear(1, 5)
        self.relu = torch.nn.ReLU()
        self.layer2 = torch.nn.Linear(5, 1)

    def forward(self, x):
        x = self.layer1(x)
        x = self.relu(x)
        x = self.layer2(x)
        return x

nn_model = NeuralNetwork()

print("Neural Network Model:")
print(nn_model)

# 6. Model Evaluation
with torch.no_grad():
    output = nn_model(X)
    print("Model Output:")
    print(output)

print("Lab 6 Completed Successfully!")
