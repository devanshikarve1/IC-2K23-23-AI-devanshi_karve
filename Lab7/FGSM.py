# Lab 7: Adversarial AI using FGSM

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

# Load MNIST Dataset
transform = transforms.ToTensor()

train_data = torchvision.datasets.MNIST(
    root="./data",
    train=True,
    download=True,
    transform=transform
)

test_data = torchvision.datasets.MNIST(
    root="./data",
    train=False,
    download=True,
    transform=transform
)

train_loader = torch.utils.data.DataLoader(
    train_data, batch_size=64, shuffle=True
)

test_loader = torch.utils.data.DataLoader(
    test_data, batch_size=64, shuffle=False
)

# Define Neural Network
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv = nn.Conv2d(1, 16, 3)
        self.pool = nn.MaxPool2d(2)
        self.fc = nn.Linear(16 * 13 * 13, 10)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv(x)))
        x = x.view(x.size(0), -1)
        return self.fc(x)

model = CNN()

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train Model
for epoch in range(2):
    for images, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

    print("Epoch:", epoch + 1, "Loss:", loss.item())

# FGSM Attack
def fgsm_attack(image, epsilon, data_gradient):
    perturbed_image = image + epsilon * data_gradient.sign()
    return torch.clamp(perturbed_image, 0, 1)

# Test Original and Adversarial Images
model.eval()

images, labels = next(iter(test_loader))
images.requires_grad = True

outputs = model(images)
loss = criterion(outputs, labels)

model.zero_grad()
loss.backward()

data_gradient = images.grad.data

epsilon = 0.2
perturbed_images = fgsm_attack(images, epsilon, data_gradient)

original_prediction = model(images.detach()).argmax(dim=1)
adversarial_prediction = model(perturbed_images.detach()).argmax(dim=1)

print("Original Accuracy:",
      (original_prediction == labels).float().mean().item())

print("Adversarial Accuracy:",
      (adversarial_prediction == labels).float().mean().item())

# Display Images
plt.figure(figsize=(8, 4))

plt.subplot(1, 2, 1)
plt.imshow(images[0].detach().squeeze(), cmap="gray")
plt.title("Original Image")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(perturbed_images[0].detach().squeeze(), cmap="gray")
plt.title("FGSM Adversarial Image")
plt.axis("off")

plt.savefig("fgsm_output.png")
plt.show()

print("FGSM Attack Completed Successfully!")
