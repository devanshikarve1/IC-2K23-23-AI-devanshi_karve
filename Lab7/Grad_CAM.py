# Lab 7: Explainable AI using Grad-CAM

import torch
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt

# Load Pre-trained ResNet18 Model
weights = models.ResNet18_Weights.DEFAULT
model = models.resnet18(weights=weights)
model.eval()

# Load Image
image = Image.open("sample.jpg").convert("RGB")

transform = weights.transforms()
input_image = transform(image).unsqueeze(0)

# Store Feature Maps and Gradients
features = None
gradients = None

def forward_hook(module, input, output):
    global features
    features = output

def backward_hook(module, grad_input, grad_output):
    global gradients
    gradients = grad_output[0]

# Register Hooks
target_layer = model.layer4
target_layer.register_forward_hook(forward_hook)
target_layer.register_full_backward_hook(backward_hook)

# Forward Pass
output = model(input_image)
class_index = output.argmax(dim=1).item()

# Backward Pass
model.zero_grad()
output[0, class_index].backward()

# Calculate Grad-CAM
weights_cam = gradients.mean(dim=(2, 3), keepdim=True)
cam = (weights_cam * features).sum(dim=1, keepdim=True)
cam = F.relu(cam)

# Normalize Heatmap
cam = F.interpolate(
    cam,
    size=(224, 224),
    mode="bilinear",
    align_corners=False
)

cam = cam.squeeze().detach()
cam = cam / (cam.max() + 1e-8)

# Display Original Image and Heatmap
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.imshow(image)
plt.title("Original Image")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(image.resize((224, 224)))
plt.imshow(cam.numpy(), cmap="jet", alpha=0.5)
plt.title("Grad-CAM Visualization")
plt.axis("off")

plt.savefig("gradcam_output.png")
plt.show()

print("Predicted Class:", weights.meta["categories"][class_index])
print("Grad-CAM Completed Successfully!")
