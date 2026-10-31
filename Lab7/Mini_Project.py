# Lab 7: Mini Project – Image Classification using ResNet18

import torch
import torchvision.models as models
from PIL import Image
import matplotlib.pyplot as plt

# 1. Load Pre-trained ResNet18 Model
weights = models.ResNet18_Weights.DEFAULT
model = models.resnet18(weights=weights)
model.eval()

# 2. Load Image
image = Image.open("sample.jpg").convert("RGB")

# 3. Preprocess Image
transform = weights.transforms()
input_image = transform(image).unsqueeze(0)

# 4. Perform Image Classification
with torch.no_grad():
    output = model(input_image)

# 5. Get Prediction
probabilities = torch.nn.functional.softmax(output[0], dim=0)
confidence, class_index = torch.max(probabilities, dim=0)

predicted_class = weights.meta["categories"][class_index.item()]

# 6. Display Results
print("Predicted Class:", predicted_class)
print("Confidence:", round(confidence.item() * 100, 2), "%")

plt.imshow(image)
plt.title("Predicted: " + predicted_class)
plt.axis("off")
plt.savefig("mini_project_output.png")
plt.show()

print("Image Classification Project Completed!")
