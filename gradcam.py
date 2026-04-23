import os
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
import cv2
import matplotlib.pyplot as plt

IMG_SIZE = 224

model = models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, 5)
model.load_state_dict(torch.load("dr_model.pth", map_location="cpu"))
model.eval()

target_layer = model.layer4[-1]

transform = transforms.Compose(
    [
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)

img_path = "train_images/000c1434d8d7.png"

img = Image.open(img_path).convert("RGB")
input_tensor = transform(img).unsqueeze(0)

features = []
gradients = []


def forward_hook(module, input, output):
    features.append(output)


def backward_hook(module, grad_in, grad_out):
    gradients.append(grad_out[0])


target_layer.register_forward_hook(forward_hook)
target_layer.register_backward_hook(backward_hook)

output = model(input_tensor)
pred_class = output.argmax()

model.zero_grad()
output[0, pred_class].backward()

grad = gradients[0]
feat = features[0]

weights = torch.mean(grad, dim=(2, 3))[0]

cam = torch.zeros(feat.shape[2:], dtype=torch.float32)

for i, w in enumerate(weights):
    cam += w * feat[0, i, :, :]

cam = cam.detach().numpy()
cam = np.maximum(cam, 0)
cam = cam / cam.max()
cam = cv2.resize(cam, (224, 224))

img_np = np.array(img.resize((224, 224)))

heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
result = heatmap * 0.4 + img_np

plt.imshow(result.astype(np.uint8))
plt.axis("off")
plt.show()
