import cv2
import torch
import torch.nn as nn
from torchvision import models, transforms
import numpy as np
from PIL import Image

# Load the model
model = models.resnet18(weights=None)  # Correct way to initialize without pre-trained weights
model.fc = nn.Linear(model.fc.in_features, 2)  # Adjust if you have more/less classes

# Load model weights
model.load_state_dict(torch.load(
    "C:\\Users\\PC\\OneDrive - Lebanese American University\\LAU\\FYP\\Sanitization and food delivery\\AI model\\model_weights_updated.pth",
    map_location='cpu'))

model.eval()

# Define preprocessing transformation
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # ImageNet normalization
])

# Define class labels
labels = ["Class 0", "Class 1"]

# Initialize webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert frame from BGR (OpenCV default) to RGB
    frame_rgb = cv2.cvtColor(framSe, cv2.COLOR_BGR2RGB)

    # Convert numpy array (frame) to PIL Image
    pil_image = Image.fromarray(frame_rgb)

    # Preprocess the PIL image
    input_tensor = preprocess(pil_image)
    input_batch = input_tensor.unsqueeze(0)

    # Inference
    with torch.no_grad():
        output = model(input_batch)
        pred = torch.argmax(output, dim=1).item()
        label = labels[pred]

        
    # Display the prediction on the frame
    cv2.putText(frame, label, (10, 40), cv2.FONT_HERSHEY_SIMPLEX,
                1, (0, 255, 0), 2, cv2.LINE_AA)

    # Show the frame with the prediction
    cv2.imshow('Real-time Inference', frame)

    # Exit the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
