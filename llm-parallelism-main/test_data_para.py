import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from transformers import BertModel, BertConfig
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Step 1: Define the model architecture (must match the trained model)
config = BertConfig(
    vocab_size=30522,
    hidden_size=128,
    num_hidden_layers=2,
    num_attention_heads=2,
    intermediate_size=512,
)
base_model = BertModel(config)


class TransformerClassifier(nn.Module):
    def __init__(self, bert_model):
        super(TransformerClassifier, self).__init__()
        self.bert = bert_model
        self.projection = nn.Linear(1, 128)  # Matches the trained model
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(128, 10)  # 128 → 10 (MNIST classes)

    def forward(self, x):
        x = x.view(-1, 28, 28)  # (batch_size, 28, 28)
        x = x.mean(dim=2)  # (batch_size, 28)
        x = x.unsqueeze(-1)  # (batch_size, 28, 1)
        x = self.projection(x)  # (batch_size, 28, 128)
        outputs = self.bert(inputs_embeds=x)  # (batch_size, 28, 128)
        pooled_output = outputs.last_hidden_state[:, 0, :]  # (batch_size, 128)
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)  # (batch_size, 10)
        return logits


model = TransformerClassifier(base_model).to(device)

# Step 2: Load the saved weights
# Check if the model was saved with DataParallel (module prefix)
state_dict = torch.load("model_data_parallel.pth")
if "module." in list(state_dict.keys())[0]:
    # Remove 'module.' prefix if present
    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
model.load_state_dict(state_dict)
print("Model weights loaded successfully!")

# Step 3: Load the test dataset
transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
)
test_dataset = datasets.MNIST(
    root="./data", train=False, download=True, transform=transform
)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# Step 4: Evaluate the model
model.eval()  # Set to evaluation mode
correct = 0
total = 0
with torch.no_grad():  # Disable gradient computation for efficiency
    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        data = data.view(data.size(0), -1)  # (batch_size, 28*28)
        output = model(data)
        _, predicted = torch.max(output.data, 1)  # Get the predicted class
        total += target.size(0)
        correct += (predicted == target).sum().item()

accuracy = 100.0 * correct / total
print(f"Test Accuracy: {accuracy:.2f}%")


# Step 5: Manually test the model with a few numbers
def test_single_image(image, model):
    model.eval()
    with torch.no_grad():
        image = (
            transform(image).unsqueeze(0).to(device)
        )  # Add batch dimension and move to device
        image = image.view(image.size(0), -1)  # Flatten the image
        output = model(image)
        _, predicted = torch.max(output.data, 1)
        return predicted.item()


# Load a few sample images from the test dataset
sample_indices = [0, 1, 2, 3, 4, 78, 6, 9, 132, 7]  # Indices of the images to test
for idx in sample_indices:
    image, label = test_dataset[idx]
    # Convert the tensor image to a PIL Image
    image_pil = transforms.ToPILImage()(image)
    predicted_label = test_single_image(image_pil, model)
    plt.imshow(image.squeeze(), cmap="gray")
    plt.title(f"True Label: {label}, Predicted: {predicted_label}")
    plt.show()
