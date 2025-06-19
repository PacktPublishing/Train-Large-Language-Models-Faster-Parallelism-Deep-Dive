# install: pip install torch transformers numpy torchvision datasets
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from transformers import BertModel, BertConfig


# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Step 1: Create a simple transformer model
config = BertConfig(
    vocab_size=30522,
    hidden_size=128,
    num_hidden_layers=2,
    num_attention_heads=2,
    intermediate_size=512,
)

model = BertModel(config)

# Modified TransformerClassifier with a projection layer
class TransformerClassifier(nn.Module):
    def __init__(self, bert_model):
        super(TransformerClassifier, self).__init__()
        self.bert = bert_model
        # Project the input to match BERT's hidden_size
        self.projection = nn.Linear(1, 128)  # 1 → 128 (hidden_size)
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(128, 10)  # 128 → 10 (MNIST classes)

    def forward(self, x):
        # x shape: (batch_size, 28*28) for MNIST flattened
        x = x.view(-1, 28, 28)  # Reshape to (batch_size, 28, 28)
        x = x.mean(dim=2)  # (batch_size, 28)
        x = x.unsqueeze(-1)  # (batch_size, 28, 1)
        x = self.projection(x)  # (batch_size, 28, 128)
        outputs = self.bert(inputs_embeds=x)  # (batch_size, 28, 128)
        pooled_output = outputs.last_hidden_state[:, 0, :]  # (batch_size, 128)
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)  # (batch_size, 10)
        return logits
    
model = TransformerClassifier(model).to(device)


# Step 2: Enable Data Parallelism
if torch.cuda.device_count() > 1:
    print(f"Using {torch.cuda.device_count()} GPUs!")
    model = nn.DataParallel(model)

# Step 3: Load MNIST dataset
transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
)


train_dataset = datasets.MNIST(
    root="./data", train=True, download=True, transform=transform
)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)


# Step 4: Set up optimizer and loss
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()


# Step 5: Training loop
try:
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        data = data.view(data.size(0), -1)  # (batch_size, 28*28)

        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        if batch_idx % 100 == 0:
            print(f"Batch {batch_idx}, Loss: {loss.item():.4f}")
except Exception as e:
    print(f"An error occurred during training: {e}")
else:
    print("Training complete for 1 epoch!")
    
    
# Step 6: Evaluate on test set
test_dataset = datasets.MNIST(
    root="./data", train=False, download=True, transform=transform
)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)


# Overall, this code evaluates the model's performance on the test dataset by 
# calculating the total number of correct predictions and the total number of samples, 
# which can later be used to compute the accuracy of the model.
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        data = data.view(data.size(0), -1)
        output = model(data)
        _, predicted = torch.max(output.data, 1)
        total += target.size(0)
        correct += (predicted == target).sum().item()

accuracy = 100.0 * correct / total
print(f"Test Accuracy: {accuracy:.2f}%")

# Step 7: Save the model
torch.save(model.state_dict(), "model_data_parallel.pth")
print("Model saved!")


