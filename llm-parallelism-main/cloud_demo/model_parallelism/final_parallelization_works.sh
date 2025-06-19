#!/bin/bash

echo "Setting up IMDB sentiment analysis demo on RunPod.io with 2 GPUs using Hugging Face datasets..."

# Install required packages (no Pillow needed for text data)
pip install torch transformers matplotlib pandas seaborn tqdm datasets

# Create the main training script
cat > bert_imdb_parallel.py << 'EOL'
#!/usr/bin/env python3
"""
BERT-like Model Training on IMDB for Sentiment Analysis with Model Parallelism on RunPod.io (2 GPUs)
Using Hugging Face datasets and transformers
"""

import os
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from transformers import BertTokenizer
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from tqdm import tqdm
import random

# Set random seeds for reproducibility
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

# Utility Functions
def print_gpu_info():
    """Print detailed GPU information for students"""
    print("===== GPU Configuration Check =====")
    if torch.cuda.is_available():
        print(f"PyTorch version: {torch.__version__}")
        print(f"CUDA version: {torch.version.cuda}")
        print(f"Number of GPUs detected: {torch.cuda.device_count()}")
        if torch.cuda.device_count() < 2:
            print("WARNING: This demo requires 2 GPUs but fewer detected!")
        for i in range(min(2, torch.cuda.device_count())):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
            mem_free, mem_total = torch.cuda.mem_get_info(i)
            print(f"GPU {i} Memory: {mem_free/1e9:.2f} GB free / {mem_total/1e9:.2f} GB total")
    else:
        print("ERROR: No GPUs available! This demo requires 2 GPUs.")
        exit(1)
    print("==================================\n")

def print_model_info(model, name="Model"):
    """Print model statistics with GPU distribution"""
    num_params = sum(p.numel() for p in model.parameters())
    num_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"{name} statistics:")
    print(f"  Total Parameters: {num_params:,}")
    print(f"  Trainable Parameters: {num_trainable:,}")
    print(f"  Model split: First half on GPU 0, Second half on GPU 1")

# Custom Dataset Class for IMDB
class IMDBForTransformer(Dataset):
    """Preprocess IMDB dataset for transformer model"""
    def __init__(self, hf_dataset, tokenizer, max_len=128):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.data = hf_dataset
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        try:
            item = self.data[idx]
            text = item['text']
            label = item['label']  # 0 for negative, 1 for positive
            
            # Tokenize text
            encoding = self.tokenizer(
                text,
                add_special_tokens=True,
                max_length=self.max_len,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            
            return {
                'input_ids': encoding['input_ids'].squeeze(0),  # Remove batch dim
                'attention_mask': encoding['attention_mask'].squeeze(0),
                'label': torch.tensor(label, dtype=torch.long)
            }
        except Exception as e:
            print(f"Error processing item {idx}: {e}")
            raise

# Transformer Block
class TransformerBlock(nn.Module):
    """Single transformer encoder block"""
    def __init__(self, hidden_size=128, num_heads=4, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.head_size = hidden_size // num_heads
        self.hidden_size = hidden_size
        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)
        self.output = nn.Linear(hidden_size, hidden_size)
        self.layer_norm1 = nn.LayerNorm(hidden_size)
        self.layer_norm2 = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.ff = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 4, hidden_size),
            nn.Dropout(dropout)
        )
        
    def split_heads(self, x):
        batch_size, seq_length, _ = x.size()
        return x.view(batch_size, seq_length, self.num_heads, self.head_size).transpose(1, 2)
    
    def merge_heads(self, x):
        batch_size, _, seq_length, _ = x.size()
        return x.transpose(1, 2).contiguous().view(batch_size, seq_length, self.hidden_size)
    
    def attention(self, query, key, value, mask=None):
        scores = torch.matmul(query, key.transpose(-1, -2)) / (self.head_size ** 0.5)
        if mask is not None:
            mask = mask.unsqueeze(1).unsqueeze(2)
            scores = scores.masked_fill(mask == 0, -1e9)
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        context = torch.matmul(attention_weights, value)
        return context
    
    def forward(self, x, attention_mask=None):
        residual = x
        x = self.layer_norm1(x)
        q = self.split_heads(self.query(x))
        k = self.split_heads(self.key(x))
        v = self.split_heads(self.value(x))
        context = self.attention(q, k, v, attention_mask)
        context = self.merge_heads(context)
        output = self.output(context)
        output = self.dropout(output)
        x = residual + output
        residual = x
        x = self.layer_norm2(x)
        x = self.ff(x)
        x = residual + x
        return x

# BERT-like Model for IMDB Sentiment Analysis
class BERTForIMDB(nn.Module):
    """BERT-like model with 2-GPU parallelism for sentiment analysis"""
    def __init__(self, vocab_size, hidden_size=128, num_layers=4, num_heads=4, num_classes=2, dropout=0.1, max_len=128):
        super().__init__()
        if torch.cuda.device_count() < 2:
            print("ERROR: This demo requires exactly 2 GPUs!")
            exit(1)
        
        self.device_0 = torch.device('cuda:0')
        self.device_1 = torch.device('cuda:1')
        print(f"Initializing model: GPU 0 = {self.device_0}, GPU 1 = {self.device_1}")
        
        # GPU 0 components
        self.embedding = nn.Embedding(vocab_size, hidden_size).to(self.device_0)
        self.position_embeddings = nn.Parameter(torch.randn(1, max_len, hidden_size)).to(self.device_0)
        self.layer_norm = nn.LayerNorm(hidden_size).to(self.device_0)
        self.dropout = nn.Dropout(dropout).to(self.device_0)
        
        self.split_point = num_layers // 2  # Make split_point an instance variable
        self.layers_first_half = nn.ModuleList([
            TransformerBlock(hidden_size, num_heads, dropout).to(self.device_0)
            for _ in range(self.split_point)
        ])
        print(f"Placed {self.split_point} transformer layers on GPU 0")
        
        # GPU 1 components
        self.layers_second_half = nn.ModuleList([
            TransformerBlock(hidden_size, num_heads, dropout).to(self.device_1)
            for _ in range(num_layers - self.split_point)
        ])
        print(f"Placed {num_layers - self.split_point} transformer layers on GPU 1")
        
        self.pooler = nn.Linear(hidden_size, hidden_size).to(self.device_1)
        self.classifier = nn.Linear(hidden_size, num_classes).to(self.device_1)
        print("Placed pooler and classifier on GPU 1")
    
    def forward(self, input_ids, attention_mask=None):
        print(f"Starting forward pass: Moving input to GPU 0 ({self.device_0})")
        input_ids = input_ids.to(self.device_0)
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device_0)
        
        x = self.embedding(input_ids)
        batch_size, seq_length, _ = x.size()
        x = x + self.position_embeddings[:, :seq_length, :]
        x = self.layer_norm(x)
        x = self.dropout(x)
        print(f"Processed embedding on GPU 0. Shape: {x.shape}")
        
        for i, layer in enumerate(self.layers_first_half):
            x = layer(x, attention_mask)
            print(f"Completed transformer layer {i+1} on GPU 0")
        
        print(f"Transferring intermediate features from GPU 0 to GPU 1. Memory: {x.element_size() * x.nelement() / 1e6:.2f} MB")
        x = x.to(self.device_1)
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device_1)
        
        for i, layer in enumerate(self.layers_second_half):
            x = layer(x, attention_mask)
            print(f"Completed transformer layer {self.split_point + i + 1} on GPU 1")
        
        pooled_output = torch.tanh(self.pooler(x[:, 0]))  # Use [CLS] token equivalent
        logits = self.classifier(pooled_output)
        print(f"Completed classification on GPU 1. Output shape: {logits.shape}")
        
        return logits

# Training and Evaluation Functions
def train_model(model, train_dataloader, val_dataloader, args):
    """Train the model with detailed logging"""
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.CrossEntropyLoss()
    train_losses = []
    val_accuracies = []
    
    print("Starting initial validation...")
    val_acc = evaluate_model(model, val_dataloader)
    val_accuracies.append(val_acc)
    print(f"Initial validation accuracy: {val_acc:.4f}")
    
    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0.0
        epoch_start = time.time()
        print(f"\n=== Starting Epoch {epoch+1}/{args.epochs} ===")
        
        for batch_idx, batch in enumerate(tqdm(train_dataloader, desc=f"Epoch {epoch+1}")):
            input_ids = batch['input_ids']
            attention_mask = batch['attention_mask']
            labels = batch['label'].to(model.device_1)
            
            print(f"Batch {batch_idx+1}: Moving data to GPUs")
            outputs = model(input_ids, attention_mask)
            loss = criterion(outputs, labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            
            mem0 = torch.cuda.memory_allocated(0) / 1e9
            mem1 = torch.cuda.memory_allocated(1) / 1e9
            print(f"Batch {batch_idx+1}: Loss = {loss.item():.4f}, GPU 0 Memory = {mem0:.2f} GB, GPU 1 Memory = {mem1:.2f} GB")
        
        scheduler.step()
        avg_loss = epoch_loss / len(train_dataloader)
        train_losses.append(avg_loss)
        
        val_acc = evaluate_model(model, val_dataloader)
        val_accuracies.append(val_acc)
        epoch_time = time.time() - epoch_start
        
        print(f"Epoch {epoch+1} completed: Avg Loss = {avg_loss:.4f}, Val Acc = {val_acc:.4f}, Time = {epoch_time:.2f}s")
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': avg_loss,
            'val_accuracy': val_acc
        }, f"model_epoch_{epoch+1}.pt")
    
    pd.DataFrame({
        'epoch': list(range(args.epochs + 1)),
        'train_loss': [0.0] + train_losses,
        'val_accuracy': val_accuracies
    }).to_csv('training_history.csv', index=False)
    return {'train_losses': train_losses, 'val_accuracies': val_accuracies}

def evaluate_model(model, dataloader):
    """Evaluate the model"""
    model.eval()
    correct = 0
    total = 0
    print("Starting evaluation...")
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids']
            attention_mask = batch['attention_mask']
            labels = batch['label'].to(model.device_1)
            outputs = model(input_ids, attention_mask)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = correct / total
    print(f"Evaluation complete: Accuracy = {accuracy:.4f}")
    return accuracy

def test_predictions(model, test_dataloader, num_samples=10):
    """Generate predictions for visualization"""
    model.eval()
    all_preds = []
    all_labels = []
    samples = []
    print("Starting test predictions...")
    with torch.no_grad():
        for i, batch in enumerate(test_dataloader):
            input_ids = batch['input_ids']
            attention_mask = batch['attention_mask']
            labels = batch['label'].to(model.device_1)
            outputs = model(input_ids, attention_mask)
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())
            if i == 0:
                for j in range(min(num_samples, len(input_ids))):
                    samples.append({
                        'label': labels[j].item(),
                        'prediction': predicted[j].item()
                    })
    accuracy = sum(p == l for p, l in zip(all_preds, all_labels)) / len(all_labels)
    cm = np.zeros((2, 2), dtype=int)
    for p, l in zip(all_preds, all_labels):
        cm[l][p] += 1
    print(f"Test predictions complete: Accuracy = {accuracy:.4f}")
    return {'accuracy': accuracy, 'confusion_matrix': cm, 'samples': samples}

# Visualization Functions
def plot_training_history(history):
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history['train_losses'])
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.subplot(1, 2, 2)
    plt.plot(history['val_accuracies'])
    plt.title('Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.close()
    print("Training history plot saved to 'training_history.png'")

def plot_confusion_matrix(cm):
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Negative', 'Positive'], yticklabels=['Negative', 'Positive'])
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.savefig('confusion_matrix.png')
    plt.close()
    print("Confusion matrix saved to 'confusion_matrix.png'")

def visualize_samples(samples):
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    axes = axes.flatten()
    for i, sample in enumerate(samples[:10]):
        label = sample['label']
        prediction = sample['prediction']
        color = 'green' if label == prediction else 'red'
        axes[i].text(0.5, 0.5, f"True: {'Pos' if label == 1 else 'Neg'}\nPred: {'Pos' if prediction == 1 else 'Neg'}", 
                     ha='center', va='center', fontsize=12, color=color)
        axes[i].axis('off')
    plt.tight_layout()
    plt.savefig('sample_predictions.png')
    plt.close()
    print("Sample predictions saved to 'sample_predictions.png'")

# Main Function
def main():
    parser = argparse.ArgumentParser(description="BERT-like Model for IMDB Sentiment Analysis on RunPod.io")
    parser.add_argument("--hidden_size", type=int, default=128)
    parser.add_argument("--num_layers", type=int, default=4)
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--max_len", type=int, default=128)
    args = parser.parse_args()
    
    print("===== Starting IMDB Sentiment Analysis Demo on RunPod.io =====")
    print_gpu_info()
    
    print("Loading IMDB dataset from Hugging Face...")
    imdb_dataset = load_dataset("imdb")
    train_dataset = imdb_dataset['train']
    test_dataset = imdb_dataset['test']
    
    # Split train into train and validation (90% train, 10% val)
    train_size = int(0.9 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_indices = list(range(len(train_dataset)))
    random.shuffle(train_indices)
    train_subset = train_dataset.select(train_indices[:train_size])
    val_subset = train_dataset.select(train_indices[train_size:])
    
    # Initialize tokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    
    # Wrap in custom dataset class
    train_dataset = IMDBForTransformer(train_subset, tokenizer, max_len=args.max_len)
    val_dataset = IMDBForTransformer(val_subset, tokenizer, max_len=args.max_len)
    test_dataset = IMDBForTransformer(test_dataset, tokenizer, max_len=args.max_len)
    
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)
    print(f"Dataset loaded: Train={len(train_dataset)}, Val={len(val_dataset)}, Test={len(test_dataset)}")
    
    print("Creating model with 2-GPU parallelism...")
    model = BERTForIMDB(
        vocab_size=tokenizer.vocab_size,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        max_len=args.max_len
    )
    print_model_info(model, "BERT for IMDB")
    
    print("Starting training process...")
    history = train_model(model, train_dataloader, val_dataloader, args)
    plot_training_history(history)
    print("Testing model on test set...")
    test_results = test_predictions(model, test_dataloader)
    plot_confusion_matrix(test_results['confusion_matrix'])
    visualize_samples(test_results['samples'])
    print("Demo complete!")

if __name__ == "__main__":
    main()
EOL

# Create a run script
cat > run_imdb_demo.sh << 'EOL'
#!/bin/bash
echo "Running BERT-like model with model parallelism for IMDB sentiment analysis on RunPod.io..."
python bert_imdb_parallel.py --epochs 2 --batch_size 32 --hidden_size 128 --num_layers 4

echo "Training complete. Creating visualizations..."
python bert_imdb_parallel.py --epochs 2 --batch_size 32 --hidden_size 128 --num_layers 4  # Run again to generate test results
EOL

# Add execute permissions
chmod +x bert_imdb_parallel.py
chmod +x run_imdb_demo.sh

# Run the demo
echo "Running BERT-like model with model parallelism for IMDB sentiment analysis on RunPod.io..."
python bert_imdb_parallel.py --epochs 2 --batch_size 32 --hidden_size 128 --num_layers 4

echo "Training complete. Creating visualizations..."
python bert_imdb_parallel.py --epochs 2 --batch_size 32 --hidden_size 128 --num_layers 4

echo "All done! Check the generated files:"
echo "- training_history.png - Training loss and validation accuracy curves"
echo "- confusion_matrix.png - Confusion matrix on test set"
echo "- sample_predictions.png - Examples of model predictions"