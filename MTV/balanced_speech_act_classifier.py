import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from transformers import AutoTokenizer, AutoModel
from datasets import load_dataset
import numpy as np
from collections import Counter
from sklearn.metrics import classification_report
from download_swda import clean_swda_text
import torch.nn.functional as F
from tqdm import tqdm
import time
from datetime import datetime
import os
import re

class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, weight=None):
        super().__init__()
        self.gamma = gamma
        self.weight = weight
        self.ce = nn.CrossEntropyLoss(weight=weight, reduction='none')
    
    def forward(self, logits, targets):
        ce_loss = self.ce(logits, targets)
        pt = torch.exp(-ce_loss)  # prob of true class
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        return focal_loss.mean()

class SWDADataset(Dataset):
    def __init__(self, split="train", tag2idx=None):
        raw_dataset = load_dataset("swda", trust_remote_code=True)[split]
        self.damsl_tag_names = raw_dataset.features['damsl_act_tag'].names
        
        # Build or reuse tag-to-index mapping
        if tag2idx is None:
            unique_tags = set(self.damsl_tag_names)
            self.tag2idx = {tag: idx for idx, tag in enumerate(sorted(unique_tags))}
        else:
            self.tag2idx = tag2idx
            
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        
        # Pre-filter utterances whose cleaned text is empty
        self.samples = []  # each item: {"text": str, "label": int}
        for item in raw_dataset:
            cleaned = clean_swda_text(item["text"])
            if cleaned == "":
                continue  # skip empty utterances

            tag = self.damsl_tag_names[item['damsl_act_tag']]
            label = self.tag2idx.get(tag, -1)
            self.samples.append({"text": cleaned, "label": label})

        # Store labels for sampling weights / metrics
        self.labels = [sample["label"] for sample in self.samples]
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]

        encoding = self.tokenizer(
            sample["text"],
            padding='max_length',
            truncation=True,
            max_length=128,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'label': torch.tensor(sample['label'])
        }

class BalancedSpeechActClassifier(nn.Module):
    def __init__(self, num_classes=None, hidden_size=768, model_path=None):
        super().__init__()
        self.bert = AutoModel.from_pretrained("bert-base-uncased")
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

        # Freeze BERT parameters
        for param in self.bert.parameters():
            param.requires_grad = False

        if num_classes is not None:
            self.classifier = nn.Sequential(
                nn.Linear(hidden_size, 512),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(512, num_classes)
            )
        
        if model_path and os.path.exists(model_path):
            self.load_state_dict(torch.load(model_path))
            print(f"Loaded model from {model_path}")

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.last_hidden_state[:, 0, :]
        logits = self.classifier(pooled_output)
        return logits

    def classify_utterance(self, utterance, device="cuda"):
        self.eval()
        with torch.no_grad():
            encoded = self.tokenizer(
                utterance,
                padding="max_length",
                truncation=True,
                max_length=128,
                return_tensors="pt"
            )
            
            input_ids = encoded["input_ids"].to(device)
            attention_mask = encoded["attention_mask"].to(device)
            
            outputs = self(input_ids, attention_mask)
            probabilities = F.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs.data, 1)
            
            return predicted.item(), probabilities.squeeze()

def compute_class_weights(dataset):
    """Compute inverse frequency weights for each class"""
    label_counts = Counter(dataset.labels)
    num_classes = len(dataset.tag2idx)
    
    # Get counts for each class
    counts = np.array([label_counts[i] for i in range(num_classes)])
    
    # Compute inverse frequency weights
    weights = 1.0 / counts
    weights = weights / weights.sum() * num_classes  # normalize
    
    return torch.tensor(weights, dtype=torch.float)

def train_model(model, train_loader, val_loader, class_weights, num_epochs=3, 
                device="cuda", save_path="best_balanced_model.pth"):
    
    criterion = FocalLoss(gamma=2.0, weight=class_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)

    best_val_acc = 0
    start_time = time.time()

    print(f"\nStarting training at {datetime.now().strftime('%H:%M:%S')}")
    print(f"Training on {len(train_loader.dataset)} examples")
    print(f"Validating on {len(val_loader.dataset)} examples")

    for epoch in range(num_epochs):
        epoch_start = time.time()
        print(f"\nEpoch {epoch+1}/{num_epochs}")

        # Training phase
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        all_preds = []
        all_labels = []

        train_pbar = tqdm(train_loader, desc=f"Training")
        for batch_idx, batch in enumerate(train_pbar):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            current_loss = total_loss / (batch_idx + 1)
            current_acc = 100. * correct / total
            train_pbar.set_postfix({
                'loss': f'{current_loss:.4f}',
                'acc': f'{current_acc:.2f}%'
            })

        # Print training metrics including per-class performance
        print("\nTraining Metrics:")
        print(classification_report(all_labels, all_preds))

        # Validation phase
        model.eval()
        val_correct = 0
        val_total = 0
        val_loss = 0
        val_preds = []
        val_labels = []

        print("\nRunning validation...")
        with torch.no_grad():
            val_pbar = tqdm(val_loader, desc="Validation")
            for batch_idx, batch in enumerate(val_pbar):
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["label"].to(device)

                outputs = model(input_ids, attention_mask)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
                
                val_preds.extend(predicted.cpu().numpy())
                val_labels.extend(labels.cpu().numpy())

                current_val_acc = 100. * val_correct / val_total
                val_pbar.set_postfix({
                    'loss': f'{val_loss/(batch_idx+1):.4f}',
                    'acc': f'{current_val_acc:.2f}%'
                })

        # Print validation metrics including per-class performance
        print("\nValidation Metrics:")
        print(classification_report(val_labels, val_preds))

        val_acc = 100 * val_correct / val_total
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), save_path)
            print(f"New best model saved! (Validation Accuracy: {val_acc:.2f}%)")

        epoch_time = time.time() - epoch_start
        print(f"\nEpoch {epoch+1} Time: {epoch_time:.2f} seconds")

    total_time = time.time() - start_time
    print(f"\nTraining completed in {total_time/60:.2f} minutes")
    print(f"Best validation accuracy: {best_val_acc:.2f}%")

def train_and_save_classifier(save_path="best_balanced_model.pth", num_epochs=3):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print("\nLoading datasets...")
    train_dataset = SWDADataset(split="train")
    test_dataset = SWDADataset(split="test", tag2idx=train_dataset.tag2idx)

    # Save the tag2idx mapping
    torch.save(train_dataset.tag2idx, "tag2idx.pth")

    # Compute class weights
    class_weights = compute_class_weights(train_dataset).to(device)
    print("\nClass weights:", class_weights)

    # Create weighted sampler for training data
    class_counts = np.bincount(train_dataset.labels)
    weights = 1.0 / class_counts[train_dataset.labels]
    sampler = WeightedRandomSampler(weights, len(weights), replacement=True)

    print("\nCreating dataloaders...")
    train_loader = DataLoader(
        train_dataset,
        batch_size=32,
        sampler=sampler,  # Use weighted sampler
        collate_fn=lambda x: {
            key: torch.stack([d[key] for d in x])
            for key in x[0].keys()
        }
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=32,
        shuffle=False,
        collate_fn=lambda x: {
            key: torch.stack([d[key] for d in x if d["label"] != -1])
            for key in x[0].keys()
        }
    )

    print("\nInitializing model...")
    num_classes = len(train_dataset.tag2idx)
    model = BalancedSpeechActClassifier(num_classes=num_classes).to(device)

    # Train model with class weights
    train_model(model, train_loader, test_loader, class_weights, 
                num_epochs=num_epochs, device=device, save_path=save_path)

def load_classifier(model_path="best_balanced_model.pth", tag2idx_path="tag2idx.pth"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    tag2idx = torch.load(tag2idx_path)
    num_classes = len(tag2idx)
    
    model = BalancedSpeechActClassifier(num_classes=num_classes, model_path=model_path).to(device)
    model.eval()
    
    return model, tag2idx

def clean_swda_text(text):
    """Clean SWDA transcription symbols but keep the spoken content."""
    text = re.sub(r'#.*?#', '', text)                     # remove hashtags and their content
    text = re.sub(r'\{[A-Z] ([^}]*)\}', r'\1', text)   # keep content inside {...}
    text = re.sub(r'<[^>]*>', '', text)                  # remove <laughter> etc.
    text = text.replace('/', '')                         # remove intonation slashes
    text = re.sub(r'[\[\]\+\-\#]', '', text)        # remove misc symbols
    text = re.sub(r'\s+([.,!?])', r'\1', text)         # remove space before punctuation
    text = re.sub(r'[{}()]', '', text)                   # remove remaining braces / parens
    text = re.sub(r'\s+', ' ', text)                    # collapse whitespace
    if text.strip() == '.':
        return ''
    return text.strip()

if __name__ == "__main__":
    train_and_save_classifier() 