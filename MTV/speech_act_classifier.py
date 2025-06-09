        #####
        #my Code 
        #####

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
from datasets import load_dataset
import numpy as np
from sklearn.metrics import classification_report
import torch.nn.functional as F
from tqdm import tqdm
import time
from datetime import datetime
import os

class SWDADataset(Dataset):
    def __init__(self, split="train", tag2idx=None):
        self.dataset = load_dataset("swda", trust_remote_code=True)[split]
        
        # Get the mapping from damsl_act_tag indices to names
        self.damsl_tag_names = self.dataset.features['damsl_act_tag'].names
        
        # Create tag2idx mapping if not provided
        if tag2idx is None:
            # Use damsl_act_tag names as our labels
            unique_tags = set(self.damsl_tag_names)
            self.tag2idx = {tag: idx for idx, tag in enumerate(sorted(unique_tags))}
        else:
            self.tag2idx = tag2idx
            
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        item = self.dataset[idx]
        
        # Get the dialog act tag from damsl_act_tag
        damsl_idx = item['damsl_act_tag']
        tag = self.damsl_tag_names[damsl_idx]
        
        # Convert tag to index
        label = self.tag2idx.get(tag, -1)  # -1 for unknown tags
        
        # Tokenize the text
        encoding = self.tokenizer(
            item['text'],
            padding='max_length',
            truncation=True,
            max_length=128,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'label': torch.tensor(label)
        }
    
    # 2. Create the MLP Classifier
class SpeechActClassifier(nn.Module):
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
        # Get BERT embeddings
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        # Use [CLS] token embedding
        pooled_output = outputs.last_hidden_state[:, 0, :]
        # Pass through classifier
        logits = self.classifier(pooled_output)
        return logits

    def classify_utterance(self, utterance, device="cuda"):
        """Classify a single utterance into a dialog act."""
        self.eval()
        with torch.no_grad():
            # Tokenize the utterance
            encoded = self.tokenizer(
                utterance,
                padding="max_length",
                truncation=True,
                max_length=128,
                return_tensors="pt"
            )
            
            # Move to device
            input_ids = encoded["input_ids"].to(device)
            attention_mask = encoded["attention_mask"].to(device)
            
            # Get prediction
            outputs = self(input_ids, attention_mask)
            _, predicted = torch.max(outputs.data, 1)
            
            return predicted.item()

def train_model(model, train_loader, val_loader, num_epochs=3, device="cuda", save_path="best_model.pth"):
    criterion = nn.CrossEntropyLoss()
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

        # Create progress bar for training
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

            # Update progress bar
            current_loss = total_loss / (batch_idx + 1)
            current_acc = 100. * correct / total
            train_pbar.set_postfix({
                'loss': f'{current_loss:.4f}',
                'acc': f'{current_acc:.2f}%'
            })

        # Validation phase
        model.eval()
        val_correct = 0
        val_total = 0
        val_loss = 0

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

                # Update progress bar
                current_val_acc = 100. * val_correct / val_total
                val_pbar.set_postfix({
                    'loss': f'{val_loss/(batch_idx+1):.4f}',
                    'acc': f'{current_val_acc:.2f}%'
                })

        # Calculate final metrics for this epoch
        train_acc = 100 * correct / total
        val_acc = 100 * val_correct / val_total
        epoch_time = time.time() - epoch_start

        print(f"\nEpoch {epoch+1} Summary:")
        print(f"Training Loss: {total_loss/len(train_loader):.4f}")
        print(f"Training Accuracy: {train_acc:.2f}%")
        print(f"Validation Loss: {val_loss/len(val_loader):.4f}")
        print(f"Validation Accuracy: {val_acc:.2f}%")
        print(f"Epoch Time: {epoch_time:.2f} seconds")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), save_path)
            print(f"New best model saved! (Validation Accuracy: {val_acc:.2f}%)")

    total_time = time.time() - start_time
    print(f"\nTraining completed in {total_time/60:.2f} minutes")
    print(f"Best validation accuracy: {best_val_acc:.2f}%")

def train_and_save_classifier(save_path="best_model.pth", num_epochs=3):
    """Train the classifier and save it for later use."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print("\nLoading datasets...")
    train_dataset = SWDADataset(split="train")
    test_dataset = SWDADataset(split="test", tag2idx=train_dataset.tag2idx)

    # Save the tag2idx mapping
    torch.save(train_dataset.tag2idx, "tag2idx.pth")

    print("\nCreating dataloaders...")
    train_loader = DataLoader(
        train_dataset,
        batch_size=32,
        shuffle=True,
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
    model = SpeechActClassifier(num_classes=num_classes).to(device)

    # Train model
    train_model(model, train_loader, test_loader, num_epochs=num_epochs, device=device, save_path=save_path)

def load_classifier(model_path="best_model.pth", tag2idx_path="tag2idx.pth"):
    """Load a trained classifier for inference."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load the tag2idx mapping
    tag2idx = torch.load(tag2idx_path)
    num_classes = len(tag2idx)
    
    # Initialize and load the model
    model = SpeechActClassifier(num_classes=num_classes, model_path=model_path).to(device)
    model.eval()
    
    return model, tag2idx

if __name__ == "__main__":
    # Train the model
    train_and_save_classifier()