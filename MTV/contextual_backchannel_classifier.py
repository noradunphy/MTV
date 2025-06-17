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
import re

def clean_swda_text(text):
    """
    Remove SWDA transcription symbols but keep the content inside curly braces.
    For example, '{F uh, }' becomes 'uh, '.
    """
    # Remove hashtags and their content
    text = re.sub(r'#.*?#', '', text)
    # Replace {X content} with just 'content' (keep content, remove code and braces)
    text = re.sub(r'\{[A-Z] ([^}]*)\}', r'\1', text)
    # Remove angle-bracket annotations like <laughter>
    text = re.sub(r'<[^>]*>', '', text)
    # Remove slashes (intonation units)
    text = text.replace('/', '')
    # Remove plus, minus, brackets, and other common symbols
    text = re.sub(r'[\[\]\+\-\#]', '', text)
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

class ContextualBackchannelDataset(Dataset):
    def __init__(self, split="train", debug=True):
        self.dataset = load_dataset("swda", trust_remote_code=True)[split]
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        self.debug = debug
        
        if self.debug:
            print(f"\nInitializing {split} dataset with {len(self.dataset)} examples")
            print("\nExample data format:")
            self._print_example(0)
            self._print_example(1)
        
    def _print_example(self, idx):
        """Print a single example for debugging purposes."""
        item = self.dataset[idx]
        damsl_idx = item['damsl_act_tag']
        tag = self.dataset.features['damsl_act_tag'].names[damsl_idx]
        
        # Get previous turn if it exists
        prev_text = ""
        if idx > 0:
            prev_item = self.dataset[idx-1]
            prev_text = clean_swda_text(prev_item['text'])
        
        current_text = clean_swda_text(item['text'])
        combined_text = f"{prev_text} [SEP] {current_text}"
        
        print(f"\nExample {idx}:")
        print(f"Dialog Act Tag: {tag}")
        print(f"Previous Turn: {prev_text}")
        print(f"Current Turn: {current_text}")
        print(f"Combined Text: {combined_text}")
        print(f"Is Backchannel: {tag == 'b'}")
        
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        item = self.dataset[idx]
        
        # Get the dialog act tag from damsl_act_tag
        damsl_idx = item['damsl_act_tag']
        tag = self.dataset.features['damsl_act_tag'].names[damsl_idx]
        
        # Convert to binary label: 1 for backchannel (b), 0 for non-backchannel
        label = 1 if tag == 'b' else 0
        
        # Get the previous turn's text if it exists
        prev_text = ""
        if idx > 0:
            prev_item = self.dataset[idx-1]
            prev_text = clean_swda_text(prev_item['text'])
        
        # Clean and combine previous turn and current turn with a special separator
        current_text = clean_swda_text(item['text'])
        combined_text = f"{prev_text} [SEP] {current_text}"
        
        # Tokenize the combined text
        encoding = self.tokenizer(
            combined_text,
            padding='max_length',
            truncation=True,
            max_length=256,  # Increased max length to accommodate both turns
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'label': torch.tensor(label)
        }

class ContextualBackchannelClassifier(nn.Module):
    def __init__(self, hidden_size=768, model_path=None):
        super().__init__()
        self.bert = AutoModel.from_pretrained("bert-base-uncased")
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

        # Freeze BERT parameters
        for param in self.bert.parameters():
            param.requires_grad = False

        # Binary classification head with increased capacity
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 1),
            nn.Sigmoid()
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

    def classify_utterance(self, current_utterance, previous_utterance="", device="cuda", pre_cleaned=False):
        """
        Classify a single utterance as backchannel (1) or not (0), considering the previous turn.
        
        Args:
            current_utterance: The utterance to classify
            previous_utterance: The previous turn's text (optional)
            device: The device to run inference on
            pre_cleaned: If True, assumes the text is already cleaned (for model outputs)
        """
        self.eval()
        with torch.no_grad():
            # Clean text if needed
            if not pre_cleaned:
                current_utterance = clean_swda_text(current_utterance)
                previous_utterance = clean_swda_text(previous_utterance)
            
            # Combine utterances
            combined_text = f"{previous_utterance} [SEP] {current_utterance}"
            
            # Tokenize the combined text
            encoded = self.tokenizer(
                combined_text,
                padding="max_length",
                truncation=True,
                max_length=256,
                return_tensors="pt"
            )
            
            # Move to device
            input_ids = encoded["input_ids"].to(device)
            attention_mask = encoded["attention_mask"].to(device)
            
            # Get prediction
            outputs = self(input_ids, attention_mask)
            predicted = (outputs > 0.5).float()
            
            return predicted.item()

    def classify_batch(self, current_utterances, previous_utterances=None, device="cuda", pre_cleaned=False):
        """
        Classify a batch of utterances, optionally with their previous turns.
        
        Args:
            current_utterances: List of utterances to classify
            previous_utterances: List of previous turns (optional, defaults to empty strings)
            device: The device to run inference on
            pre_cleaned: If True, assumes the text is already cleaned (for model outputs)
        """
        self.eval()
        with torch.no_grad():
            if previous_utterances is None:
                previous_utterances = [""] * len(current_utterances)
            
            # Clean text if needed
            if not pre_cleaned:
                current_utterances = [clean_swda_text(u) for u in current_utterances]
                previous_utterances = [clean_swda_text(u) for u in previous_utterances]
            
            # Combine utterances
            combined_texts = [f"{prev} [SEP] {curr}" for prev, curr in zip(previous_utterances, current_utterances)]
            
            # Tokenize the combined texts
            encoded = self.tokenizer(
                combined_texts,
                padding="max_length",
                truncation=True,
                max_length=256,
                return_tensors="pt"
            )
            
            # Move to device
            input_ids = encoded["input_ids"].to(device)
            attention_mask = encoded["attention_mask"].to(device)
            
            # Get predictions
            outputs = self(input_ids, attention_mask)
            predicted = (outputs > 0.5).float()
            
            return predicted.squeeze().cpu().numpy()

def train_model(model, train_loader, val_loader, num_epochs=3, device="cuda", save_path="best_contextual_backchannel_model.pth"):
    criterion = nn.BCELoss()
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

        train_pbar = tqdm(train_loader, desc=f"Training")
        for batch_idx, batch in enumerate(train_pbar):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].float().to(device)

            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask)
            loss = criterion(outputs.squeeze(), labels)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            predicted = (outputs.squeeze() > 0.5).float()
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

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
                labels = batch["label"].float().to(device)

                outputs = model(input_ids, attention_mask)
                loss = criterion(outputs.squeeze(), labels)
                val_loss += loss.item()

                predicted = (outputs.squeeze() > 0.5).float()
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

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

def train_and_save_classifier(save_path="best_contextual_backchannel_model.pth", num_epochs=3):
    """Train the contextual backchannel classifier and save it for later use."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print("\nLoading datasets...")
    train_dataset = ContextualBackchannelDataset(split="train", debug=True)
    val_dataset = ContextualBackchannelDataset(split="validation", debug=True)
    test_dataset = ContextualBackchannelDataset(split="test", debug=True)

    # Print dataset statistics
    print("\nDataset Statistics:")
    print(f"Training set size: {len(train_dataset)}")
    print(f"Validation set size: {len(val_dataset)}")
    print(f"Test set size: {len(test_dataset)}")
    
    # Count backchannels more efficiently by checking the raw dataset
    train_bc = sum(1 for item in train_dataset.dataset if train_dataset.dataset.features['damsl_act_tag'].names[item['damsl_act_tag']] == 'b')
    val_bc = sum(1 for item in val_dataset.dataset if val_dataset.dataset.features['damsl_act_tag'].names[item['damsl_act_tag']] == 'b')
    test_bc = sum(1 for item in test_dataset.dataset if test_dataset.dataset.features['damsl_act_tag'].names[item['damsl_act_tag']] == 'b')
    
    print("\nBackchannel Distribution:")
    print(f"Training set: {train_bc} backchannels ({train_bc/len(train_dataset)*100:.2f}%)")
    print(f"Validation set: {val_bc} backchannels ({val_bc/len(val_dataset)*100:.2f}%)")
    print(f"Test set: {test_bc} backchannels ({test_bc/len(test_dataset)*100:.2f}%)")

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

    val_loader = DataLoader(
        val_dataset,
        batch_size=32,
        shuffle=False,
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
            key: torch.stack([d[key] for d in x])
            for key in x[0].keys()
        }
    )

    print("\nInitializing model...")
    model = ContextualBackchannelClassifier().to(device)

    # Train model
    train_model(model, train_loader, val_loader, num_epochs=num_epochs, device=device, save_path=save_path)

def load_classifier(model_path="best_contextual_backchannel_model.pth"):
    """Load a trained contextual backchannel classifier for inference."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Initialize and load the model
    model = ContextualBackchannelClassifier(model_path=model_path).to(device)
    model.eval()
    
    return model

if __name__ == "__main__":
    # Train the model
    train_and_save_classifier() 