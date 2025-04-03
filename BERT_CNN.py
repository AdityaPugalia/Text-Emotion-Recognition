from transformers import DistilBertModel
from torch.utils.data import Dataset, DataLoader
import torch
import numpy as np
import pandas as pd
from torch.optim import Adam
from tqdm import tqdm




class BertCNN(torch.nn.Module):
    def __init__(self, num_labels, model_path='distilbert-base-uncased'):
        super(BertCNN, self).__init__()
        if torch.backends.mps.is_available():
            self.device = torch.device('mps')
        else:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.bert = DistilBertModel.from_pretrained(model_path)
        for param in self.bert.parameters():
            param.requires_grad = False
        self.conv1 = torch.nn.Conv2d(1, 64, (3, 768))
        self.conv2 = torch.nn.Conv2d(1, 64, (4, 768))
        self.dropout = torch.nn.Dropout(0.5)
        self.fc = torch.nn.Linear(128, num_labels)
        self.to(self.device)
    
    def forward(self, input_ids, attention_mask):
        # Get the BERT embeddings
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state[:, 1:, :]  # Exclude [CLS] token
        # Expand attention mask to match last_hidden_state shape
        mask = attention_mask[:, 1:].unsqueeze(-1).float()  # [B, L, 1]
        masked_embeddings = outputs * mask  # Zero out [PAD] token embeddings
        x = masked_embeddings.unsqueeze(1)  # Add a channel dimension for Conv2d
        x1 = torch.nn.functional.relu(self.conv1(x)).squeeze(3)
        x2 = torch.nn.functional.relu(self.conv2(x)).squeeze(3)
        x1 = torch.nn.functional.max_pool1d(x1, x1.size(2)).squeeze(2)
        x2 = torch.nn.functional.max_pool1d(x2, x2.size(2)).squeeze(2)
        x = torch.cat((x1, x2), 1)
        x = self.dropout(x)
        x = self.fc(x)
        return x
    
    def train_CNN(self, train_dataloader, val_dataloader, num_epochs = 100, learning_rate = 0.001, patience = 3, save_path = 'models/best_CNN_model.pt'):
        optimizer = Adam(self.parameters(), lr=learning_rate)
        criterion = torch.nn.CrossEntropyLoss()
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(num_epochs):
            self.train()
            total_loss = 0
            correct = 0
            total = 0
            train_accuracies = []
            train_losses = []
            val_accuracies = []
            val_losses = []
            for batch in tqdm(train_dataloader):
                # Move the batch to the device
                input_ids = batch['input_ids']
                attention_mask = batch['attention_mask']
                labels = batch['labels']
                input_ids, attention_mask, labels = input_ids.to(self.device), attention_mask.to(self.device), labels.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.forward(input_ids, attention_mask)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                _, prediction = torch.max(outputs, 1)
                correct += (prediction == labels).sum().item()
                total += labels.size(0)
                total_loss += loss.item() * labels.size(0)
            
            avg_train_loss = total_loss / total
            train_accuracy = correct / total
            train_accuracies.append(train_accuracy)
            train_losses.append(avg_train_loss)
            print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {avg_train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}')
            
            val_loss = 0
            correct = 0
            total = 0
            self.eval()
            for batch in val_dataloader:
                input_ids = batch['input_ids']
                attention_mask = batch['attention_mask']
                labels = batch['labels']
                input_ids, attention_mask, labels = input_ids.to(self.device), attention_mask.to(self.device), labels.to(self.device)
                
                with torch.no_grad():
                    outputs = self.forward(input_ids, attention_mask)
                    loss = criterion(outputs, labels)
                    _, prediction = torch.max(outputs, 1)
                    correct += (prediction == labels).sum().item()
                    total += labels.size(0)
                    val_loss += loss.item() * labels.size(0)
            avg_val_loss = val_loss / total
            val_accuracy = correct / total
            val_accuracies.append(val_accuracy)
            val_losses.append(avg_val_loss)
            print(f'Epoch {epoch+1}/{num_epochs}, Val Loss: {avg_val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}')
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                torch.save(self.state_dict(), save_path)
                print(f'Saved best model at epoch {epoch+1}')
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f'Early stopping at epoch {epoch+1}')
                    break
        return train_accuracies, train_losses, val_accuracies, val_losses
    
    def evaluate(self, dataloader):
        self.eval()
        predictions = []
        criterion = torch.nn.CrossEntropyLoss()
        total_loss = 0
        correct = 0
        total = 0
        logits = []
        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch['input_ids']
                attention_mask = batch['attention_mask']
                labels = batch['labels']
                input_ids, attention_mask, labels = input_ids.to(self.device), attention_mask.to(self.device), labels.to(self.device)
                outputs = self.forward(input_ids, attention_mask)
                _, prediction = torch.max(outputs, 1)
                loss = criterion(outputs, labels)
                total_loss += loss.item() * labels.size(0)
                total += labels.size(0)
                correct += (prediction == labels).sum().item()
                predictions.extend(prediction.cpu().numpy())
                logits.extend(outputs.detach().cpu().numpy())
        avg_loss = total_loss / total
        accuracy = correct / total
        
        return np.array(predictions), logits, avg_loss, accuracy