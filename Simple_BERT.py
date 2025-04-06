from transformers import BertTokenizer, BertModel, DistilBertTokenizer, DistilBertModel
from torch.utils.data import Dataset, DataLoader
import torch
import numpy as np
import pandas as pd
from torch.optim import Adam
import gc
from tqdm import tqdm

class DistilBERTDataset(Dataset):
    def __init__(self, texts, labels, tokenizer_name='distilbert-base-uncased', max_len=64):
        self.texts = texts
        self.labels = labels
        self.tokenizer = DistilBertTokenizer.from_pretrained(tokenizer_name)
        self.max_len = max_len

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding='max_length',
            max_length=self.max_len,
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'labels': torch.tensor(self.labels[idx])
        }

class MemoryMappedDataset(Dataset):
    def __init__(self, emb_path, label_path):
        self.embeddings = np.load(emb_path, mmap_mode='r')
        self.labels = np.load(label_path, mmap_mode='r')

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        emb = torch.from_numpy(self.embeddings[idx]).float()
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return emb, label

class BertFeatureExtractor:
    def __init__(self, model_name='distilbert-base-uncased'):
        if torch.backends.mps.is_available():
            self.device = torch.device('mps')
        else:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = DistilBertTokenizer.from_pretrained(model_name)
        self.model = DistilBertModel.from_pretrained(model_name).to(self.device)
        self.model.eval()

    def encode(self, texts, batch_size=128, max_length=64):
        all_embeddings = []
        with torch.no_grad():
            for i in tqdm(range(0, len(texts), batch_size)):
                batch = texts[i:i+batch_size]
                encoded = self.tokenizer(batch, padding=True, truncation=True, max_length=max_length, return_tensors='pt')
                input_ids = encoded['input_ids'].to(self.device)
                attention_mask = encoded['attention_mask'].to(self.device)
                outputs = self.model(input_ids, attention_mask=attention_mask)
                cls_embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy()
                all_embeddings.append(cls_embedding)
        return np.concatenate(all_embeddings, axis=0)  # shape: [N, 768]
    
    def save_bert_features(self, texts, labels, emb_file, label_file , batch_size=128, max_length=64):
        features = self.encode(texts, batch_size, max_length)
        np.save(emb_file, features.astype(np.float32))
        np.save(label_file, np.array(labels, dtype=np.int64))
    
class SimpleBERT(torch.nn.Module):
    def __init__(self, num_labels, model_path='distilbert-base-uncased'):
        super(SimpleBERT, self).__init__()
        if torch.backends.mps.is_available():
            self.device = torch.device('mps')
        else:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.dropout = torch.nn.Dropout(0.1)
        self.linear = torch.nn.Linear(768, num_labels)
        self.to(self.device)

    def forward(self, embedding):
        pooled_output = self.dropout(embedding)
        logits = self.linear(pooled_output)
        return logits
    
    def train_data(self, train_loader, val_loader, epochs = 10, lr = 0.001, patience = 3, save_path = 'best_model.pt'):
        optimizer = Adam(self.parameters(), lr=lr)
        criterion = torch.nn.CrossEntropyLoss()
        min_loss = float('inf')
        patience_counter = 0
        correct = 0
        total = 0
        total_loss = 0
        # Initialize lists to store accuracies and losses
        train_accuracies = []
        train_losses = []
        val_accuracies = []
        val_losses = []
        # Training loop
        for epoch in range(epochs):
            print(f'Epoch {epoch+1}/{epochs}')
            self.train()
            for emb, labels in train_loader:
                optimizer.zero_grad()
                labels = labels.to(self.device)
                logits = self.forward(emb.to(self.device))
                loss = criterion(logits, labels)
                total_loss += loss.item() * labels.size(0)
                loss.backward()
                optimizer.step()
                _, predicted = torch.max(logits, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                # Free memory
                del emb, labels, logits, loss, predicted
                gc.collect()
                if self.device.type == 'mps':
                    torch.mps.empty_cache()
            print(f'Epoch {epoch + 1}/{epochs} - Training Loss: {total_loss/total} - Training Accuracy: {correct/total}')
            train_accuracies.append(correct/total)
            train_losses.append(total_loss/total)
            # Validation
            self.eval()
            val_loss = 0
            correct = 0
            total = 0
            total_loss = 0
            with torch.no_grad():
                for emb, labels in val_loader:
                    labels = labels.to(self.device)
                    logits = self.forward(emb.to(self.device))
                    loss = criterion(logits, labels)
                    val_loss += loss.item()
                    _, predicted = torch.max(logits, 1)
                    total_loss += loss.item() * labels.size(0)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
                    # Free memory
                    del labels, logits, loss, predicted
                    gc.collect()
                    if self.device.type == 'mps':
                        torch.mps.empty_cache()

            print(f'Epoch {epoch+1}/{epochs} - Validation Loss: {total_loss/total} - Validation Accuracy: {correct/total}')
            val_accuracies.append(correct/total)
            val_losses.append(val_loss/len(val_loader))
            #early stopping criteria
            if val_loss < min_loss:
                min_loss = val_loss
                patience_counter = 0
                self.save_model(save_path)
                print(f'Saved best model at epoch {epoch+1}')
            else:
                patience_counter += 1
                if patience == patience_counter:
                    print(f'Early stopping at epoch {epoch + 1}')
                    return train_accuracies, train_losses, val_accuracies, val_losses

    def predict(self, texts):
        self.eval()
        texts.to(self.device)
        if isinstance(texts, pd.DataFrame):
            texts = texts['text'].tolist()
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        encodings = tokenizer(texts, truncation=True, padding=True, max_length=128, return_tensors='pt')
        input_ids = encodings['input_ids'].to(self.device)
        attention_mask = encodings['attention_mask'].to(self.device)
        with torch.no_grad():
            logits = self.forward(input_ids, attention_mask)
            _, predicted = torch.max(logits, 1)
            return predicted.numpy()
    
    def evaluate(self, test_loader):
        self.eval()
        correct = 0
        total = 0
        total_loss = 0
        criterion = torch.nn.CrossEntropyLoss()
        with torch.no_grad():
            for emb, labels in test_loader:
                labels = labels.to(self.device)
                logits = self.forward(emb.to(self.device))
                # Calculate loss
                loss = criterion(logits, labels)
                total_loss += loss.item() * labels.size(0)
                _, predicted = torch.max(logits, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                # Free memory
                del emb, labels, logits
                gc.collect()
                if self.device.type == 'mps':
                    torch.mps.empty_cache()
        return correct / total, total_loss/total
    
    def save_model(self, path):
        torch.save(self.state_dict(), path)

    def load_model(self, path):
        self.load_state_dict(torch.load(path))

    def save_BERT_model(self, path):
        self.bert.save_pretrained(path)

    
if __name__ == "__main__":
    # TODO add code for running the python module
    pass