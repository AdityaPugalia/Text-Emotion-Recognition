import os
import pickle
import time
import numpy as np
import pandas as pd
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import DistilBertModel

from src.datasets import DistilBERTDataset
from src.utils import get_best_device


class BertLSTM(torch.nn.Module):
    def __init__(
        self,
        num_labels,
        hidden_dim=64,
        num_layers=1,
        model_path="distilbert-base-uncased",
    ):
        super(BertLSTM, self).__init__()

        self.device = get_best_device()

        self.bert = DistilBertModel.from_pretrained(model_path)
        for param in self.bert.parameters():
            param.requires_grad = False  # Freeze BERT weights

        self.lstm = torch.nn.LSTM(
            input_size=768,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
        )

        self.dropout = torch.nn.Dropout(0.5)
        self.fc = torch.nn.Linear(
            hidden_dim * 2, num_labels
        )  # bidirectional = hidden*2
        self.to(self.device)

    def forward(self, input_ids, attention_mask):
        # BERT embeddings
        bert_output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        embeddings = bert_output.last_hidden_state[:, 1:, :]  # Skip [CLS] token

        # Mask out PAD tokens
        mask = attention_mask[:, 1:].unsqueeze(-1).float()
        masked_embeddings = embeddings * mask

        lstm_out, (hn, cn) = self.lstm(masked_embeddings)
        final_hidden = lstm_out[:, -1, :]  # Last time step

        x = self.dropout(final_hidden)
        x = self.fc(x)
        return x

    def train_model(
        self,
        train_dataloader,
        val_dataloader,
        num_epochs=10,
        learning_rate=0.001,
        patience=3,
        save_model=True,
        save_path="models/best_LSTM_model.pt",
    ):
        optimizer = Adam(self.parameters(), lr=learning_rate)
        criterion = torch.nn.CrossEntropyLoss()
        best_val_loss = float("inf")
        patience_counter = 0

        train_accuracies, train_losses = [], []
        val_accuracies, val_losses = [], []

        best_state_dict = None

        for epoch in range(num_epochs):
            self.train()
            total_loss, correct, total = 0, 0, 0

            for batch in tqdm(train_dataloader):
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)

                optimizer.zero_grad()
                outputs = self.forward(input_ids, attention_mask)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                preds = torch.argmax(outputs, dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
                total_loss += loss.item() * labels.size(0)

            avg_train_loss = total_loss / total
            train_acc = correct / total
            train_losses.append(avg_train_loss)
            train_accuracies.append(train_acc)
            print(
                f"[Epoch {epoch+1}] Train Loss: {avg_train_loss:.4f}, Accuracy: {train_acc:.4f}"
            )

            # Validation
            self.eval()
            val_loss, correct, total = 0, 0, 0
            with torch.no_grad():
                for batch in val_dataloader:
                    input_ids = batch["input_ids"].to(self.device)
                    attention_mask = batch["attention_mask"].to(self.device)
                    labels = batch["labels"].to(self.device)

                    outputs = self.forward(input_ids, attention_mask)
                    loss = criterion(outputs, labels)
                    preds = torch.argmax(outputs, dim=1)
                    correct += (preds == labels).sum().item()
                    total += labels.size(0)
                    val_loss += loss.item() * labels.size(0)

            avg_val_loss = val_loss / total
            val_acc = correct / total
            val_losses.append(avg_val_loss)
            val_accuracies.append(val_acc)
            print(
                f"[Epoch {epoch+1}] Val Loss: {avg_val_loss:.4f}, Accuracy: {val_acc:.4f}"
            )

            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                if save_model:
                    torch.save(self.state_dict(), save_path)
                    print("✔️ Saved best model")
                else:
                    # store state dict in memory
                    best_state_dict = self.state_dict()
                    print("✔️ Stored best model in memory")

            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print("⏹️ Early stopping triggered")
                    break

        # restore best state dict
        if best_state_dict is not None:
            self.load_state_dict(best_state_dict)
            print("✔️ Restored best model from memory")

        return train_accuracies, train_losses, val_accuracies, val_losses

    def evaluate(self, dataloader):
        self.eval()
        predictions, logits = [], []
        criterion = torch.nn.CrossEntropyLoss()
        total_loss, correct, total = 0, 0, 0

        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)

                outputs = self.forward(input_ids, attention_mask)
                loss = criterion(outputs, labels)
                preds = torch.argmax(outputs, dim=1)

                predictions.extend(preds.cpu().numpy())
                total_loss += loss.item() * labels.size(0)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
                logits.append(outputs.detach().cpu())
        logits = torch.cat(logits, dim=0)  # Shape: [num_samples, num_classes]
        avg_loss = total_loss / total
        accuracy = correct / total
        return np.array(predictions), logits, avg_loss, accuracy
