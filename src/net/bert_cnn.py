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


class BertCNN(torch.nn.Module):
    def __init__(
        self,
        num_labels,
        num_layers=2,
        num_filters=64,
        n_grams=None,
        model_path="distilbert-base-uncased",
    ):
        super().__init__()

        if n_grams is None:
            n_grams = [3, 4]
        if len(n_grams) < num_layers:
            raise ValueError(
                "The length of n_grams must be at least equal to num_layers."
            )

        self.device = get_best_device()
        self.num_layers = num_layers
        self.bert = DistilBertModel.from_pretrained(model_path)
        for param in self.bert.parameters():
            param.requires_grad = False
        self.conv_layers = torch.nn.ModuleList()
        for i in range(num_layers):
            self.conv_layers.append(
                torch.nn.Conv2d(
                    1,
                    num_filters,
                    (n_grams[i], 768),
                ).to(self.device)
            )
        self.dropout = torch.nn.Dropout(0.5)
        self.fc = torch.nn.Linear(num_layers * num_filters, num_labels)
        self.to(self.device)

    def forward(self, input_ids, attention_mask):
        # Get the BERT embeddings
        outputs = self.bert(
            input_ids=input_ids, attention_mask=attention_mask
        ).last_hidden_state[
            :, 1:, :
        ]  # Exclude [CLS] token
        # Expand attention mask to match last_hidden_state shape
        mask = attention_mask[:, 1:].unsqueeze(-1).float()  # [B, L, 1]
        masked_embeddings = outputs * mask  # Zero out [PAD] token embeddings
        x = masked_embeddings.unsqueeze(1)  # Add a channel dimension for Conv2d
        _x = []
        for i in range(self.num_layers):
            _x.append(torch.nn.functional.relu(self.conv_layers[i](x)).squeeze(3))
            _x[i] = torch.nn.functional.max_pool1d(_x[i], _x[i].size(2)).squeeze(2)
        x = torch.cat([_x[i] for i in range(self.num_layers)], 1)
        x = self.dropout(x)
        x = self.fc(x)
        return x

    def train_model(
        self,
        train_dataloader,
        val_dataloader,
        num_epochs=100,
        learning_rate=0.001,
        patience=3,
        save_model=True,
        save_path="models/best_CNN_model.pt",
    ):
        if not os.path.exists(os.path.dirname(save_path)):
            os.makedirs(os.path.dirname(save_path))

        optimizer = Adam(self.parameters(), lr=learning_rate)
        criterion = torch.nn.CrossEntropyLoss()
        best_val_loss = float("inf")
        patience_counter = 0

        train_accuracies, train_losses, val_accuracies, val_losses = [], [], [], []

        best_state_dict = None

        for epoch in range(num_epochs):
            self.train()
            total_loss, correct, total = 0, 0, 0

            for batch in tqdm(train_dataloader):
                # Move the batch to the device
                input_ids = batch["input_ids"]
                attention_mask = batch["attention_mask"]
                labels = batch["labels"]
                input_ids, attention_mask, labels = (
                    input_ids.to(self.device),
                    attention_mask.to(self.device),
                    labels.to(self.device),
                )

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
            print(
                f"Epoch {epoch+1}/{num_epochs}, Train Loss: {avg_train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}"
            )

            val_loss = 0
            correct = 0
            total = 0
            self.eval()
            for batch in val_dataloader:
                input_ids = batch["input_ids"]
                attention_mask = batch["attention_mask"]
                labels = batch["labels"]
                input_ids, attention_mask, labels = (
                    input_ids.to(self.device),
                    attention_mask.to(self.device),
                    labels.to(self.device),
                )

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
            print(
                f"Epoch {epoch+1}/{num_epochs}, Val Loss: {avg_val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}"
            )

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                if save_model:
                    torch.save(self.state_dict(), save_path)
                    print(f"Saved best model at epoch {epoch+1}")
                else:
                    # store state dict in memory
                    best_state_dict = self.state_dict()
                    print("✔️ Stored best model in memory")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break

        if best_state_dict is not None:
            self.load_state_dict(best_state_dict)
            print("✔️ Restored best model from memory")

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
                input_ids = batch["input_ids"]
                attention_mask = batch["attention_mask"]
                labels = batch["labels"]
                input_ids, attention_mask, labels = (
                    input_ids.to(self.device),
                    attention_mask.to(self.device),
                    labels.to(self.device),
                )
                outputs = self.forward(input_ids, attention_mask)
                _, prediction = torch.max(outputs, 1)
                loss = criterion(outputs, labels)
                total_loss += loss.item() * labels.size(0)
                total += labels.size(0)
                correct += (prediction == labels).sum().item()
                predictions.extend(prediction.cpu())
                logits.append(outputs.detach().cpu())
        logits = torch.cat(logits, dim=0)  # Shape: [num_samples, num_classes]
        avg_loss = total_loss / total
        accuracy = correct / total

        return np.array(predictions), logits, avg_loss, accuracy
