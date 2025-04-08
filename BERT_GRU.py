import os
import time
import torch
import pickle
import numpy as np
import pandas as pd

from tqdm import tqdm
from torch.optim import Adam
from torch.utils.data import DataLoader
from transformers import DistilBertModel

from Simple_BERT import DistilBERTDataset
from utils import get_best_device


class BertGRU(torch.nn.Module):
    def __init__(
        self,
        num_labels,
        hidden_dim=64,
        num_layers=1,
        model_path="distilbert-base-uncased",
    ):
        super(BertGRU, self).__init__()

        self.device = get_best_device()

        self.bert = DistilBertModel.from_pretrained(model_path)
        for param in self.bert.parameters():
            param.requires_grad = False  # Freeze BERT weights

        self.gru = torch.nn.GRU(
            input_size=768,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
        )

        self.dropout = torch.nn.Dropout(0.5)
        self.fc = torch.nn.Linear(hidden_dim * 2, num_labels)  # *2 for bidirectional
        self.to(self.device)

    def forward(self, input_ids, attention_mask):
        # BERT embeddings
        outputs = self.bert(
            input_ids=input_ids, attention_mask=attention_mask
        ).last_hidden_state
        mask = attention_mask.unsqueeze(-1).float()
        masked_outputs = outputs * mask  # Mask padding tokens

        # RNN
        rnn_out, _ = self.gru(masked_outputs)
        pooled_output = rnn_out[:, -1, :]  # Take the last timestep

        x = self.dropout(pooled_output)
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
        save_path="models/best_GRU_model.pt",
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

            for batch in tqdm(
                train_dataloader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]"
            ):
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)

                optimizer.zero_grad()
                outputs = self.forward(input_ids, attention_mask)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                total_loss += loss.item() * labels.size(0)
                correct += (outputs.argmax(1) == labels).sum().item()
                total += labels.size(0)

            avg_train_loss = total_loss / total
            train_accuracy = correct / total
            train_losses.append(avg_train_loss)
            train_accuracies.append(train_accuracy)
            print(f"Train Loss: {avg_train_loss:.4f}, Accuracy: {train_accuracy:.4f}")

            self.eval()
            val_loss, correct, total = 0, 0, 0

            with torch.no_grad():
                for batch in tqdm(
                    val_dataloader, desc=f"Epoch {epoch+1}/{num_epochs} [Val]"
                ):
                    input_ids = batch["input_ids"].to(self.device)
                    attention_mask = batch["attention_mask"].to(self.device)
                    labels = batch["labels"].to(self.device)

                    outputs = self.forward(input_ids, attention_mask)
                    loss = criterion(outputs, labels)

                    val_loss += loss.item() * labels.size(0)
                    correct += (outputs.argmax(1) == labels).sum().item()
                    total += labels.size(0)

            avg_val_loss = val_loss / total
            val_accuracy = correct / total
            val_losses.append(avg_val_loss)
            val_accuracies.append(val_accuracy)
            print(f"Val Loss: {avg_val_loss:.4f}, Accuracy: {val_accuracy:.4f}")

            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
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
        logits = []
        criterion = torch.nn.CrossEntropyLoss()
        total_loss = 0
        correct = 0
        total = 0

        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)

                outputs = self.forward(input_ids, attention_mask)
                loss = criterion(outputs, labels)

                total_loss += loss.item() * labels.size(0)
                total += labels.size(0)
                correct += (outputs.argmax(1) == labels).sum().item()

                predictions.extend(outputs.argmax(1).cpu().numpy())
                logits.extend(outputs.detach().cpu())

        avg_loss = total_loss / total
        accuracy = correct / total
        return np.array(predictions), logits, avg_loss, accuracy


if __name__ == "__main__":

    print("Training BERT GRU: ")

    start_time = time.time()
    # read the data
    emotion_train = pd.read_csv("data/emotion_train.csv")
    emotion_val = pd.read_csv("data/emotion_val.csv")
    emotion_test = pd.read_csv("data/emotion_test.csv")

    # create Dataset
    emotion_GRU_train = DistilBERTDataset(
        emotion_train["text"].tolist(), emotion_train["label"].to_list()
    )
    emotion_GRU_val = DistilBERTDataset(
        emotion_val["text"].tolist(), emotion_val["label"].to_list()
    )
    emotion_GRU_test = DistilBERTDataset(
        emotion_test["text"].tolist(), emotion_test["label"].to_list()
    )

    # set seed for reproducibility
    np.random.seed(42)
    torch.manual_seed(42)

    # create DataLoader for each dataset
    emotion_GRU_train_data = DataLoader(emotion_GRU_train, batch_size=128, shuffle=True)
    emotion_GRU_val_data = DataLoader(emotion_GRU_val, batch_size=128, shuffle=False)
    emotion_GRU_test_data = DataLoader(emotion_GRU_test, batch_size=128, shuffle=False)

    print(f"Data loaded in {time.time() - start_time:.2f} seconds")

    # training the model
    emotion_GRU_model = BertGRU(num_labels=6)

    train_start_time = time.time()
    train_accuracies, train_losses, val_accuracies, val_losses = (
        emotion_GRU_model.train_model(
            train_dataloader=emotion_GRU_train_data,
            val_dataloader=emotion_GRU_val_data,
            num_epochs=100,
            patience=3,
        )
    )
    print(f"Training completed in {time.time() - train_start_time:.2f} seconds")

    eval_start_time = time.time()

    # evaluating the model
    test_predictions, test_logits, test_loss, test_accuracy = (
        emotion_GRU_model.evaluate(emotion_GRU_test_data)
    )
    print(f"Evaluation completed in {time.time() - eval_start_time:.2f} seconds")

    print(
        f"""
        BERT GRU Model Evaluation:
        Test Loss: {test_loss:.4f}
        Test Accuracy: {test_accuracy:.4f}
        Test Predictions: {test_predictions[:10]}
        Test Logits: {test_logits[:10]}
        """
    )

    if not os.path.exists("results"):
        os.makedirs("results")

    # save results as python objects
    with open("results/emotion_GRU_results.pkl", "wb") as f:
        pickle.dump(
            (
                train_accuracies,
                train_losses,
                val_accuracies,
                val_losses,
                test_predictions,
                test_logits,
                test_accuracy,
                test_loss,
            ),
            f,
        )
    print("Results saved to emotion_GRU_results.pkl")
    print(f"Total time taken: {time.time() - start_time:.2f} seconds")
