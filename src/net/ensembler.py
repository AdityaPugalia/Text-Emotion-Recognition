import numpy as np

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.utils import get_best_device
from src.datasets import EnsembleDataset, DistilBERTDataset


class EnsembleLearner(torch.nn.Module):
    def __init__(self, num_labels, model_list):
        super(EnsembleLearner, self).__init__()
        self.device = get_best_device()
        self.models = [model.to(self.device) for model in model_list]
        self.fc = torch.nn.Linear(len(model_list) * num_labels, num_labels)
        self.to(self.device)

    def forward(self, logits):
        logits = logits.view(
            logits.size(0), -1
        )  # [batch_size, num_models * num_labels]
        logits = self.fc(logits)
        return logits

    def train_ensemble(
        self,
        train_data,
        val_data,
        model_path,
        num_epochs=100,
        learning_rate=0.001,
        patience=3,
    ):
        # create emotion dataset
        train_loader = DataLoader(train_data, batch_size=128, shuffle=False)
        val_loader = DataLoader(val_data, batch_size=128, shuffle=False)

        # retreive the logits from the ensemble models
        train_logits = []
        val_logits = []
        for model in self.models:
            _, train_logit, _, _ = model.evaluate(train_loader)
            _, val_logit, _, _ = model.evaluate(val_loader)
            train_logit = torch.softmax(train_logit, dim=1)
            val_logit = torch.softmax(val_logit, dim=1)
            train_logits.append(train_logit)
            val_logits.append(val_logit)
        # stack the logits
        train_logits = torch.stack(train_logits)
        val_logits = torch.stack(val_logits)

        # create the ensemble dataset consisting of logits and labels
        train_dataset = EnsembleDataset(train_logits, torch.tensor(train_data.labels))
        val_dataset = EnsembleDataset(val_logits, torch.tensor(val_data.labels))
        train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False)

        # initialize the optimizer and loss function
        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        criterion = torch.nn.CrossEntropyLoss()
        patience_counter = 0
        max_loss = float("inf")

        train_accuracies = []
        val_accuracies = []
        train_losses = []
        val_losses = []

        # training loop
        for epoch in range(num_epochs):
            self.train()
            total_loss = 0
            correct = 0
            total = len(train_dataset)
            for batch in tqdm(train_loader):
                optimizer.zero_grad()
                logits, labels = batch
                logits = logits.to(self.device)
                labels = labels.to(self.device)
                outputs = self(logits)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * labels.size(0)
                _, predicted = torch.max(outputs, 1)
                correct += (predicted == labels).sum().item()
            accuracy = correct / total
            avg_loss = total_loss / len(train_loader)
            print(
                f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}"
            )
            train_accuracies.append(accuracy)
            train_losses.append(avg_loss)

            # Validation
            self.eval()
            val_loss = 0
            correct = 0
            total = len(val_dataset)
            with torch.no_grad():
                for batch in val_loader:
                    logits, labels = batch
                    logits = logits.to(self.device)
                    labels = labels.to(self.device)
                    outputs = self(logits)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item() * labels.size(0)
                    _, predicted = torch.max(outputs, 1)
                    correct += (predicted == labels).sum().item()
            val_accuracy = correct / total
            avg_val_loss = val_loss / len(val_loader)
            print(
                f"Validation Loss: {avg_val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}"
            )
            val_accuracies.append(val_accuracy)
            val_losses.append(avg_val_loss)

            # Early stopping
            if avg_val_loss < max_loss:
                max_loss = avg_val_loss
                patience_counter = 0
                print("Saving model...")
                torch.save(self.state_dict(), model_path)
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print("Early stopping")
                    break

        return train_accuracies, train_losses, val_accuracies, val_losses

    def evaluate(self, test_data):
        test_loader = DataLoader(test_data, batch_size=128, shuffle=False)

        logits = []
        for model in self.models:
            _, logit, _, _ = model.evaluate(test_loader)
            logit = torch.softmax(logit, dim=1)
            logits.append(logit)
        logits = torch.stack(logits)

        test_logits_data = EnsembleDataset(logits, torch.tensor(test_data.labels))
        test_loader = DataLoader(test_logits_data, batch_size=128, shuffle=False)
        self.eval()
        predictions = []
        criterion = torch.nn.CrossEntropyLoss()
        total_loss = 0
        correct = 0
        total = len(test_logits_data)
        with torch.no_grad():
            for batch in test_loader:
                logits, labels = batch
                logits = logits.to(self.device)
                outputs = self(logits)
                _, predicted = torch.max(outputs, 1)
                loss = criterion(outputs, labels)
                total_loss += loss.item() * labels.size(0)
                correct += (predicted == labels).sum().item()
                predictions.extend(predicted.cpu().numpy())
        accuracy = correct / total
        avg_loss = total_loss / total
        return np.array(predictions), avg_loss, accuracy


class SimpleEnsembler:
    def __init__(self, model_list):
        self.device = get_best_device()
        print(type(model_list))
        print(type(model_list[0]))
        self.models = [model.to(self.device) for model in model_list]
        self.ensemble_learner = None

    def mean_ensemble(self, test_data: DistilBERTDataset):
        test_loader = DataLoader(test_data, batch_size=128, shuffle=False)
        logits = []
        total_loss = 0
        correct = 0
        total = len(test_data)
        criterion = torch.nn.CrossEntropyLoss()
        for model in self.models:
            _, logit, _, _ = model.evaluate(test_loader)
            logit = torch.softmax(logit, dim=1)
            logits.append(logit)
        logits = torch.stack(logits).mean(dim=0)
        predictions = torch.argmax(logits, dim=1)
        for i in range(total):
            if predictions[i] == test_data.labels[i]:
                correct += 1
            total_loss += criterion(logits[i], torch.tensor(test_data.labels[i]))
        accuracy = correct / total
        avg_loss = total_loss / total
        return predictions, logits, avg_loss, accuracy

    def weighted_ensemble(self, test_data: DistilBERTDataset):
        test_loader = DataLoader(test_data, batch_size=128, shuffle=False)
        logits = []
        weights = []
        total_loss = 0
        correct = 0
        total = len(test_data)
        criterion = torch.nn.CrossEntropyLoss()
        for model in self.models:
            _, logit, _, accuracy = model.evaluate(test_loader)
            logit = torch.softmax(logit, dim=1)
            logits.append(logit)
            weights.append(accuracy)
        weights = torch.softmax(torch.tensor(weights), dim=0)
        logits = torch.stack(logits)
        logits = torch.sum(logits * weights.view(-1, 1, 1), dim=0)
        predictions = torch.argmax(logits, dim=1)
        for i in range(total):
            if predictions[i] == test_data.labels[i]:
                correct += 1
            total_loss += criterion(logits[i], torch.tensor(test_data.labels[i]))
        accuracy = correct / total
        avg_loss = total_loss / total
        return predictions, logits, avg_loss, accuracy

    def max_ensemble(self, test_data: DistilBERTDataset):
        test_loader = DataLoader(test_data, batch_size=128, shuffle=False)
        logits = []
        total_loss = 0
        correct = 0
        total = len(test_data)
        criterion = torch.nn.CrossEntropyLoss()
        for model in self.models:
            _, logit, _, _ = model.evaluate(test_loader)
            logit = torch.softmax(logit, dim=1)
            logits.append(logit)
        logits, _ = torch.stack(logits).max(dim=0)
        predictions = torch.argmax(logits, dim=1)
        for i in range(total):
            if predictions[i] == test_data.labels[i]:
                correct += 1
            total_loss += criterion(logits[i], torch.tensor(test_data.labels[i]))
        accuracy = correct / total
        avg_loss = total_loss / total
        return predictions, logits, avg_loss, accuracy
