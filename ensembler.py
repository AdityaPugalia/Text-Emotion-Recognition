from utils import get_best_device
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
from Simple_BERT import DistilBERTDataset
import numpy as np
import pandas as pd
import pickle

class EnsembleDataset(Dataset):
    def __init__ (self, logits : torch.Tensor, labels : torch.Tensor):
        self.logits = logits
        self.labels = labels
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return self.logits[:, idx, :], self.labels[idx] # logits are [num_models, batch_size, num_labels], labels are [batch_size]

class EnsembleLearner(torch.nn.Module):
    def __init__(self, num_labels, model_list):
        super(EnsembleLearner, self).__init__()
        self.device = get_best_device()
        self.models = [model.to(self.device) for model in model_list]
        self.fc = torch.nn.Linear(len(model_list) * num_labels, num_labels)
        self.to(self.device)

    def forward(self, logits):
        logits = logits.view(logits.size(0), -1)  # [batch_size, num_models * num_labels]
        logits = self.fc(logits)
        return logits
    
    def train_ensemble(self, train_data, val_data, model_path, num_epochs=100, learning_rate=0.001, patience = 3):
        # create emotion dataset
        train_loader = DataLoader(train_data, batch_size=128, shuffle = False)
        val_loader = DataLoader(val_data, batch_size=128, shuffle = False)
        
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
        train_loader = DataLoader(train_dataset, batch_size=128, shuffle = True)
        val_loader = DataLoader(val_dataset, batch_size=128, shuffle = False) 

        # initialize the optimizer and loss function
        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        criterion = torch.nn.CrossEntropyLoss()
        patience_counter = 0     
        max_loss = float('inf')  

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
            print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}")
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
            print(f"Validation Loss: {avg_val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}")
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
        self.models = [model.to(self.device) for model in model_list]
        self.ensemble_learner = None

    def mean_ensemble(self, test_data : DistilBERTDataset):
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
    
    def weighted_ensemble(self, test_data : DistilBERTDataset):
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
        logits = torch.sum(logits * weights.unsqueeze(1), dim=0)
        predictions = torch.argmax(logits, dim=1)
        for i in range(total):
            if predictions[i] == test_data.labels[i]:
                correct += 1
            total_loss += criterion(logits[i], torch.tensor(test_data.labels[i]))
        accuracy = correct / total
        avg_loss = total_loss / total
        return predictions, logits, avg_loss, accuracy
    
    def max_ensemble(self, test_data : DistilBERTDataset):
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

if __name__ == "__main__":
    print("Testing Ensembling...")
    # read the data
    emotion_train = pd.read_csv("data/emotion_train.csv")
    emotion_val = pd.read_csv("data/emotion_val.csv")
    emotion_test = pd.read_csv("data/emotion_test.csv")

    # create Dataset
    emotion_train = DistilBERTDataset(
        emotion_train["text"].tolist(), emotion_train["label"].to_list()
    )
    emotion_val = DistilBERTDataset(
        emotion_val["text"].tolist(), emotion_val["label"].to_list()
    )
    emotion_test = DistilBERTDataset(
        emotion_test["text"].tolist(), emotion_test["label"].to_list()
    )

    # load the models
    print("Loading models...")
    bert_cnn = torch.load('models/best_CNN_model.pt')
    bert_rnn = torch.load("models/best_simpleRNN_model.pt")
    bert_lstm = torch.load("models/best_LSTM_model.pt")
    bert_gru = torch.load("models/best_GRU_model.pt")
    print("Models loaded.")

    # Test the ensemble methods
    simple_ensembler = SimpleEnsembler([bert_cnn, bert_rnn, bert_lstm, bert_gru])
    print("Testing Mean Ensemble...")
    predictions, logits, avg_loss, accuracy = simple_ensembler.mean_ensemble(emotion_test)
    print(f"Mean Ensemble Accuracy: {accuracy:.4f}, Loss: {avg_loss:.4f}")
    with open("mean_ensemble_results.pkl", "wb") as f:
        pickle.dump((predictions, avg_loss, accuracy), f)

    print("Testing Weighted Ensemble...")
    predictions, logits, avg_loss, accuracy = simple_ensembler.weighted_ensemble(emotion_test)
    print(f"Weighted Ensemble Accuracy: {accuracy:.4f}, Loss: {avg_loss:.4f}")
    with open("weighted_ensemble_results.pkl", "wb") as f:
        pickle.dump((predictions, avg_loss, accuracy), f)

    print("Testing Max Ensemble...")
    predictions, logits, avg_loss, accuracy = simple_ensembler.max_ensemble(emotion_test)
    print(f"Max Ensemble Accuracy: {accuracy:.4f}, Loss: {avg_loss:.4f}")
    with open("max_ensemble_results.pkl", "wb") as f:
        pickle.dump((predictions, avg_loss, accuracy), f)
    print("Simple Ensemble methods tested.")

    # Train the ensemble learner
    print("Training Ensemble Learner...")
    ensemble_learner = EnsembleLearner(6, [bert_cnn, bert_rnn, bert_lstm, bert_gru])
    train_accuracies, train_losses, val_accuracies, val_losses = ensemble_learner.train_ensemble(
        emotion_train, emotion_val, "models/best_ensemble_model.pt", num_epochs=100
    )
    print("Ensemble Learner trained.")
    with open("ensemble_learner_training_results.pkl", "wb") as f:
        pickle.dump((train_accuracies, train_losses, val_accuracies, val_losses), f)
    print("Ensemble Learner training results saved.")

    # Evaluate the ensemble learner
    print("Evaluating Ensemble Learner...")
    predictions, avg_loss, accuracy = ensemble_learner.evaluate(emotion_test)
    print(f"Ensemble Learner Accuracy: {accuracy:.4f}, Loss: {avg_loss:.4f}")
    with open("ensemble_learner_results.pkl", "wb") as f:
        pickle.dump((predictions, avg_loss, accuracy), f)
    print("All tests completed.")

    

    

     


            
        

