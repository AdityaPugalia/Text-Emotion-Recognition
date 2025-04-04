from transformers import DistilBertModel
from torch.utils.data import Dataset, DataLoader
import torch
import numpy as np
import pandas as pd
from torch.optim import Adam
from tqdm import tqdm
from Simple_BERT import DistilBERTDataset
import pickle

class BertSimpleRNN(torch.nn.Module):
    def __init__(self, num_labels, hidden_dim=64, model_path='distilbert-base-uncased'):
        super(BertSimpleRNN, self).__init__()
        if torch.backends.mps.is_available():
            self.device = torch.device('mps')
        else:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.bert = DistilBertModel.from_pretrained(model_path)
        for param in self.bert.parameters():
            param.requires_grad = False  # Freeze BERT

        self.rnn = torch.nn.RNN(input_size=768, hidden_size=hidden_dim,
                                batch_first=True, nonlinearity='tanh', bidirectional=True)

        self.dropout = torch.nn.Dropout(0.5)
        self.fc = torch.nn.Linear(hidden_dim * 2, num_labels)
        self.to(self.device)

    def forward(self, input_ids, attention_mask):
        bert_output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        embeddings = bert_output.last_hidden_state[:, 1:, :]  # Remove [CLS] token

        mask = attention_mask[:, 1:].unsqueeze(-1).float()
        masked_embeddings = embeddings * mask

        rnn_out, _ = self.rnn(masked_embeddings)  # [batch, seq_len, hidden*2]
        final_hidden = rnn_out[:, -1, :]  # Take last time step
        x = self.dropout(final_hidden)
        x = self.fc(x)
        return x

    def train_RNN(self, train_dataloader, val_dataloader, num_epochs=10, learning_rate=0.001, 
                  patience=3, save_path='models/best_simpleRNN_model.pt'):
        optimizer = Adam(self.parameters(), lr=learning_rate)
        criterion = torch.nn.CrossEntropyLoss()
        best_val_loss = float('inf')
        patience_counter = 0

        train_accuracies, train_losses = [], []
        val_accuracies, val_losses = [], []

        for epoch in range(num_epochs):
            self.train()
            total_loss, correct, total = 0, 0, 0

            for batch in tqdm(train_dataloader):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)

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
            print(f"[Epoch {epoch+1}] Train Loss: {avg_train_loss:.4f}, Accuracy: {train_acc:.4f}")

            # Validation
            self.eval()
            val_loss, correct, total = 0, 0, 0
            with torch.no_grad():
                for batch in val_dataloader:
                    input_ids = batch['input_ids'].to(self.device)
                    attention_mask = batch['attention_mask'].to(self.device)
                    labels = batch['labels'].to(self.device)

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
            print(f"[Epoch {epoch+1}] Val Loss: {avg_val_loss:.4f}, Accuracy: {val_acc:.4f}")

            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                torch.save(self.state_dict(), save_path)
                print("✔️ Saved best model")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print("⏹️ Early stopping triggered")
                    break

        return train_accuracies, train_losses, val_accuracies, val_losses

    def evaluate(self, dataloader):
        self.eval()
        predictions, logits = [], []
        criterion = torch.nn.CrossEntropyLoss()
        total_loss, correct, total = 0, 0, 0

        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)

                outputs = self.forward(input_ids, attention_mask)
                loss = criterion(outputs, labels)
                preds = torch.argmax(outputs, dim=1)

                predictions.extend(preds.cpu().numpy())
                logits.extend(outputs.cpu().numpy())
                total_loss += loss.item() * labels.size(0)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

        avg_loss = total_loss / total
        accuracy = correct / total
        return np.array(predictions), logits, avg_loss, accuracy

if __name__ == "__main__":
    #read the data
    emotion_train = pd.read_csv('data/emotion_train.csv')
    emotion_val = pd.read_csv('data/emotion_val.csv')
    emotion_test = pd.read_csv('data/emotion_test.csv')

    #create Dataset
    emotion_RNN_train = DistilBERTDataset(emotion_train['text'].tolist(), emotion_train['label'].to_list())
    emotion_RNN_val = DistilBERTDataset(emotion_val['text'].tolist(), emotion_val['label'].to_list())
    emotion_RNN_test = DistilBERTDataset(emotion_test['text'].tolist(), emotion_test['label'].to_list())

    # set seed for reproducibility
    np.random.seed(42)
    torch.manual_seed(42)

    # create DataLoader for each dataset
    emotion_RNN_train_data = DataLoader(emotion_RNN_train, batch_size=128, shuffle=True)
    emotion_RNN_val_data = DataLoader(emotion_RNN_val, batch_size=128, shuffle=False)
    emotion_RNN_test_data = DataLoader(emotion_RNN_test, batch_size=128, shuffle=False)

    #training the model
    emotion_RNN_model = BertSimpleRNN(num_labels = 6)
    train_accuracies, train_losses, val_accuracies, val_losses = emotion_RNN_model.train_RNN(train_dataloader= emotion_RNN_train_data, val_dataloader= emotion_RNN_val_data, num_epochs= 100, patience= 3)

    #evaluating the model
    test_predictions, test_logits,test_loss, test_accuracy  = emotion_RNN_model.evaluate(emotion_RNN_test_data)
    print(test_logits[:10], test_predictions[:10], test_accuracy, test_loss)

    # save results as python objects
    with open('results/emotion_RNN_results.pkl', 'wb') as f:
        pickle.dump((train_accuracies, train_losses, val_accuracies, val_losses, test_predictions, test_logits, test_accuracy, test_loss), f)


