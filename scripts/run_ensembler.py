import pickle
import sys
import pandas as pd
import torch
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.net import EnsembleLearner, SimpleEnsembler, BertCNN, BertLSTM, BertGRU
from src.datasets import DistilBERTDataset
from src.utils import get_best_device

if __name__ == "__main__":
    print("Testing Ensembling...")
    # read the data

    project_directory = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    train_filename = os.path.join(project_directory, "data", "emotion_train.csv")
    val_filename = os.path.join(project_directory, "data", "emotion_val.csv")
    test_filename = os.path.join(project_directory, "data", "emotion_test.csv")

    # read the data
    emotion_train = pd.read_csv(train_filename)
    emotion_val = pd.read_csv(val_filename)
    emotion_test = pd.read_csv(test_filename)

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

    best_device = get_best_device()

    # load the models
    print("Loading models...")
    bert_cnn = BertCNN(num_labels=6, num_layers=3, num_filters=128, n_grams=[2, 3, 4])
    bert_cnn.load_state_dict(
        torch.load("models/CNN/best_model.pt", map_location=best_device)
    )
    # bert_rnn = torch.load("models/best_model.pt", map_location= torch.device('cpu'))
    bert_lstm = BertLSTM(num_labels=6, num_layers=3, hidden_dim=256)
    bert_lstm.load_state_dict(
        torch.load("models/LSTM/best_model.pt", map_location=best_device)
    )
    bert_gru = BertGRU(num_labels=6, num_layers=3, hidden_dim=256)
    bert_gru.load_state_dict(
        torch.load("models/GRU/best_model.pt", map_location=best_device)
    )
    print("Models loaded.")

    # Test the ensemble methods
    simple_ensembler = SimpleEnsembler([bert_cnn, bert_lstm, bert_gru])
    print("Testing Mean Ensemble...")
    predictions, logits, avg_loss, accuracy = simple_ensembler.mean_ensemble(
        emotion_test
    )
    print(f"Mean Ensemble Accuracy: {accuracy:.4f}, Loss: {avg_loss:.4f}")
    with open("results/mean_ensemble_results.pkl", "wb") as f:
        pickle.dump((predictions, avg_loss, accuracy), f)

    print("Testing Weighted Ensemble...")
    predictions, logits, avg_loss, accuracy = simple_ensembler.weighted_ensemble(
        emotion_test
    )
    print(f"Weighted Ensemble Accuracy: {accuracy:.4f}, Loss: {avg_loss:.4f}")
    with open("results/weighted_ensemble_results.pkl", "wb") as f:
        pickle.dump((predictions, avg_loss, accuracy), f)

    print("Testing Max Ensemble...")
    predictions, logits, avg_loss, accuracy = simple_ensembler.max_ensemble(
        emotion_test
    )
    print(f"Max Ensemble Accuracy: {accuracy:.4f}, Loss: {avg_loss:.4f}")
    with open("results/max_ensemble_results.pkl", "wb") as f:
        pickle.dump((predictions, avg_loss, accuracy), f)
    print("Simple Ensemble methods tested.")
