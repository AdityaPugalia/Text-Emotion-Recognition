import argparse
import itertools
import json
import os
import pickle
import time
from typing import Tuple, Literal, List, Union

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from BERT_CNN import BertCNN
from BERT_GRU import BertGRU
from BERT_LSTM import BertLSTM
from BERT_RNN import BertSimpleRNN
from Simple_BERT import DistilBERTDataset

# set seed for reproducibility
seed = 42
np.random.seed(seed)
torch.manual_seed(seed)

# stores the datasets in memory to avoid reloading every time
datasets = None


def cli() -> argparse.Namespace:
    """
    Command-line interface for running the script.
    """
    parser = argparse.ArgumentParser(
        description="Run a sweep to find optimal hyperparameters."
    )
    parser.add_argument(
        "models",
        type=str,
        nargs="+",
        choices=["CNN", "RNN", "LSTM", "GRU"],
        help="Types of models to train (space-separated).",
    )
    return parser.parse_args()


def load_data(batch_size: int) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Load the datasets for training, validation, and testing.
    Args:
        batch_size (int): The batch size for the DataLoader.
    Returns:
        Tuple[DataLoader, DataLoader, DataLoader]: The train, validation, and test DataLoaders.
    """
    global datasets

    if datasets is None:
        emotion_train = pd.read_csv("data/emotion_train.csv")
        emotion_val = pd.read_csv("data/emotion_val.csv")
        emotion_test = pd.read_csv("data/emotion_test.csv")

        datasets = {
            "train": DistilBERTDataset(
                emotion_train["text"].tolist(), emotion_train["label"].to_list()
            ),
            "val": DistilBERTDataset(
                emotion_val["text"].tolist(), emotion_val["label"].to_list()
            ),
            "test": DistilBERTDataset(
                emotion_test["text"].tolist(), emotion_test["label"].to_list()
            ),
        }

    train_loader = DataLoader(
        datasets["train"],
        batch_size=batch_size,
        shuffle=True,
        num_workers=6,
        pin_memory=True,
    )
    val_loader = DataLoader(
        datasets["val"],
        batch_size=batch_size,
        shuffle=False,
        num_workers=6,
        pin_memory=True,
    )
    test_loader = DataLoader(
        datasets["test"],
        batch_size=batch_size,
        shuffle=False,
        num_workers=6,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader


def get_model(
    model_type: Literal["CNN", "RNN", "LSTM", "GRU"],
    num_labels: int,
    num_layers: int,
    model_path: str = "distilbert-base-uncased",
    **kwargs,
) -> Union[BertCNN, BertSimpleRNN, BertLSTM, BertGRU]:
    """
    Returns a model instance based on the specified type and hyperparameters.

    Args:
        model_type (Literal["CNN", "RNN", "LSTM", "GRU"]): The type of model to create.
        num_labels (int): The number of output labels.
        num_layers (int): The number of layers in the model.
        model_path (str, optional): The pre-trained model path. Defaults to "distilbert-base-uncased".
        **kwargs: Additional arguments specific to the model type.

    Returns:
        torch.nn.Module: The initialized model.
    """
    if model_type == "CNN":
        num_filters = kwargs.get("num_filters")
        n_grams = kwargs.get("n_grams")
        if num_filters is None or n_grams is None:
            raise ValueError("num_filters and n_grams must be provided for CNN.")
        return BertCNN(
            num_labels=num_labels,
            num_layers=num_layers,
            num_filters=num_filters,
            n_grams=n_grams,
            model_path=model_path,
        )
    elif model_type == "RNN":
        hidden_size = kwargs.get("hidden_size")
        if hidden_size is None:
            raise ValueError("hidden_size must be provided for RNN.")
        return BertSimpleRNN(
            num_labels=num_labels,
            hidden_dim=hidden_size,
            num_layers=num_layers,
            model_path=model_path,
        )
    elif model_type == "LSTM":
        hidden_size = kwargs.get("hidden_size")
        if hidden_size is None:
            raise ValueError("hidden_size must be provided for LSTM.")
        return BertLSTM(
            num_labels=num_labels,
            hidden_dim=hidden_size,
            num_layers=num_layers,
            model_path=model_path,
        )
    elif model_type == "GRU":
        hidden_size = kwargs.get("hidden_size")
        if hidden_size is None:
            raise ValueError("hidden_size must be provided for GRU.")
        return BertGRU(
            num_labels=num_labels,
            hidden_dim=hidden_size,
            num_layers=num_layers,
            model_path=model_path,
        )
    else:
        raise ValueError(f"Unsupported model type: {model_type}")


def perform_sweep(
    model_type: Literal["CNN", "RNN", "LSTM", "GRU"],
    num_labels: int,
    num_layers_list: List[int],
    hidden_size_list: List[int] = None,
    num_filters_list: List[int] = None,
    ngram_min_filter_list: List[int] = None,
    batch_size: int = 128,
    num_epochs: int = 100,
    patience: int = 5,
    learning_rate: float = 0.001,
):
    """
    Perform a hyperparameter sweep for the specified model type.

    Args:
        model_type (Literal["CNN", "RNN", "LSTM", "GRU"]): The type of model to create.
        num_labels (int): The number of output labels.
        num_layers_list (List[int]): List of values for the number of layers.
        hidden_size_list (List[int], optional): List of hidden sizes (for RNN, LSTM, GRU).
        num_filters_list (List[int], optional): List of filter counts (for CNN).
        ngram_min_filter_list (List[int], optional): List of min n-gram filter sizes (for CNN). Ngrams are calculated according to [min_size + i for i in range(num_layers)]
        batch_size (int, optional): Batch size for training. Defaults to 128.
        num_epochs (int, optional): Number of epochs for training. Defaults to 100.
        patience (int, optional): Early stopping patience. Defaults to 3.
        learning_rate (float, optional): Learning rate for training. Defaults to 0.001.
    """
    train_loader, val_loader, test_loader = load_data(batch_size)

    # Generate hyperparameter combinations
    if model_type == "CNN":
        if num_filters_list is None or ngram_min_filter_list is None:
            raise ValueError(
                "num_filters_list and ngram_min_filter_list must be provided for CNN."
            )

        original_combinations = itertools.product(num_layers_list, num_filters_list)
        # modify param combinations to include n_grams
        param_combinations = []
        for combination in original_combinations:
            num_layers, num_filters = combination
            for min_size in ngram_min_filter_list:
                ngrams = [min_size + i for i in range(num_layers)]
                param_combinations.append((num_layers, num_filters, ngrams))

    elif model_type in ["RNN", "LSTM", "GRU"]:
        if hidden_size_list is None:
            raise ValueError(
                "hidden_size_list must be provided for RNN, LSTM, and GRU."
            )

        param_combinations = itertools.product(num_layers_list, hidden_size_list)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    best_accuracy = 0

    if not os.path.exists(f"results/{model_type}"):
        os.makedirs(f"results/{model_type}", exist_ok=True)

    if not os.path.exists(f"models/{model_type}"):
        os.makedirs(f"models/{model_type}", exist_ok=True)

    sweep_start_time = time.time()

    for params in param_combinations:
        if model_type == "CNN":
            num_layers, num_filters, n_grams = params

            model = get_model(
                model_type=model_type,
                num_labels=num_labels,
                num_layers=num_layers,
                num_filters=num_filters,
                n_grams=n_grams,
            )
            current_params = {
                "num_labels": num_labels,
                "num_layers": num_layers,
                "num_filters": num_filters,
                "n_grams": n_grams,
            }
        else:
            num_layers, hidden_size = params
            model = get_model(
                model_type=model_type,
                num_labels=num_labels,
                num_layers=num_layers,
                hidden_size=hidden_size,
            )
            current_params = {
                "num_labels": num_labels,
                "num_layers": num_layers,
                "hidden_size": hidden_size,
            }

        print(f"Training {model_type} with params: {current_params}")

        run_start_time = time.time()

        # Train the model
        train_acc, train_loss, val_acc, val_loss = model.train_model(
            train_dataloader=train_loader,
            val_dataloader=val_loader,
            num_epochs=num_epochs,
            learning_rate=learning_rate,
            patience=patience,
            save_model=False,  # restore best model after training instead of saving to file
        )

        print(f"Model trained in {time.time() - run_start_time:.2f} seconds")

        # Evaluate the model on the test set
        predictions, logits, loss, accuracy = model.evaluate(test_loader)

        print(
            f"""
            Model Type: {model_type}
            Params: {params}
            Test Loss: {loss:.4f}
            Test Accuracy: {accuracy:.4f}
            """
        )

        # save model metrics
        current_params_str = "_".join(
            f"{key}={value}" for key, value in current_params.items()
        )
        current_params_str = (
            current_params_str.replace(" ", "")
            .replace("[", "")
            .replace("]", "")
            .replace(",", "_")
        )
        metrics_filename = f"results/{model_type}/{current_params_str}.pkl"

        with open(metrics_filename, "wb") as f:
            pickle.dump(
                (
                    train_acc,
                    train_loss,
                    val_acc,
                    val_loss,
                    predictions,
                    logits,
                    accuracy,
                    loss,
                ),
                f,
            )
        print("Model metrics saved to", metrics_filename)

        # Check if the current model is the best one
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            # save the best model
            torch.save(model.state_dict(), f"models/{model_type}/best_model.pt")
            # save best model params
            with open(f"models/{model_type}/best_model_params.json", "w") as f:
                json.dump(current_params, f)
            print(f"New best model saved to models/{model_type}/best_model.pt !")

    print(f"Sweep completed in {time.time() - sweep_start_time:.2f} seconds")


if __name__ == "__main__":

    # Parse command-line arguments
    args = cli()
    models = args.models
    print(f"Running sweep for model types: {models}")

    # Define parameters for each model type
    sweep_params = {
        "CNN": {
            "num_layers_list": [1, 2, 3],
            "num_filters_list": [32, 64, 128],
            "ngram_min_filter_list": [2, 3],
        },
        "RNN": {
            "num_layers_list": [1, 2, 3],
            "hidden_size_list": [64, 128, 256],
        },
        "LSTM": {
            "num_layers_list": [1, 2, 3],
            "hidden_size_list": [64, 128, 256],
        },
        "GRU": {
            "num_layers_list": [1, 2, 3],
            "hidden_size_list": [64, 128, 256],
        },
    }

    # Common parameters
    num_labels = 6  # Number of output labels
    batch_size = 64
    num_epochs = 50
    patience = 3
    learning_rate = 0.001

    # Run sweep for each model type
    for model_type, params in sweep_params.items():
        if model_type not in models:
            print(f"Skipping sweep for model type: {model_type}")
            continue

        print(f"Running sweep for model type: {model_type}")
        if model_type == "CNN":
            perform_sweep(
                model_type=model_type,
                num_labels=num_labels,
                num_layers_list=params["num_layers_list"],
                num_filters_list=params["num_filters_list"],
                ngram_min_filter_list=params["ngram_min_filter_list"],
                batch_size=batch_size,
                num_epochs=num_epochs,
                patience=patience,
                learning_rate=learning_rate,
            )
        else:
            perform_sweep(
                model_type=model_type,
                num_labels=num_labels,
                num_layers_list=params["num_layers_list"],
                hidden_size_list=params["hidden_size_list"],
                batch_size=batch_size,
                num_epochs=num_epochs,
                patience=patience,
                learning_rate=learning_rate,
            )
