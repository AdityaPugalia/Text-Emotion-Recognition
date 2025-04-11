import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import DistilBertTokenizer


class DistilBERTDataset(Dataset):
    def __init__(
        self, texts, labels, tokenizer_name="distilbert-base-uncased", max_len=64
    ):
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
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt",
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(self.labels[idx]),
        }


class MemoryMappedDataset(Dataset):
    def __init__(self, emb_path, label_path):
        self.embeddings = np.load(emb_path, mmap_mode="r")
        self.labels = np.load(label_path, mmap_mode="r")

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        emb = torch.from_numpy(self.embeddings[idx]).float()
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return emb, label


class EnsembleDataset(Dataset):
    def __init__(self, logits: torch.Tensor, labels: torch.Tensor):
        self.logits = logits
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return (
            self.logits[:, idx, :],
            self.labels[idx],
        )  # logits are [num_models, batch_size, num_labels], labels are [batch_size]
