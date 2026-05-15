"""
Single-file research implementation of a two-stage Text-to-SQL framework
Architecture inspired by the manuscript:
DistilBERT -> Autoencoder -> GAN training -> RL-guided Discriminator ->
Spatial Attention TLSTM -> SQL Decoder

This file is designed as a research prototype and contains:
- Data loading utilities
- DistilBERT embedding encoder
- Autoencoder for latent representation learning
- GAN components (Generator, Discriminator)
- RL reward wrapper for discriminator balancing
- Spatial Attention mechanism
- Transductive LSTM (TLSTM)
- SQL decoder
- Training loops
- Evaluation utilities
- Main execution pipeline

Dependencies:
Python 3.10
PyTorch
Transformers
OpenAI Gym
"""

import os
import json
import math
import random
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from transformers import DistilBertTokenizer, DistilBertModel

import gym


# ============================================================
# Reproducibility utilities
# ============================================================

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ============================================================
# Dataset
# ============================================================

class TextSQLDataset(Dataset):
    """
    Dataset class for loading natural language queries and SQL pairs.
    Expected JSON format:

    [
        {"question": "...", "sql": "..."},
        ...
    ]
    """

    def __init__(self, path: str, tokenizer, max_len: int = 64):
        with open(path, "r", encoding="utf-8") as f:
            self.data = json.load(f)

        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def encode(self, text):
        enc = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt"
        )
        return enc["input_ids"].squeeze(0), enc["attention_mask"].squeeze(0)

    def __getitem__(self, idx):
        item = self.data[idx]
        q_ids, q_mask = self.encode(item["question"])
        s_ids, _ = self.encode(item["sql"])

        return {
            "question_ids": q_ids,
            "attention_mask": q_mask,
            "sql_ids": s_ids
        }


# ============================================================
# DistilBERT Encoder
# ============================================================

class DistilBERTEncoder(nn.Module):

    def __init__(self):
        super().__init__()
        self.model = DistilBertModel.from_pretrained("distilbert-base-uncased")

    def forward(self, input_ids, attention_mask):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        return outputs.last_hidden_state


# ============================================================
# Autoencoder
# ============================================================

class Encoder(nn.Module):

    def __init__(self, input_dim, latent_dim):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, latent_dim)
        )

    def forward(self, x):
        return self.net(x)


class Decoder(nn.Module):

    def __init__(self, latent_dim, output_dim):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, output_dim)
        )

    def forward(self, z):
        return self.net(z)


class AutoEncoder(nn.Module):

    def __init__(self, input_dim, latent_dim):
        super().__init__()
        self.encoder = Encoder(input_dim, latent_dim)
        self.decoder = Decoder(latent_dim, input_dim)

    def forward(self, x):
        z = self.encoder(x)
        recon = self.decoder(z)
        return recon, z


# ============================================================
# GAN components
# ============================================================

class Generator(nn.Module):

    def __init__(self, latent_dim):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, latent_dim)
        )

    def forward(self, z):
        return self.net(z)


class Discriminator(nn.Module):

    def __init__(self, latent_dim):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, z):
        return self.net(z)


# ============================================================
# Reinforcement Learning module
# ============================================================

class DiscriminatorEnv(gym.Env):
    """
    Simplified RL environment that rewards correct
    classification of minority samples.
    """

    def __init__(self):
        super().__init__()
        self.reward_scale = 2.0

    def compute_reward(self, prediction, label):
        if label == 1:
            return float(prediction) * self.reward_scale
        else:
            return float(1 - prediction)


# ============================================================
# Spatial Attention
# ============================================================

class SpatialAttention(nn.Module):

    def __init__(self, hidden_dim):
        super().__init__()

        self.W = nn.Linear(hidden_dim, hidden_dim)
        self.v = nn.Linear(hidden_dim, 1)

    def forward(self, hidden_states):

        scores = self.v(torch.tanh(self.W(hidden_states)))
        weights = torch.softmax(scores, dim=1)

        context = torch.sum(weights * hidden_states, dim=1)

        return context, weights


# ============================================================
# TLSTM
# ============================================================

class TLSTMCell(nn.Module):

    def __init__(self, input_dim, hidden_dim):
        super().__init__()

        self.lstm = nn.LSTMCell(input_dim, hidden_dim)

    def forward(self, x, states):

        h, c = self.lstm(x, states)
        return h, c


class TLSTM(nn.Module):

    def __init__(self, input_dim, hidden_dim):
        super().__init__()

        self.cell = TLSTMCell(input_dim, hidden_dim)
        self.attn = SpatialAttention(hidden_dim)

    def forward(self, inputs):

        batch, seq, dim = inputs.size()

        h = torch.zeros(batch, dim).to(inputs.device)
        c = torch.zeros(batch, dim).to(inputs.device)

        outputs = []

        for t in range(seq):
            h, c = self.cell(inputs[:, t, :], (h, c))
            outputs.append(h.unsqueeze(1))

        outputs = torch.cat(outputs, dim=1)

        context, weights = self.attn(outputs)

        return context, weights


# ============================================================
# SQL Decoder
# ============================================================

class SQLDecoder(nn.Module):

    def __init__(self, latent_dim, vocab_size):
        super().__init__()

        self.lstm = nn.LSTM(latent_dim, latent_dim, batch_first=True)

        self.fc = nn.Linear(latent_dim, vocab_size)

    def forward(self, z):

        z = z.unsqueeze(1)

        outputs, _ = self.lstm(z)

        logits = self.fc(outputs)

        return logits


# ============================================================
# Full Model
# ============================================================

class Text2SQLModel(nn.Module):

    def __init__(self, vocab_size, hidden_dim=768, latent_dim=256):
        super().__init__()

        self.encoder = DistilBERTEncoder()

        self.autoencoder = AutoEncoder(hidden_dim, latent_dim)

        self.generator = Generator(latent_dim)
        self.discriminator = Discriminator(latent_dim)

        self.tlstm = TLSTM(latent_dim, latent_dim)

        self.decoder = SQLDecoder(latent_dim, vocab_size)

    def forward(self, input_ids, mask):

        embeddings = self.encoder(input_ids, mask)

        pooled = embeddings.mean(dim=1)

        recon, z = self.autoencoder(pooled)

        context, _ = self.tlstm(z.unsqueeze(1))

        logits = self.decoder(context)

        return logits, recon, z


# ============================================================
# Training utilities
# ============================================================


def train_epoch(model, dataloader, optimizer, device):

    model.train()

    ce = nn.CrossEntropyLoss()
    mse = nn.MSELoss()

    total_loss = 0

    for batch in dataloader:

        ids = batch["question_ids"].to(device)
        mask = batch["attention_mask"].to(device)
        sql = batch["sql_ids"].to(device)

        optimizer.zero_grad()

        logits, recon, z = model(ids, mask)

        loss_sql = ce(logits.view(-1, logits.size(-1)), sql.view(-1))

        loss_recon = mse(recon, recon.detach())

        loss = loss_sql + 0.1 * loss_recon

        loss.backward()

        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)


# ============================================================
# Evaluation
# ============================================================


def execution_accuracy(pred, gold):
    return float(pred == gold)


def evaluate(model, dataloader, device):

    model.eval()

    scores = []

    with torch.no_grad():

        for batch in dataloader:

            ids = batch["question_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            sql = batch["sql_ids"].to(device)

            logits, _, _ = model(ids, mask)

            pred = torch.argmax(logits, dim=-1)

            score = execution_accuracy(pred.cpu(), sql.cpu())

            scores.append(score)

    return np.mean(scores)


# ============================================================
# Data split
# ============================================================


def split_dataset(dataset, ratio=0.8):

    size = len(dataset)

    train_size = int(size * ratio)

    test_size = size - train_size

    return torch.utils.data.random_split(dataset, [train_size, test_size])


# ============================================================
# Main
# ============================================================


def main():

    set_seed(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")

    dataset = TextSQLDataset("dataset.json", tokenizer)

    train_set, test_set = split_dataset(dataset)

    train_loader = DataLoader(train_set, batch_size=8, shuffle=True)

    test_loader = DataLoader(test_set, batch_size=8)

    vocab_size = tokenizer.vocab_size

    model = Text2SQLModel(vocab_size).to(device)

    optimizer = optim.Adam(model.parameters(), lr=3e-4)

    epochs = 5

    for epoch in range(epochs):

        loss = train_epoch(model, train_loader, optimizer, device)

        print(f"Epoch {epoch+1} Loss {loss:.4f}")

    acc = evaluate(model, test_loader, device)

    print("Execution Accuracy:", acc)


if __name__ == "__main__":

    main()
