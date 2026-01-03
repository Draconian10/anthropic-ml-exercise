"""
Training Script for Transformer Sentiment Classifier
Author: Yash
"""

import os
import argparse
import random
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Tuple

import yaml
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.cuda.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from transformers import AutoTokenizer
from datasets import load_dataset

from model import TransformerSentimentClassifier, ModelConfig


class EarlyStopping:
    def __init__(self, patience: int = 5, min_delta: float = 0.0, mode: str = "min"):
        self.patience, self.min_delta, self.mode = patience, min_delta, mode
        self.counter, self.best_score, self.should_stop = 0, None, False

    def __call__(self, score: float) -> bool:
        if self.best_score is None:
            self.best_score = score
            return False
        improved = score < self.best_score - self.min_delta if self.mode == "min" else score > self.best_score + self.min_delta
        if improved:
            self.best_score, self.counter = score, 0
        else:
            self.counter += 1
            self.should_stop = self.counter >= self.patience
        return self.should_stop


class SentimentDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=256):
        self.texts, self.labels, self.tokenizer, self.max_length = texts, labels, tokenizer, max_length

    def __len__(self): return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(self.texts[idx], truncation=True, padding="max_length",
                                  max_length=self.max_length, return_tensors="pt")
        return {"input_ids": encoding["input_ids"].squeeze(0),
                "attention_mask": encoding["attention_mask"].squeeze(0),
                "labels": torch.tensor(self.labels[idx], dtype=torch.long)}


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_data(config: Dict):
    print("Loading dataset...")
    dataset = load_dataset("imdb")
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    train_data = dataset["train"].shuffle(seed=config["experiment"]["seed"])
    train_size = int(0.9 * len(train_data))

    train_dataset = SentimentDataset(train_data["text"][:train_size], train_data["label"][:train_size],
                                     tokenizer, config["data"]["max_length"])
    val_dataset = SentimentDataset(train_data["text"][train_size:], train_data["label"][train_size:],
                                   tokenizer, config["data"]["max_length"])
    test_dataset = SentimentDataset(dataset["test"]["text"], dataset["test"]["label"],
                                    tokenizer, config["data"]["max_length"])

    batch_size = config["data"]["batch_size"]
    return (DataLoader(train_dataset, batch_size=batch_size, shuffle=True),
            DataLoader(val_dataset, batch_size=batch_size),
            DataLoader(test_dataset, batch_size=batch_size))


def train_epoch(model, loader, optimizer, scheduler, criterion, device, scaler, config, writer, global_step):
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for batch in tqdm(loader, desc="Training"):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        optimizer.zero_grad()

        if scaler:
            with autocast():
                outputs = model(input_ids, attention_mask)
                loss = criterion(outputs["logits"], labels)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config["training"]["max_grad_norm"])
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(input_ids, attention_mask)
            loss = criterion(outputs["logits"], labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config["training"]["max_grad_norm"])
            optimizer.step()

        if scheduler: scheduler.step()

        total_loss += loss.item()
        correct += (outputs["logits"].argmax(dim=-1) == labels).sum().item()
        total += labels.size(0)
        global_step += 1

        if global_step % config["logging"]["log_every_n_steps"] == 0:
            writer.add_scalar("train/loss", loss.item(), global_step)

    return total_loss / len(loader), correct / total, global_step


@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0

    for batch in tqdm(loader, desc="Validating"):
        outputs = model(batch["input_ids"].to(device), batch["attention_mask"].to(device))
        labels = batch["labels"].to(device)
        total_loss += criterion(outputs["logits"], labels).item()
        correct += (outputs["logits"].argmax(dim=-1) == labels).sum().item()
        total += labels.size(0)

    return total_loss / len(loader), correct / total


def train(config_path: str):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    set_seed(config["experiment"]["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_loader, val_loader, test_loader = load_data(config)

    model = TransformerSentimentClassifier(ModelConfig.from_dict(config["model"])).to(device)
    print(f"Model parameters: {model.count_parameters():,}")

    criterion = nn.CrossEntropyLoss(label_smoothing=config["regularization"]["label_smoothing"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["training"]["learning_rate"],
                                  weight_decay=config["training"]["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, len(train_loader) * config["training"]["epochs"])
    scaler = GradScaler() if device.type == "cuda" else None

    writer = SummaryWriter(
        Path(config["logging"]["log_dir"]) / f"{config['experiment']['name']}_{datetime.now():%Y%m%d_%H%M%S}")
    checkpoint_dir = Path(config["training"]["checkpoint"]["save_dir"])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    early_stopping = EarlyStopping(config["training"]["early_stopping"]["patience"])
    best_val_loss, global_step = float("inf"), 0

    for epoch in range(config["training"]["epochs"]):
        train_loss, train_acc, global_step = train_epoch(model, train_loader, optimizer, scheduler,
                                                         criterion, device, scaler, config, writer, global_step)
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        print(
            f"Epoch {epoch + 1}: Train Loss={train_loss:.4f}, Acc={train_acc * 100:.1f}% | Val Loss={val_loss:.4f}, Acc={val_acc * 100:.1f}%")
        writer.add_scalars("epoch", {"train_loss": train_loss, "val_loss": val_loss}, epoch)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({"model_state_dict": model.state_dict(), "config": config}, checkpoint_dir / "best_model.pt")

        if early_stopping(val_loss):
            print(f"Early stopping at epoch {epoch + 1}")
            break

    model.load_state_dict(torch.load(checkpoint_dir / "best_model.pt")["model_state_dict"])
    test_loss, test_acc = validate(model, test_loader, criterion, device)
    print(f"Test: Loss={test_loss:.4f}, Acc={test_acc * 100:.1f}%")
    writer.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    train(parser.parse_args().config)
