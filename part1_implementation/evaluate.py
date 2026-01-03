"""
Evaluation Script for Transformer Sentiment Classifier
Author: Yash
"""

import argparse
import json
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, roc_auc_score, roc_curve
from tqdm import tqdm
from transformers import AutoTokenizer
from datasets import load_dataset

from model import TransformerSentimentClassifier, ModelConfig
from train import SentimentDataset


def load_model(checkpoint_path: str, device: torch.device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = TransformerSentimentClassifier(ModelConfig.from_dict(checkpoint["config"]["model"]))
    model.load_state_dict(checkpoint["model_state_dict"])
    return model.to(device).eval()


def evaluate_model(model, dataloader, device):
    all_preds, all_labels, all_probs = [], [], []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            outputs = model(batch["input_ids"].to(device), batch["attention_mask"].to(device))
            probs = torch.softmax(outputs["logits"], dim=-1)
            all_preds.extend(probs.argmax(dim=-1).cpu().numpy())
            all_labels.extend(batch["labels"].numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())

    preds, labels, probs = np.array(all_preds), np.array(all_labels), np.array(all_probs)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average="binary")

    return {
        "accuracy": accuracy_score(labels, preds),
        "precision": precision, "recall": recall, "f1_score": f1,
        "auc_roc": roc_auc_score(labels, probs),
        "predictions": preds, "labels": labels, "probabilities": probs
    }


def plot_results(results, output_dir):
    # Confusion Matrix
    cm = confusion_matrix(results["labels"], results["predictions"])
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Neg", "Pos"], yticklabels=["Neg", "Pos"])
    plt.title("Confusion Matrix")
    plt.savefig(output_dir / "confusion_matrix.png", dpi=150)
    plt.close()

    # ROC Curve
    fpr, tpr, _ = roc_curve(results["labels"], results["probabilities"])
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'AUC = {results["auc_roc"]:.3f}')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel("FPR");
    plt.ylabel("TPR");
    plt.title("ROC Curve");
    plt.legend()
    plt.savefig(output_dir / "roc_curve.png", dpi=150)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output_dir", default="evaluation_results")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model = load_model(args.checkpoint, device)
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    dataset = load_dataset("imdb")
    test_loader = DataLoader(SentimentDataset(dataset["test"]["text"], dataset["test"]["label"], tokenizer),
                             batch_size=32)

    results = evaluate_model(model, test_loader, device)

    print(f"Accuracy: {results['accuracy'] * 100:.2f}%")
    print(f"F1 Score: {results['f1_score']:.4f}")
    print(f"AUC-ROC: {results['auc_roc']:.4f}")

    plot_results(results, output_dir)
    with open(output_dir / "metrics.json", "w") as f:
        json.dump({k: float(v) for k, v in results.items() if k not in ['predictions', 'labels', 'probabilities']}, f,
                  indent=2)


if __name__ == "__main__":
    main()
