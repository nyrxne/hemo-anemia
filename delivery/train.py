"""
H.E.M.A. — Baseline training script.

Usage (Colab or any machine with internet access for pretrained weights):
    python train.py --data_root /path/to/clean_output_split --epochs_head 5 --epochs_finetune 10

Trains on train/, selects the best checkpoint by validation AUROC on val/,
then reports final metrics on BOTH val/ (in-domain) and test_unseen_site/
(the core H.E.M.A. generalization comparison) — never used for model
selection, only for final reporting.
"""
import argparse
import json
import os

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, confusion_matrix, brier_score_loss

from dataset import HemaConjunctivaDataset, set_seed
from model import build_model, set_backbone_trainable


def compute_metrics(y_true, y_prob, threshold=0.5):
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else float("nan")  # recall on Anemic (class 1)
    specificity = tn / (tn + fp) if (tn + fp) > 0 else float("nan")  # recall on Non-anemic (class 0)
    try:
        auroc = roc_auc_score(y_true, y_prob)
    except ValueError:
        auroc = float("nan")
    brier = brier_score_loss(y_true, y_prob)  # calibration: lower is better
    return {
        "n": len(y_true),
        "accuracy": accuracy,
        "sensitivity_anemic_recall": sensitivity,
        "specificity_nonanemic_recall": specificity,
        "auroc": auroc,
        "brier_score": brier,
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    all_probs, all_labels = [], []
    for imgs, labels, _ in loader:
        imgs = imgs.to(device)
        logits = model(imgs)
        probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()  # P(Anemic)
        all_probs.extend(probs.tolist())
        all_labels.extend(labels.numpy().tolist())
    return np.array(all_labels), np.array(all_probs)


def train_one_phase(model, train_loader, val_loader, device, epochs, lr, class_weights, log_prefix=""):
    optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=lr)
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    best_auroc = -1
    best_state = None
    history = []

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for imgs, labels, _ in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = model(imgs)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * imgs.size(0)

        train_loss = running_loss / len(train_loader.dataset)
        y_true, y_prob = evaluate(model, val_loader, device)
        val_metrics = compute_metrics(y_true, y_prob)
        history.append({"epoch": epoch + 1, "train_loss": train_loss, **val_metrics})
        print(f"{log_prefix} epoch {epoch+1}/{epochs}  train_loss={train_loss:.4f}  "
              f"val_acc={val_metrics['accuracy']:.3f}  val_auroc={val_metrics['auroc']:.3f}")

        if val_metrics["auroc"] > best_auroc:
            best_auroc = val_metrics["auroc"]
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, history


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, required=True,
                         help="Folder containing train/, val/, test_unseen_site/")
    parser.add_argument("--architecture", type=str, default="efficientnet_b0",
                         choices=["efficientnet_b0", "mobilenet_v2"])
    parser.add_argument("--epochs_head", type=int, default=5)
    parser.add_argument("--epochs_finetune", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr_head", type=float, default=1e-3)
    parser.add_argument("--lr_finetune", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--pretrained", action="store_true", default=True)
    parser.add_argument("--no_pretrained", dest="pretrained", action="store_false",
                         help="Use random init instead of ImageNet weights (for smoke-testing "
                              "in network-restricted environments only — NOT the real baseline).")
    parser.add_argument("--out_dir", type=str, default="./results")
    args = parser.parse_args()

    set_seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device, "| pretrained weights:", args.pretrained)

    train_ds = HemaConjunctivaDataset(os.path.join(args.data_root, "train"), augment=True)
    val_ds = HemaConjunctivaDataset(os.path.join(args.data_root, "val"), augment=False)
    test_ds = HemaConjunctivaDataset(os.path.join(args.data_root, "test_unseen_site"), augment=False)

    print("Train class counts:", train_ds.class_counts())
    print("Val class counts:", val_ds.class_counts())
    print("Unseen-site test class counts:", test_ds.class_counts())

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=2)

    # Inverse-frequency class weights from the TRAIN set only
    counts = train_ds.class_counts()
    total = counts[0] + counts[1]
    class_weights = torch.tensor([total / (2 * counts[0]), total / (2 * counts[1])], dtype=torch.float32)
    print("Class weights (Non-anemic, Anemic):", class_weights.tolist())

    model, backbone_params = build_model(args.architecture, pretrained=args.pretrained)
    model = model.to(device)
    backbone_params = list(model.features.parameters())

    # Phase 1: head-only
    set_backbone_trainable(backbone_params, trainable=False)
    model, hist1 = train_one_phase(model, train_loader, val_loader, device,
                                    args.epochs_head, args.lr_head, class_weights, log_prefix="[head]")

    # Phase 2: fine-tune whole network
    set_backbone_trainable(backbone_params, trainable=True)
    model, hist2 = train_one_phase(model, train_loader, val_loader, device,
                                    args.epochs_finetune, args.lr_finetune, class_weights, log_prefix="[finetune]")

    # Final reporting: in-domain val vs. unseen-site test — the core H.E.M.A. comparison
    y_true_val, y_prob_val = evaluate(model, val_loader, device)
    val_final = compute_metrics(y_true_val, y_prob_val)

    y_true_test, y_prob_test = evaluate(model, test_loader, device)
    test_final = compute_metrics(y_true_test, y_prob_test)

    results = {
        "architecture": args.architecture,
        "pretrained": args.pretrained,
        "seed": args.seed,
        "in_domain_validation": val_final,
        "unseen_site_test": test_final,
        "training_history_head_phase": hist1,
        "training_history_finetune_phase": hist2,
    }

    with open(os.path.join(args.out_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2)

    torch.save(model.state_dict(), os.path.join(args.out_dir, "model_weights.pt"))

    print("\n=== FINAL COMPARISON ===")
    print("In-domain validation:", json.dumps(val_final, indent=2))
    print("Unseen-site test:    ", json.dumps(test_final, indent=2))
    print(f"\nSaved results to {args.out_dir}/results.json and model_weights.pt")


if __name__ == "__main__":
    main()
