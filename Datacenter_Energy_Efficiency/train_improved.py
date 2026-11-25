"""
Projet : Efficacité Énergétique des Centres de Données (Datacenters)
Fichier : train_improved.py
Description : Entraînement, étude d'ablation et sélection du CNN optimisé
              EXCLUSIVEMENT sur les données de validation (zéro fuite du Test Set).

Étudie systématiquement :
  - Batch Normalization (avec vs sans)
  - Dropout spatial & dense (avec vs sans, niveaux modéré vs fort)
  - Weight Decay AdamW (0, 1e-3, 2e-3, 3e-3)
  - Augmentation thermique (avec vs sans)
  - Dual Global Pooling GAP+GMP vs GAP seul
  - Ordonnanceur LR (CosineAnnealingLR vs ReduceLROnPlateau) + Early Stopping
"""

import os
import time
import json
import random
import tempfile
import numpy as np
import pandas as pd

os.environ.setdefault("MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "matplotlib_cache"))

import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from cnn_model_improved import OptimizedThermalCNN
from preprocessing_improved import (
    load_and_preprocess_improved_data,
    load_improved_splits,
    ImprovedThermalDataset,
    CLASS_NAMES,
)
from torch.utils.data import DataLoader


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, targets in dataloader:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == targets).sum().item()
        total += targets.size(0)

    return running_loss / total, correct / total


def evaluate_split(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            running_loss += loss.item() * inputs.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == targets).sum().item()
            total += targets.size(0)

    return running_loss / total, correct / total


def train_single_config(config: dict, data_path: str = None, seed: int = 42, verbose: bool = True):
    """
    Entraîne une configuration candidate et retourne les métriques Train & Validation
    du meilleur checkpoint (sans jamais accéder au jeu de test).
    """
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    batch_size = config.get("batch_size", 32)
    epochs = config.get("epochs", 35)
    patience = config.get("patience", 8)
    lr = config.get("lr", 0.002)
    weight_decay = config.get("weight_decay", 1e-3)
    augment = config.get("augment", True)
    scheduler_type = config.get("scheduler", "cosine")

    train_loader, val_loader, _, norm_stats, _ = load_and_preprocess_improved_data(
        data_path=data_path,
        batch_size=batch_size,
        augment=augment,
        aug_shift=config.get("aug_shift", True),
        aug_noise_std=config.get("aug_noise_std", 0.06),
        seed=seed,
    )

    # Loader déterministe sans augmentation pour mesurer l'exacte Train Loss / Train Acc en mode eval()
    (X_train, y_train), _, _, _, _ = load_improved_splits(data_path)
    train_eval_loader = DataLoader(
        ImprovedThermalDataset(X_train, y_train, norm_stats["mean"], norm_stats["std"], augment=False),
        batch_size=batch_size,
        shuffle=False,
    )

    model_cfg = {
        "in_channels": 2,
        "num_classes": 4,
        "use_bn": config.get("use_bn", True),
        "dropout_conv1": config.get("dropout_conv1", 0.10),
        "dropout_conv2": config.get("dropout_conv2", 0.15),
        "dropout_fc1": config.get("dropout_fc1", 0.35),
        "dropout_fc2": config.get("dropout_fc2", 0.20),
        "pooling_mode": config.get("pooling_mode", "dual"),
    }
    model = OptimizedThermalCNN(**model_cfg).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    if scheduler_type == "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=3, min_lr=1e-5
        )
    else:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs, eta_min=1e-5
        )

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [], "lr": []}
    best_val_loss = float("inf")
    best_val_acc = 0.0
    best_epoch = 0
    best_state = None
    epochs_no_improve = 0

    t0 = time.time()
    for epoch in range(1, epochs + 1):
        tr_loss, tr_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        va_loss, va_acc = evaluate_split(model, val_loader, criterion, device)

        current_lr = optimizer.param_groups[0]["lr"]
        if scheduler_type == "plateau":
            scheduler.step(va_loss)
        else:
            scheduler.step()

        history["train_loss"].append(float(tr_loss))
        history["train_acc"].append(float(tr_acc))
        history["val_loss"].append(float(va_loss))
        history["val_acc"].append(float(va_acc))
        history["lr"].append(float(current_lr))

        if va_loss < best_val_loss - 1e-4 or (abs(va_loss - best_val_loss) <= 1e-4 and va_acc > best_val_acc):
            best_val_loss = va_loss
            best_val_acc = va_acc
            best_epoch = epoch
            epochs_no_improve = 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            epochs_no_improve += 1

        if verbose and (epoch % 5 == 0 or epoch == 1):
            print(
                f"  [{config['name']}] Ep {epoch:02d}/{epochs:02d} | "
                f"Train L={tr_loss:.4f} A={tr_acc*100:.2f}% | "
                f"Val L={va_loss:.4f} A={va_acc*100:.2f}% | lr={current_lr:.5f}"
            )

        if epochs_no_improve >= patience:
            if verbose:
                print(f"  [{config['name']}] Early Stopping à l'époque {epoch} (meilleure={best_epoch}).")
            break

    elapsed = time.time() - t0
    model.load_state_dict(best_state)
    train_eval_loss, train_eval_acc = evaluate_split(model, train_eval_loader, criterion, device)
    val_eval_loss, val_eval_acc = evaluate_split(model, val_loader, criterion, device)

    return {
        "name": config["name"],
        "description": config.get("description", ""),
        "model_config": model_cfg,
        "hyperparameters": {
            "lr": lr,
            "weight_decay": weight_decay,
            "batch_size": batch_size,
            "epochs_max": epochs,
            "patience": patience,
            "scheduler": scheduler_type,
            "augment": augment,
            "optimizer": "AdamW",
        },
        "num_parameters": int(model.count_parameters()),
        "best_epoch": int(best_epoch),
        "epochs_ran": len(history["train_loss"]),
        "train_eval_loss": float(train_eval_loss),
        "train_eval_acc": float(train_eval_acc),
        "best_val_loss": float(val_eval_loss),
        "best_val_acc": float(val_eval_acc),
        "generalization_gap_acc": float(train_eval_acc - val_eval_acc),
        "elapsed_sec": float(elapsed),
        "history": history,
        "best_state": best_state,
        "norm_stats": norm_stats,
    }


def plot_training_curves(history: dict, save_path: str, title: str = "Courbes d'Apprentissage — CNN Optimisé"):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    epochs = range(1, len(history["train_loss"]) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.5, 5.0))
    ax1.plot(epochs, history["train_loss"], "o-", color="#1f77b4", label="Train Loss", lw=2, markersize=4)
    ax1.plot(epochs, history["val_loss"], "s-", color="#ff7f0e", label="Validation Loss", lw=2, markersize=4)
    ax1.set_title("Évolution de la Perte (Cross-Entropy)", fontsize=11.5, fontweight="bold")
    ax1.set_xlabel("Époque")
    ax1.set_ylabel("Perte")
    ax1.legend(frameon=True)
    ax1.grid(True, alpha=0.3, linestyle="--")

    ax2.plot(epochs, [a * 100 for a in history["train_acc"]], "o-", color="#1f77b4", label="Train Accuracy", lw=2, markersize=4)
    ax2.plot(epochs, [a * 100 for a in history["val_acc"]], "s-", color="#ff7f0e", label="Validation Accuracy", lw=2, markersize=4)
    ax2.axhspan(90.0, 93.0, color="green", alpha=0.08, label="Zone Cible (~90–93%)")
    ax2.set_title("Évolution de l'Exactitude (%)", fontsize=11.5, fontweight="bold")
    ax2.set_xlabel("Époque")
    ax2.set_ylabel("Exactitude (%)")
    ax2.set_ylim(45, 102)
    ax2.legend(loc="lower right", frameon=True)
    ax2.grid(True, alpha=0.3, linestyle="--")

    plt.suptitle(title, fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_ablation_study(ablation_rows: list, save_path: str):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    names = [r["name"] for r in ablation_rows]
    val_accs = [r["best_val_acc"] * 100 for r in ablation_rows]
    val_losses = [r["best_val_loss"] for r in ablation_rows]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.2))
    y_pos = np.arange(len(names))

    colors = ["#2ca02c" if i == int(np.argmin(val_losses)) else "#4c72b0" for i in range(len(names))]
    ax1.barh(y_pos, val_accs, color=colors, edgecolor="#222222", alpha=0.9)
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(names, fontsize=9)
    ax1.invert_yaxis()
    ax1.set_xlabel("Validation Accuracy (%)")
    ax1.set_title("Validation Accuracy par Configuration", fontweight="bold")
    ax1.set_xlim(max(60, min(val_accs) - 5), 100)
    ax1.grid(axis="x", alpha=0.3, linestyle="--")
    for idx, v in enumerate(val_accs):
        ax1.text(v + 0.3, idx, f"{v:.2f}%", va="center", fontsize=8.5)

    ax2.barh(y_pos, val_losses, color=colors, edgecolor="#222222", alpha=0.9)
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(["" for _ in names])
    ax2.invert_yaxis()
    ax2.set_xlabel("Validation Loss (plus bas = meilleur)")
    ax2.set_title("Validation Loss par Configuration", fontweight="bold")
    ax2.set_xlim(0, max(val_losses) * 1.20)
    ax2.grid(axis="x", alpha=0.3, linestyle="--")
    for idx, v in enumerate(val_losses):
        ax2.text(v + max(val_losses) * 0.02, idx, f"{v:.4f}", va="center", fontsize=8.5)

    plt.suptitle("Étude d'Ablation et Optimisation (Sélection sur Validation Uniquement)", fontsize=12.5, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def run_validation_optimization(data_path: str = None, seed: int = 42):
    """
    Exécute la suite complète d'ablations (BatchNorm, Dropout, Weight Decay,
    Augmentation, Dual Pooling, Scheduler) et sélectionne le meilleur modèle
    exclusivement selon la validation.
    """
    project_root = os.path.dirname(os.path.abspath(__file__))
    saved_dir = os.path.join(project_root, "saved_model_improved")
    graphs_dir = os.path.join(project_root, "results_improved", "graphs")
    metrics_dir = os.path.join(project_root, "results_improved", "metrics")
    os.makedirs(saved_dir, exist_ok=True)
    os.makedirs(graphs_dir, exist_ok=True)
    os.makedirs(metrics_dir, exist_ok=True)

    candidates = [
        {
            "name": "ablation_no_bn",
            "description": "Sans BatchNorm (teste l'impact de BatchNorm)",
            "use_bn": False, "dropout_conv1": 0.10, "dropout_conv2": 0.15,
            "dropout_fc1": 0.35, "dropout_fc2": 0.20, "weight_decay": 1e-3,
            "augment": True, "pooling_mode": "dual", "scheduler": "cosine", "lr": 0.002,
        },
        {
            "name": "ablation_no_dropout",
            "description": "Sans Dropout (teste l'impact du Dropout)",
            "use_bn": True, "dropout_conv1": 0.0, "dropout_conv2": 0.0,
            "dropout_fc1": 0.0, "dropout_fc2": 0.0, "weight_decay": 1e-3,
            "augment": True, "pooling_mode": "dual", "scheduler": "cosine", "lr": 0.002,
        },
        {
            "name": "ablation_no_weight_decay",
            "description": "Sans Weight Decay (wd=0)",
            "use_bn": True, "dropout_conv1": 0.10, "dropout_conv2": 0.15,
            "dropout_fc1": 0.35, "dropout_fc2": 0.20, "weight_decay": 0.0,
            "augment": True, "pooling_mode": "dual", "scheduler": "cosine", "lr": 0.002,
        },
        {
            "name": "ablation_no_augmentation",
            "description": "Sans Augmentation thermique",
            "use_bn": True, "dropout_conv1": 0.10, "dropout_conv2": 0.15,
            "dropout_fc1": 0.35, "dropout_fc2": 0.20, "weight_decay": 1e-3,
            "augment": False, "pooling_mode": "dual", "scheduler": "cosine", "lr": 0.002,
        },
        {
            "name": "ablation_gap_only",
            "description": "Pooling GAP seul (sans GMP)",
            "use_bn": True, "dropout_conv1": 0.10, "dropout_conv2": 0.15,
            "dropout_fc1": 0.35, "dropout_fc2": 0.20, "weight_decay": 1e-3,
            "augment": True, "pooling_mode": "avg", "scheduler": "cosine", "lr": 0.002,
        },
        {
            "name": "opt_cosine_wd1e3",
            "description": "CNN Optimisé (BN + DualPool + Dropout + wd=1e-3 + Cosine + Aug)",
            "use_bn": True, "dropout_conv1": 0.10, "dropout_conv2": 0.15,
            "dropout_fc1": 0.35, "dropout_fc2": 0.20, "weight_decay": 1e-3,
            "augment": True, "pooling_mode": "dual", "scheduler": "cosine", "lr": 0.002,
        },
        {
            "name": "opt_plateau_wd2e3",
            "description": "CNN Optimisé (BN + DualPool + Dropout + wd=2e-3 + Plateau + Aug)",
            "use_bn": True, "dropout_conv1": 0.10, "dropout_conv2": 0.15,
            "dropout_fc1": 0.35, "dropout_fc2": 0.20, "weight_decay": 2e-3,
            "augment": True, "pooling_mode": "dual", "scheduler": "plateau", "lr": 0.002,
        },
        {
            "name": "opt_cosine_strong_reg",
            "description": "CNN Optimisé régularisation forte (wd=3e-3 + Dropout renforcé)",
            "use_bn": True, "dropout_conv1": 0.15, "dropout_conv2": 0.20,
            "dropout_fc1": 0.40, "dropout_fc2": 0.25, "weight_decay": 3e-3,
            "augment": True, "pooling_mode": "dual", "scheduler": "cosine", "lr": 0.0018,
        },
    ]

    print("=" * 72)
    print("   ÉTUDE D'ABLATION & OPTIMISATION DU CNN (VALIDATION UNIQUEMENT)")
    print("=" * 72)

    all_results = []
    for cfg in candidates:
        print(f"\n>>> Configuration : {cfg['name']} — {cfg['description']}")
        res = train_single_config(cfg, data_path=data_path, seed=seed, verbose=True)
        all_results.append(res)
        print(
            f"  => Résultat : Best Ep={res['best_epoch']} | "
            f"Train(eval) L={res['train_eval_loss']:.4f} A={res['train_eval_acc']*100:.2f}% | "
            f"Val L={res['best_val_loss']:.4f} A={res['best_val_acc']*100:.2f}% | "
            f"Gap={res['generalization_gap_acc']*100:+.2f}%"
        )

    # Sélection strictement basée sur la validation (perte minimale, puis exactitude maximale)
    best_res = min(all_results, key=lambda r: (r["best_val_loss"], -r["best_val_acc"]))

    print("\n" + "=" * 72)
    print(f"MEILLEUR MODÈLE SÉLECTIONNÉ SUR VALIDATION : {best_res['name']}")
    print(f"  - Validation Loss     : {best_res['best_val_loss']:.4f}")
    print(f"  - Validation Accuracy : {best_res['best_val_acc']*100:.2f}%")
    print(f"  - Train(eval) Loss    : {best_res['train_eval_loss']:.4f}")
    print(f"  - Train(eval) Acc     : {best_res['train_eval_acc']*100:.2f}%")
    print("=" * 72)

    best_model_path = os.path.join(saved_dir, "best_optimized_thermal_cnn.pth")
    torch.save({
        "name": best_res["name"],
        "description": best_res["description"],
        "model_state_dict": best_res["best_state"],
        "model_config": best_res["model_config"],
        "hyperparameters": best_res["hyperparameters"],
        "norm_stats": best_res["norm_stats"],
        "class_names": CLASS_NAMES,
        "best_epoch": best_res["best_epoch"],
        "train_eval_loss": best_res["train_eval_loss"],
        "train_eval_acc": best_res["train_eval_acc"],
        "best_val_loss": best_res["best_val_loss"],
        "best_val_accuracy": best_res["best_val_acc"],
        "history": best_res["history"],
    }, best_model_path)

    ablation_summary = [
        {
            "name": r["name"],
            "description": r["description"],
            "use_bn": r["model_config"]["use_bn"],
            "pooling_mode": r["model_config"]["pooling_mode"],
            "dropout_fc1": r["model_config"]["dropout_fc1"],
            "weight_decay": r["hyperparameters"]["weight_decay"],
            "scheduler": r["hyperparameters"]["scheduler"],
            "augment": r["hyperparameters"]["augment"],
            "num_parameters": r["num_parameters"],
            "best_epoch": r["best_epoch"],
            "train_eval_loss": r["train_eval_loss"],
            "train_eval_acc": r["train_eval_acc"],
            "best_val_loss": r["best_val_loss"],
            "best_val_acc": r["best_val_acc"],
            "generalization_gap_acc": r["generalization_gap_acc"],
            "elapsed_sec": r["elapsed_sec"],
            "selected": (r["name"] == best_res["name"]),
        }
        for r in all_results
    ]

    with open(os.path.join(metrics_dir, "ablation_validation_study.json"), "w", encoding="utf-8") as f:
        json.dump(ablation_summary, f, indent=4)
    pd.DataFrame(ablation_summary).to_csv(os.path.join(metrics_dir, "ablation_validation_study.csv"), index=False)

    plot_training_curves(
        best_res["history"],
        os.path.join(graphs_dir, "optimized_training_curves.png"),
        title=f"Courbes d'Apprentissage — {best_res['name']} (Dataset Amélioré)",
    )
    plot_ablation_study(ablation_summary, os.path.join(graphs_dir, "ablation_study.png"))

    return best_model_path, best_res, ablation_summary


if __name__ == "__main__":
    run_validation_optimization()
