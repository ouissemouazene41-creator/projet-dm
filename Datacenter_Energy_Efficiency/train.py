"""
Projet : Efficacité Énergétique des Centres de Données (Datacenters)
Fichier : train.py
Description : Entraînement optimisé du modèle CNN avec contrôle rigoureux du surapprentissage.

Techniques d'optimisation et anti-surapprentissage appliquées :
  - Optimiseur AdamW avec découplage du gradient et pénalité L2 (weight_decay=1e-3).
  - Ordonnanceur de taux d'apprentissage CosineAnnealingLR pour une convergence douce.
  - Critère de perte CrossEntropyLoss avec label smoothing léger optionnel.
  - Surveillance simultanée des pertes (Train/Val) et précisions (Train/Val).
  - Mécanisme d'arrêt anticipé (Early Stopping avec patience configurable) basé sur la perte de validation.
  - Sauvegarde exclusive du meilleur point de contrôle (best checkpoint) dans 'saved_model/best_thermal_cnn.pth'.
  - Génération automatique des courbes comparatives Train vs Val (Loss & Accuracy) dans 'results/graphs/training_curves.png'.
"""

import os
import sys
import time
import tempfile
import json
import numpy as np

# Répertoire de cache sécurisé pour Matplotlib sous Windows
os.environ.setdefault("MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "matplotlib_cache"))

import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# Imports locaux
from cnn_model import DatacenterThermalCNN
from preprocessing import load_and_preprocess_data, CLASS_NAMES


def set_seed(seed: int = 42):
    """Garantit la reproductibilité absolue des expérimentations."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def train_one_epoch(model, dataloader, criterion, optimizer, device):
    """Exécute une passe complète d'entraînement avec rétropropagation."""
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

        # Gradient clipping pour stabiliser les gradients
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)

        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        _, preds = torch.max(outputs, 1)
        correct += torch.sum(preds == targets).item()
        total += targets.size(0)

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


def validate(model, dataloader, criterion, device):
    """Évalue le modèle en mode évaluation (sans calcul de gradients)."""
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
            _, preds = torch.max(outputs, 1)
            correct += torch.sum(preds == targets).item()
            total += targets.size(0)

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


def plot_training_curves(history: dict, save_path: str):
    """
    Génère des courbes scientifiques montrant l'évolution des pertes et précisions
    pour analyser visuellement la généralisation et l'absence de surapprentissage.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    epochs = range(1, len(history["train_loss"]) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # 1. Courbe de perte (Loss)
    ax1.plot(epochs, history["train_loss"], "o-", color="#1f77b4", label="Entraînement (Train Loss)", lw=2, markersize=4)
    ax1.plot(epochs, history["val_loss"], "s-", color="#ff7f0e", label="Validation (Val Loss)", lw=2, markersize=4)
    ax1.set_title("Évolution de la Perte (Cross-Entropy Loss)", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Époque", fontsize=11)
    ax1.set_ylabel("Perte (Loss)", fontsize=11)
    ax1.legend(loc="upper right", frameon=True)
    ax1.grid(True, alpha=0.3, linestyle="--")

    # 2. Courbe de précision (Accuracy %)
    train_acc_pct = [acc * 100 for acc in history["train_acc"]]
    val_acc_pct = [acc * 100 for acc in history["val_acc"]]

    ax2.plot(epochs, train_acc_pct, "o-", color="#1f77b4", label="Entraînement (Train Acc)", lw=2, markersize=4)
    ax2.plot(epochs, val_acc_pct, "s-", color="#ff7f0e", label="Validation (Val Acc)", lw=2, markersize=4)
    ax2.axhline(y=86.0, color="green", linestyle=":", lw=1.5, label="Seuil Cible (86.0%)")
    ax2.set_title("Évolution de la Précision (Accuracy %)", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Époque", fontsize=11)
    ax2.set_ylabel("Précision (%)", fontsize=11)
    ax2.set_ylim(0, 105)
    ax2.legend(loc="lower right", frameon=True)
    ax2.grid(True, alpha=0.3, linestyle="--")

    plt.suptitle("Dynamique d'Apprentissage et Contrôle du Surapprentissage (Projet Datacenter)", fontsize=13, y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[Graphiques] Courbes d'entraînement enregistrées : {save_path}")


def train(
    data_path: str = None,
    epochs: int = 25,
    batch_size: int = 32,
    lr: float = 0.002,
    weight_decay: float = 1e-3,
    patience: int = 7,
    seed: int = 42
):
    """
    Fonction principale orchestrant l'entraînement, la validation et la sauvegarde du modèle.
    """
    project_root = os.path.dirname(os.path.abspath(__file__))
    saved_model_dir = os.path.join(project_root, "saved_model")
    graphs_dir = os.path.join(project_root, "results", "graphs")
    metrics_dir = os.path.join(project_root, "results", "metrics")

    os.makedirs(saved_model_dir, exist_ok=True)
    os.makedirs(graphs_dir, exist_ok=True)
    os.makedirs(metrics_dir, exist_ok=True)

    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 70)
    print("   ENTRAÎNEMENT OPTIMISÉ DU CNN — EFFICACITÉ ÉNERGÉTIQUE DATACENTER")
    print("=" * 70)
    print(f"Périphérique de calcul : {device}")

    # 1. Chargement et prétraitement des données réelles
    train_loader, val_loader, test_loader, norm_stats, summary = load_and_preprocess_data(
        data_path=data_path,
        batch_size=batch_size,
        random_state=seed
    )
    print(f"Jeu d'entraînement    : {summary['train_size']} échantillons (Stratifié 70%)")
    print(f"Jeu de validation     : {summary['val_size']} échantillons (Stratifié 15%)")
    print(f"Jeu de test (inédit)  : {summary['test_size']} échantillons (Stratifié 15%)")

    # 2. Instanciation du modèle compact et optimisé
    model = DatacenterThermalCNN(in_channels=2, num_classes=4).to(device)
    print(f"Modèle initialisé      : {model.count_parameters():,} paramètres entraînables (compact & anti-overfitting)")

    # 3. Fonction de perte, Optimiseur et Scheduler
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    # 4. Boucle d'entraînement avec Early Stopping
    history = {
        "train_loss": [], "train_acc": [],
        "val_loss": [], "val_acc": []
    }

    best_val_loss = float("inf")
    best_val_acc = 0.0
    best_epoch = 0
    epochs_no_improve = 0
    best_model_path = os.path.join(saved_model_dir, "best_thermal_cnn.pth")

    start_time = time.time()
    print("\n" + "-" * 70)
    print(f"{'Époque':<8} | {'Train Loss':<11} | {'Train Acc':<10} | {'Val Loss':<10} | {'Val Acc':<10} | {'Statut'}")
    print("-" * 70)

    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        scheduler.step()

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        # Critère de sauvegarde : meilleure Val Loss (indicateur le plus robuste contre l'overfitting)
        is_best = val_loss < best_val_loss or (val_loss <= best_val_loss and val_acc >= best_val_acc)

        if is_best:
            best_val_loss = val_loss
            best_val_acc = val_acc
            best_epoch = epoch
            epochs_no_improve = 0
            torch.save({
                "model_state_dict": model.state_dict(),
                "epoch": epoch,
                "best_val_loss": best_val_loss,
                "best_val_accuracy": best_val_acc,
                "norm_stats": norm_stats,
                "class_names": CLASS_NAMES,
                "hyperparameters": {
                    "lr": lr,
                    "weight_decay": weight_decay,
                    "batch_size": batch_size,
                    "optimizer": "AdamW",
                    "scheduler": "CosineAnnealingLR"
                }
            }, best_model_path)
            status = "★ Meilleur modèle"
        else:
            epochs_no_improve += 1
            status = f"Patience {epochs_no_improve}/{patience}"

        print(f"{epoch:02d}/{epochs:02d}    | {train_loss:<11.4f} | {train_acc*100:>6.2f}%   | {val_loss:<10.4f} | {val_acc*100:>6.2f}%   | {status}")

        if epochs_no_improve >= patience:
            print("-" * 70)
            print(f"[Early Stopping] Arrêt anticipé à l'époque {epoch}. Pas d'amélioration depuis {patience} époques.")
            break

    elapsed = time.time() - start_time
    print("-" * 70)
    print(f"Entraînement terminé en {elapsed:.1f} secondes.")
    print(f"Meilleur modèle atteint à l'époque {best_epoch} :")
    print(f"  - Validation Loss     : {best_val_loss:.4f}")
    print(f"  - Validation Accuracy : {best_val_acc*100:.2f}%")
    print(f"  - Fichier sauvegardé  : {best_model_path}")

    # 5. Sauvegarde des courbes
    curves_path = os.path.join(graphs_dir, "training_curves.png")
    plot_training_curves(history, curves_path)

    # 6. Évaluation finale immédiate sur le jeu de test indépendant
    print("\n" + "=" * 70)
    print("   ÉVALUATION DU MEILLEUR MODÈLE SUR LE TEST SET (DONNÉES INÉDITES)")
    print("=" * 70)
    from evaluate import evaluate_model
    test_metrics = evaluate_model(
        model_path=best_model_path,
        data_path=data_path,
        batch_size=batch_size
    )

    return model, history, test_metrics


if __name__ == "__main__":
    train()
