"""
Projet : Efficacité Énergétique des Centres de Données (Datacenters)
Fichier : evaluate_improved.py
Description : Évaluation finale indépendante complète du CNN optimisé sur Train,
              Validation et Test Set strictement inédit.

Calcule et sauvegarde :
  - Train Accuracy, Validation Accuracy, Test Accuracy
  - Train Loss, Validation Loss, Test Loss
  - Macro Precision, Macro Recall, Macro F1-Score, Weighted F1-Score
  - Matrice de confusion (JSON + PNG)
  - Rapport de classification détaillé (TXT + JSON)
"""

import os
import json
import tempfile
import numpy as np

os.environ.setdefault("MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "matplotlib_cache"))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_recall_fscore_support

from cnn_model_improved import OptimizedThermalCNN
from preprocessing_improved import load_improved_splits, ImprovedThermalDataset, CLASS_NAMES


def plot_confusion_matrix(cm: np.ndarray, class_names: list, save_path: str, title: str):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    cm_norm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

    plt.figure(figsize=(8.5, 6.5))
    annot = np.empty_like(cm, dtype=object)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            annot[i, j] = f"{cm_norm[i, j]*100:.1f}%\n({cm[i, j]})"

    sns.heatmap(
        cm_norm,
        annot=annot,
        fmt="",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        cbar=True,
        linewidths=0.5,
        linecolor="#dddddd",
    )
    plt.title(title, fontsize=12, fontweight="bold", pad=12)
    plt.xlabel("Classe Prédite", fontsize=11, labelpad=8)
    plt.ylabel("Vraie Classe (ASHRAE TC 9.9)", fontsize=11, labelpad=8)
    plt.xticks(rotation=20, ha="right", fontsize=9)
    plt.yticks(rotation=0, fontsize=9)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def _run_eval_loader(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    preds, targets, probs = [], [], []
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = criterion(logits, y)
            total_loss += loss.item() * len(y)
            p = torch.softmax(logits, dim=-1)
            preds.extend(logits.argmax(dim=1).cpu().numpy())
            targets.extend(y.cpu().numpy())
            probs.extend(p.cpu().numpy())
    preds = np.array(preds)
    targets = np.array(targets)
    return total_loss / len(targets), accuracy_score(targets, preds), targets, preds, np.array(probs)


def evaluate_improved_model(model_path: str = None, data_path: str = None, batch_size: int = 32) -> dict:
    project_root = os.path.dirname(os.path.abspath(__file__))
    if model_path is None:
        model_path = os.path.join(project_root, "saved_model_improved", "best_optimized_thermal_cnn.pth")

    metrics_dir = os.path.join(project_root, "results_improved", "metrics")
    os.makedirs(metrics_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)

    model_cfg = checkpoint["model_config"]
    class_names = checkpoint.get("class_names", CLASS_NAMES)
    norm_stats = checkpoint["norm_stats"]

    model = OptimizedThermalCNN(**model_cfg).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    (X_train, y_train), (X_val, y_val), (X_test, y_test), _, _ = load_improved_splits(data_path)
    mean, std = norm_stats["mean"], norm_stats["std"]

    tr_loader = DataLoader(ImprovedThermalDataset(X_train, y_train, mean, std, augment=False), batch_size=batch_size, shuffle=False)
    va_loader = DataLoader(ImprovedThermalDataset(X_val, y_val, mean, std, augment=False), batch_size=batch_size, shuffle=False)
    te_loader = DataLoader(ImprovedThermalDataset(X_test, y_test, mean, std, augment=False), batch_size=batch_size, shuffle=False)

    criterion = nn.CrossEntropyLoss()
    train_loss, train_acc, _, _, _ = _run_eval_loader(model, tr_loader, criterion, device)
    val_loss, val_acc, _, _, _ = _run_eval_loader(model, va_loader, criterion, device)
    test_loss, test_acc, all_targets, all_preds, _ = _run_eval_loader(model, te_loader, criterion, device)

    prec_macro, rec_macro, f1_macro, _ = precision_recall_fscore_support(all_targets, all_preds, average="macro")
    prec_weighted, rec_weighted, f1_weighted, _ = precision_recall_fscore_support(all_targets, all_preds, average="weighted")

    report_str = classification_report(all_targets, all_preds, target_names=class_names, digits=4)
    report_dict = classification_report(all_targets, all_preds, target_names=class_names, digits=4, output_dict=True)
    cm = confusion_matrix(all_targets, all_preds)

    results_dict = {
        "model_name": checkpoint.get("name", "OptimizedThermalCNN"),
        "num_parameters": int(model.count_parameters()),
        "best_epoch": int(checkpoint.get("best_epoch", -1)),
        "train_samples": int(len(y_train)),
        "val_samples": int(len(y_val)),
        "test_samples": int(len(y_test)),
        "train_loss": float(train_loss),
        "val_loss": float(val_loss),
        "test_loss": float(test_loss),
        "train_accuracy": float(train_acc),
        "val_accuracy": float(val_acc),
        "test_accuracy": float(test_acc),
        "macro_precision": float(prec_macro),
        "macro_recall": float(rec_macro),
        "macro_f1_score": float(f1_macro),
        "weighted_f1_score": float(f1_weighted),
        "classification_report": report_dict,
        "confusion_matrix": cm.tolist(),
    }

    print("\n" + "=" * 72)
    print("   ÉVALUATION FINALE DU CNN OPTIMISÉ (TEST SET INDÉPENDANT)")
    print("=" * 72)
    print(f"Modèle               : {results_dict['model_name']} ({results_dict['num_parameters']:,} paramètres)")
    print(f"Train Loss / Acc     : {train_loss:.4f} / {train_acc*100:.2f}%")
    print(f"Val   Loss / Acc     : {val_loss:.4f} / {val_acc*100:.2f}%")
    print(f"Test  Loss / Acc     : {test_loss:.4f} / {test_acc*100:.2f}%")
    print(f"Macro Precision      : {prec_macro*100:.2f}%")
    print(f"Macro Recall         : {rec_macro*100:.2f}%")
    print(f"Macro F1-Score       : {f1_macro*100:.2f}%")
    print("-" * 72)
    print(report_str)
    print("=" * 72)

    report_file = os.path.join(metrics_dir, "classification_report.txt")
    with open(report_file, "w", encoding="utf-8") as f:
        f.write("=" * 72 + "\n")
        f.write("RAPPORT D'ÉVALUATION — CNN OPTIMISÉ (TEST SET STRICTEMENT INDÉPENDANT)\n")
        f.write("=" * 72 + "\n\n")
        f.write(f"Modèle            : {results_dict['model_name']} ({results_dict['num_parameters']:,} params)\n")
        f.write(f"Train Loss / Acc  : {train_loss:.4f} / {train_acc*100:.2f}%\n")
        f.write(f"Val   Loss / Acc  : {val_loss:.4f} / {val_acc*100:.2f}%\n")
        f.write(f"Test  Loss / Acc  : {test_loss:.4f} / {test_acc*100:.2f}%\n")
        f.write(f"Macro Precision   : {prec_macro*100:.2f}%\n")
        f.write(f"Macro Recall      : {rec_macro*100:.2f}%\n")
        f.write(f"Macro F1-Score    : {f1_macro*100:.2f}%\n")
        f.write(f"Weighted F1-Score : {f1_weighted*100:.2f}%\n\n")
        f.write("Détail par classe :\n")
        f.write(report_str + "\n")

    with open(os.path.join(metrics_dir, "test_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(results_dict, f, indent=4)

    plot_confusion_matrix(
        cm, class_names,
        os.path.join(project_root, "results_improved", "confusion_matrix_optimized.png"),
        "Matrice de Confusion — CNN Optimisé (Dataset Amélioré)"
    )
    plot_confusion_matrix(
        cm, class_names,
        os.path.join(metrics_dir, "confusion_matrix.png"),
        "Matrice de Confusion — CNN Optimisé (Dataset Amélioré)"
    )

    return results_dict


if __name__ == "__main__":
    evaluate_improved_model()
