"""
Projet : Efficacité Énergétique des Centres de Données (Datacenters)
Fichier : evaluate.py
Description : Évaluation rigoureuse et scientifique du modèle CNN sur le Test Set inédit.

Métriques calculées :
  - Exactitude globale (Test Accuracy %)
  - Précision (Precision macro et weighted)
  - Rappel (Recall macro et weighted)
  - F1-Score (macro et weighted)
  - Rapport de classification détaillé par classe ASHRAE
  - Matrice de confusion normalisée (sauvegardée en haute résolution)
"""

import os
import sys
import json
import tempfile
import numpy as np

os.environ.setdefault("MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "matplotlib_cache"))

import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_recall_fscore_support

from cnn_model import DatacenterThermalCNN
from preprocessing import load_and_preprocess_data, CLASS_NAMES


def plot_confusion_matrix(cm: np.ndarray, class_names: list, save_path: str):
    """Trace et sauvegarde une matrice de confusion normalisée de haute qualité."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    cm_norm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

    plt.figure(figsize=(8.5, 6.5))
    sns.heatmap(
        cm_norm,
        annot=True,
        fmt=".2%",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        cbar=True,
        linewidths=0.5,
        linecolor="#dddddd"
    )

    plt.title("Matrice de Confusion Normalisée — Modèle CNN Datacenter", fontsize=12, fontweight="bold", pad=12)
    plt.xlabel("Classe Prédite par le CNN", fontsize=11, labelpad=8)
    plt.ylabel("Vraie Classe (ASHRAE TC 9.9)", fontsize=11, labelpad=8)
    plt.xticks(rotation=20, ha="right", fontsize=9)
    plt.yticks(rotation=0, fontsize=9)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[Graphiques] Matrice de confusion enregistrée : {save_path}")


def evaluate_model(
    model_path: str = None,
    data_path: str = None,
    batch_size: int = 32
) -> dict:
    """
    Charge le meilleur modèle sauvegardé et calcule toutes les métriques
    sur le jeu de test indépendant (450 échantillons inédits).
    """
    project_root = os.path.dirname(os.path.abspath(__file__))
    if model_path is None:
        model_path = os.path.join(project_root, "saved_model", "best_thermal_cnn.pth")
    if data_path is None:
        data_path = os.path.join(project_root, "dataset", "datacenter_thermal_maps.npz")

    metrics_dir = os.path.join(project_root, "results", "metrics")
    cm_path = os.path.join(project_root, "results", "confusion_matrix.png")
    os.makedirs(metrics_dir, exist_ok=True)

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Modèle introuvable : {model_path}. Lancez d'abord train.py.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Chargement du checkpoint
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    class_names = checkpoint.get("class_names", CLASS_NAMES)

    # 2. Instanciation et chargement des poids
    model = DatacenterThermalCNN(in_channels=2, num_classes=len(class_names)).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # 3. Chargement des DataLoaders avec les mêmes statistiques de prétraitement
    _, _, test_loader, _, summary = load_and_preprocess_data(
        data_path=data_path,
        batch_size=batch_size
    )

    all_preds = []
    all_targets = []
    all_probs = []

    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            probs = torch.softmax(outputs, dim=-1)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.numpy())
            all_probs.extend(probs.cpu().numpy())

    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    all_probs = np.array(all_probs)

    # 4. Calcul des métriques scientifiques
    test_accuracy = float(accuracy_score(all_targets, all_preds))
    prec_macro, rec_macro, f1_macro, _ = precision_recall_fscore_support(all_targets, all_preds, average="macro")
    prec_weighted, rec_weighted, f1_weighted, _ = precision_recall_fscore_support(all_targets, all_preds, average="weighted")

    report_str = classification_report(all_targets, all_preds, target_names=class_names, digits=4)
    report_dict = classification_report(all_targets, all_preds, target_names=class_names, digits=4, output_dict=True)
    cm = confusion_matrix(all_targets, all_preds)

    # 5. Affichage clair des résultats
    print("\n" + "=" * 70)
    print("   RÉSULTATS DE L'ÉVALUATION SCIENTIFIQUE SUR LE TEST SET")
    print("=" * 70)
    print(f"Échantillons de test : {len(all_targets)} (données strictement inédites)")
    print(f"Test Accuracy        : {test_accuracy * 100:.2f}%  (Objectif cible : >= 86.00%)")
    print(f"Macro Precision      : {prec_macro * 100:.2f}%")
    print(f"Macro Recall         : {rec_macro * 100:.2f}%")
    print(f"Macro F1-Score       : {f1_macro * 100:.2f}%")
    print(f"Weighted F1-Score    : {f1_weighted * 100:.2f}%")
    print("-" * 70)
    print("Rapport de classification détaillé :")
    print(report_str)
    print("=" * 70)

    # 6. Sauvegarde des livrables
    report_file = os.path.join(metrics_dir, "classification_report.txt")
    with open(report_file, "w", encoding="utf-8") as f:
        f.write("=" * 70 + "\n")
        f.write("RAPPORT D'ÉVALUATION DU MODÈLE CNN — TEST SET INÉDIT\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Test Accuracy     : {test_accuracy * 100:.2f}%\n")
        f.write(f"Macro Precision   : {prec_macro * 100:.2f}%\n")
        f.write(f"Macro Recall      : {rec_macro * 100:.2f}%\n")
        f.write(f"Macro F1-Score    : {f1_macro * 100:.2f}%\n")
        f.write(f"Weighted F1-Score : {f1_weighted * 100:.2f}%\n\n")
        f.write("Détail par classe :\n")
        f.write(report_str + "\n")
    print(f"[Métriques] Rapport textuel enregistré : {report_file}")

    metrics_json_file = os.path.join(metrics_dir, "test_metrics.json")
    results_dict = {
        "test_samples": int(len(all_targets)),
        "test_accuracy": float(test_accuracy),
        "macro_precision": float(prec_macro),
        "macro_recall": float(rec_macro),
        "macro_f1_score": float(f1_macro),
        "weighted_f1_score": float(f1_weighted),
        "classification_report": report_dict,
        "confusion_matrix": cm.tolist()
    }
    with open(metrics_json_file, "w", encoding="utf-8") as f:
        json.dump(results_dict, f, indent=4)
    print(f"[Métriques] Métriques JSON enregistrées : {metrics_json_file}")

    # 7. Graphique de la matrice de confusion
    plot_confusion_matrix(cm, class_names, cm_path)

    # Également sauvegarder dans metrics/ pour respecter les deux conventions
    plot_confusion_matrix(cm, class_names, os.path.join(metrics_dir, "confusion_matrix.png"))

    return results_dict


if __name__ == "__main__":
    evaluate_model()
