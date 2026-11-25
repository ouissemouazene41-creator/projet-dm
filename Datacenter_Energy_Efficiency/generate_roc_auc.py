"""
ROC-AUC Curves — CNN Optimisé (opt_plateau_wd2e3)
Génère les courbes ROC multiclasse (One-vs-Rest) pour le meilleur modèle CNN.
Sauvegarde : results_improved/metrics/roc_auc.png
"""

import sys
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize

# ---------------------------------------------------------------------------
# Chemins
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "saved_model_improved", "best_optimized_thermal_cnn.pth")
DATASET_PATH = os.path.join(BASE_DIR, "dataset", "improved_thermal_maps.npz")
OUT_DIR = os.path.join(BASE_DIR, "results_improved", "metrics")
os.makedirs(OUT_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Import du modèle
# ---------------------------------------------------------------------------
sys.path.insert(0, BASE_DIR)
from cnn_model_improved import OptimizedThermalCNN

# ---------------------------------------------------------------------------
# Chargement dataset amélioré
# ---------------------------------------------------------------------------
print("Chargement du dataset amélioré …")
data = np.load(DATASET_PATH)
X_test = data["X_test"].astype(np.float32)
y_test = data["y_test"].astype(np.int64)
X_train = data["X_train"].astype(np.float32)

# Normalisation avec stats d'entraînement uniquement
mean_train = X_train.mean(axis=(0, 2, 3), keepdims=True)
std_train  = X_train.std(axis=(0, 2, 3),  keepdims=True) + 1e-6
X_test_norm = (X_test - mean_train) / std_train

print(f"  X_test shape : {X_test_norm.shape}  |  classes uniques : {np.unique(y_test)}")

# ---------------------------------------------------------------------------
# Chargement du modèle
# ---------------------------------------------------------------------------
print(f"Chargement du modèle depuis : {MODEL_PATH}")
device = torch.device("cpu")
checkpoint = torch.load(MODEL_PATH, map_location=device, weights_only=False)

# Instanciation du modèle
model = OptimizedThermalCNN(
    in_channels=2,
    num_classes=4,
    use_bn=True,
    dropout_conv1=0.10,
    dropout_conv2=0.15,
    dropout_fc1=0.35,
    dropout_fc2=0.20,
    pooling_mode="dual"
)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()
print(f"  Modèle chargé — meilleure époque : {checkpoint.get('epoch', 'N/A')}")

# ---------------------------------------------------------------------------
# Prédictions softmax
# ---------------------------------------------------------------------------
print("Calcul des probabilités softmax …")
with torch.no_grad():
    X_tensor = torch.tensor(X_test_norm)
    logits = model(X_tensor)
    probs = F.softmax(logits, dim=1).numpy()

# Binarisation des labels (One-vs-Rest)
n_classes = 4
class_names = [
    "Optimal / Froid (<22°C)",
    "Nominal ASHRAE (22-27°C)",
    "Alerte Point Chaud (27-32°C)",
    "Surchauffe Critique (>32°C)"
]
y_bin = label_binarize(y_test, classes=list(range(n_classes)))

# ---------------------------------------------------------------------------
# Calcul des courbes ROC par classe
# ---------------------------------------------------------------------------
fpr_dict, tpr_dict, roc_auc_dict = {}, {}, {}
for i in range(n_classes):
    fpr_dict[i], tpr_dict[i], _ = roc_curve(y_bin[:, i], probs[:, i])
    roc_auc_dict[i] = auc(fpr_dict[i], tpr_dict[i])

# Macro-average ROC
all_fpr = np.unique(np.concatenate([fpr_dict[i] for i in range(n_classes)]))
mean_tpr = np.zeros_like(all_fpr)
for i in range(n_classes):
    mean_tpr += np.interp(all_fpr, fpr_dict[i], tpr_dict[i])
mean_tpr /= n_classes
macro_auc = auc(all_fpr, mean_tpr)

# ---------------------------------------------------------------------------
# Tracé
# ---------------------------------------------------------------------------
colors = ["#2196F3", "#4CAF50", "#FF9800", "#F44336"]
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# --- Subplot 1 : courbes individuelles ---
ax1 = axes[0]
for i in range(n_classes):
    ax1.plot(fpr_dict[i], tpr_dict[i],
             color=colors[i], lw=2,
             label=f"Classe {i}: {class_names[i]}\n(AUC = {roc_auc_dict[i]:.4f})")

ax1.plot(all_fpr, mean_tpr, color="black", lw=2.5, linestyle="--",
         label=f"Macro-moyenne (AUC = {macro_auc:.4f})")
ax1.plot([0, 1], [0, 1], "k:", lw=1, label="Aléatoire (AUC = 0.50)")
ax1.set_xlim([0.0, 1.0])
ax1.set_ylim([0.0, 1.02])
ax1.set_xlabel("Taux de Faux Positifs (FPR)", fontsize=12)
ax1.set_ylabel("Taux de Vrais Positifs (TPR)", fontsize=12)
ax1.set_title("Courbes ROC — CNN Optimisé (One-vs-Rest)\nDataset Amélioré · Test Set (n=600)", fontsize=13)
ax1.legend(loc="lower right", fontsize=9)
ax1.grid(alpha=0.3)

# --- Subplot 2 : barres AUC par classe ---
ax2 = axes[1]
aucs = [roc_auc_dict[i] for i in range(n_classes)]
bars = ax2.barh(class_names, aucs, color=colors, edgecolor="black", alpha=0.85)
ax2.axvline(x=macro_auc, color="black", linestyle="--", lw=2,
            label=f"Macro-AUC = {macro_auc:.4f}")
ax2.set_xlim([0.80, 1.005])
ax2.set_xlabel("AUC-ROC", fontsize=12)
ax2.set_title("AUC-ROC par Classe\n(One-vs-Rest)", fontsize=13)
ax2.legend(fontsize=10)
ax2.grid(axis="x", alpha=0.3)
for bar, v in zip(bars, aucs):
    ax2.text(v + 0.001, bar.get_y() + bar.get_height() / 2,
             f"{v:.4f}", va="center", ha="left", fontsize=11, fontweight="bold")

plt.suptitle(
    f"Analyse ROC-AUC — Modèle opt_plateau_wd2e3  |  Test Accuracy = 91.50%  |  Macro-AUC = {macro_auc:.4f}",
    fontsize=12, fontweight="bold", y=1.02
)
plt.tight_layout()

out_path = os.path.join(OUT_DIR, "roc_auc.png")
plt.savefig(out_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"\n✅ Courbes ROC-AUC sauvegardées : {out_path}")

# Affichage résumé
print("\n" + "="*55)
print("RÉSUMÉ AUC-ROC (Test Set)")
print("="*55)
for i in range(n_classes):
    print(f"  Classe {i} ({class_names[i][:30]:30s}) : AUC = {roc_auc_dict[i]:.4f}")
print(f"  {'Macro-moyenne':42s} : AUC = {macro_auc:.4f}")
print("="*55)

# Sauvegarde JSON
import json
auc_results = {
    "model": "opt_plateau_wd2e3",
    "test_accuracy": 0.9150,
    "roc_auc_per_class": {class_names[i]: round(roc_auc_dict[i], 6) for i in range(n_classes)},
    "macro_roc_auc": round(macro_auc, 6)
}
json_path = os.path.join(OUT_DIR, "roc_auc_results.json")
with open(json_path, "w", encoding="utf-8") as f:
    json.dump(auc_results, f, indent=4, ensure_ascii=False)
print(f"✅ Résultats JSON sauvegardés : {json_path}")
