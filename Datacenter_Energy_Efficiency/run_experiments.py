"""
Projet : Efficacité Énergétique des Centres de Données (Datacenters)
Fichier : run_experiments.py
Description : Pipeline scientifique complet exécutant toutes les expérimentations
              comparatives, l'audit des datasets, les baselines classiques et le
              test de permutation spatiale.

Expérimentations exécutées :
  1. Audit complet : Dataset Original vs Dataset Amélioré (doublons, fuites, stats)
  2. Expérience 1  : Dataset Original + CNN Original (DatacenterThermalCNN)
  3. Expérience 2  : Dataset Amélioré + CNN Original (DatacenterThermalCNN)
  4. Expérience 3  : Dataset Amélioré + CNN Optimisé (étude d'ablation & sélection
                     exclusivement sur Validation, puis évaluation unique sur Test)
  5. Baselines     : Régression Logistique & Arbre de Décision (Original & Amélioré)
  6. Test Spatial  : Permutation spatiale des pixels (évalue l'apprentissage spatial)
"""

import os
import sys
import json
import time
import tempfile
import numpy as np
import pandas as pd

os.environ.setdefault("MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "matplotlib_cache"))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial.distance import cdist
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score,
    log_loss,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report,
)

from cnn_model import DatacenterThermalCNN
from preprocessing import DatacenterThermalDataset, CLASS_NAMES
from preprocessing_improved import load_improved_splits, ImprovedThermalDataset
from generate_improved_dataset import verify_split_independence
from train_improved import set_seed, train_one_epoch, evaluate_split, run_validation_optimization
from evaluate_improved import evaluate_improved_model, _run_eval_loader


def audit_original_dataset(orig_path: str) -> dict:
    loaded = np.load(orig_path)
    maps, labels = loaded["maps"], loaded["labels"]

    X_tv, X_te, y_tv, y_te = train_test_split(maps, labels, test_size=0.15, stratify=labels, random_state=42)
    X_tr, X_va, y_tr, y_va = train_test_split(X_tv, y_tv, test_size=0.15 / 0.85, stratify=y_tv, random_state=42)

    mean_tr = X_tr.mean(axis=(0, 2, 3), keepdims=True)
    std_tr = X_tr.std(axis=(0, 2, 3), keepdims=True) + 1e-7

    tr_n = ((X_tr - mean_tr) / std_tr).reshape(len(X_tr), -1)
    va_n = ((X_va - mean_tr) / std_tr).reshape(len(X_va), -1)
    te_n = ((X_te - mean_tr) / std_tr).reshape(len(X_te), -1)
    all_n = np.vstack([tr_n, va_n, te_n])

    d_va_tr = cdist(va_n, tr_n).min(axis=1)
    d_te_tr = cdist(te_n, tr_n).min(axis=1)

    class_stats = {}
    for c in range(4):
        sub_t = maps[labels == c, 0]
        sub_h = maps[labels == c, 1]
        class_stats[int(c)] = {
            "count": int((labels == c).sum()),
            "mean_temp_range": [float(sub_t.mean((1, 2)).min()), float(sub_t.mean((1, 2)).max())],
            "max_temp_range": [float(sub_t.max((1, 2)).min()), float(sub_t.max((1, 2)).max())],
            "mean_hum_range": [float(sub_h.mean((1, 2)).min()), float(sub_h.mean((1, 2)).max())],
        }

    return {
        "total_samples": int(len(labels)),
        "shape": list(maps.shape),
        "nans": int(np.isnan(maps).sum()),
        "exact_unique_samples": int(len(np.unique(all_n, axis=0))),
        "nn_l2_val_to_train": {"min": float(d_va_tr.min()), "median": float(np.median(d_va_tr))},
        "nn_l2_test_to_train": {"min": float(d_te_tr.min()), "median": float(np.median(d_te_tr))},
        "class_stats": class_stats,
    }


def audit_improved_dataset(imp_path: str) -> dict:
    (X_train, y_train), (X_val, y_val), (X_test, y_test), _, summary = load_improved_splits(imp_path)
    indep = verify_split_independence(X_train, X_val, X_test)

    class_stats = {}
    for c in range(4):
        sub_t = X_train[y_train == c, 0]
        sub_h = X_train[y_train == c, 1]
        class_stats[int(c)] = {
            "train_count": int((y_train == c).sum()),
            "mean_temp_range": [float(sub_t.mean((1, 2)).min()), float(sub_t.mean((1, 2)).max())],
            "max_temp_range": [float(sub_t.max((1, 2)).min()), float(sub_t.max((1, 2)).max())],
            "mean_hum_range": [float(sub_h.mean((1, 2)).min()), float(sub_h.mean((1, 2)).max())],
        }

    return {
        "summary": summary,
        "independence": indep,
        "class_stats_train": class_stats,
    }


def run_spatial_permutation_test(model, X_test_raw: np.ndarray, y_test: np.ndarray, norm_stats: dict, device, seeds=(10, 20, 30, 40, 50)):
    """
    Évalue la sensibilité spatiale du CNN en permutant aléatoirement les 256 positions
    (16x16) des capteurs tout en conservant exactement la distribution/histogramme des
    valeurs de température et d'humidité de chaque échantillon.
    Teste deux protocoles :
      - Permutation spatiale fixe partagée sur tout le test set (détruit la structure 2D)
      - Permutation spatiale indépendante par échantillon
    """
    model.eval()
    mean, std = norm_stats["mean"], norm_stats["std"]
    X_norm = ((X_test_raw - mean) / std).astype(np.float32)
    N, C, H, W = X_norm.shape

    fixed_accs = []
    per_sample_accs = []

    for s in seeds:
        rng = np.random.default_rng(s)
        # 1. Permutation fixe des 256 positions spatiales
        perm_idx = rng.permutation(H * W)
        X_fixed = X_norm.reshape(N, C, H * W)[:, :, perm_idx].reshape(N, C, H, W)
        with torch.no_grad():
            preds_f = []
            for start in range(0, N, 64):
                xb = torch.from_numpy(X_fixed[start:start + 64]).to(device)
                preds_f.extend(model(xb).argmax(dim=1).cpu().numpy())
            fixed_accs.append(float(accuracy_score(y_test, preds_f)))

        # 2. Permutation indépendante par échantillon
        X_samp = X_norm.reshape(N, C, H * W).copy()
        for i in range(N):
            p_i = rng.permutation(H * W)
            X_samp[i] = X_samp[i][:, p_i]
        X_samp = X_samp.reshape(N, C, H, W)
        with torch.no_grad():
            preds_s = []
            for start in range(0, N, 64):
                xb = torch.from_numpy(X_samp[start:start + 64]).to(device)
                preds_s.extend(model(xb).argmax(dim=1).cpu().numpy())
            per_sample_accs.append(float(accuracy_score(y_test, preds_s)))

    return {
        "fixed_perm_acc_mean": float(np.mean(fixed_accs)),
        "fixed_perm_acc_std": float(np.std(fixed_accs)),
        "per_sample_perm_acc_mean": float(np.mean(per_sample_accs)),
        "per_sample_perm_acc_std": float(np.std(per_sample_accs)),
        "seeds": list(seeds),
    }


def train_original_cnn_experiment(
    name: str,
    X_train, y_train, X_val, y_val, X_test, y_test,
    save_ckpt_path: str,
    seed: int = 42,
    epochs: int = 25,
    batch_size: int = 32,
    lr: float = 0.002,
    weight_decay: float = 1e-3,
    patience: int = 7,
):
    """Entraîne et évalue exactement l'architecture DatacenterThermalCNN originale."""
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    mean = X_train.mean(axis=(0, 2, 3), keepdims=True)
    std = X_train.std(axis=(0, 2, 3), keepdims=True) + 1e-7
    norm_stats = {"mean": mean, "std": std}

    tr_ds = DatacenterThermalDataset(X_train, y_train, mean=mean, std=std, augment=True)
    tr_eval_ds = DatacenterThermalDataset(X_train, y_train, mean=mean, std=std, augment=False)
    va_ds = DatacenterThermalDataset(X_val, y_val, mean=mean, std=std, augment=False)
    te_ds = DatacenterThermalDataset(X_test, y_test, mean=mean, std=std, augment=False)

    g = torch.Generator().manual_seed(seed)
    tr_loader = DataLoader(tr_ds, batch_size=batch_size, shuffle=True, generator=g)
    tr_eval_loader = DataLoader(tr_eval_ds, batch_size=batch_size, shuffle=False)
    va_loader = DataLoader(va_ds, batch_size=batch_size, shuffle=False)
    te_loader = DataLoader(te_ds, batch_size=batch_size, shuffle=False)

    model = DatacenterThermalCNN(in_channels=2, num_classes=4).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_val_loss = float("inf")
    best_val_acc = 0.0
    best_epoch = 0
    best_state = None
    no_improve = 0

    t0 = time.time()
    for ep in range(1, epochs + 1):
        tr_l, tr_a = train_one_epoch(model, tr_loader, criterion, optimizer, device)
        va_l, va_a = evaluate_split(model, va_loader, criterion, device)
        scheduler.step()
        history["train_loss"].append(float(tr_l))
        history["train_acc"].append(float(tr_a))
        history["val_loss"].append(float(va_l))
        history["val_acc"].append(float(va_a))

        if va_l < best_val_loss or (va_l <= best_val_loss and va_a >= best_val_acc):
            best_val_loss = va_l
            best_val_acc = va_a
            best_epoch = ep
            no_improve = 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            no_improve += 1

        if ep % 5 == 0 or ep == 1:
            print(f"  [{name}] Ep {ep:02d}/{epochs:02d} | Tr L={tr_l:.4f} A={tr_a*100:.2f}% | Va L={va_l:.4f} A={va_a*100:.2f}%", flush=True)

        if no_improve >= patience:
            print(f"  [{name}] Early Stopping à l'époque {ep} (meilleure={best_epoch}).", flush=True)
            break

    elapsed = time.time() - t0
    model.load_state_dict(best_state)
    torch.save({
        "name": name,
        "model_state_dict": best_state,
        "norm_stats": norm_stats,
        "class_names": CLASS_NAMES,
        "best_epoch": best_epoch,
        "history": history,
    }, save_ckpt_path)

    tr_loss, tr_acc, _, _, _ = _run_eval_loader(model, tr_eval_loader, criterion, device)
    va_loss, va_acc, _, _, _ = _run_eval_loader(model, va_loader, criterion, device)
    te_loss, te_acc, te_targets, te_preds, _ = _run_eval_loader(model, te_loader, criterion, device)

    prec_m, rec_m, f1_m, _ = precision_recall_fscore_support(te_targets, te_preds, average="macro")
    cm = confusion_matrix(te_targets, te_preds)
    perm_res = run_spatial_permutation_test(model, X_test, y_test, norm_stats, device)

    return {
        "experiment": name,
        "num_parameters": int(model.count_parameters()),
        "best_epoch": int(best_epoch),
        "elapsed_sec": float(elapsed),
        "train_loss": float(tr_loss),
        "val_loss": float(va_loss),
        "test_loss": float(te_loss),
        "train_accuracy": float(tr_acc),
        "val_accuracy": float(va_acc),
        "test_accuracy": float(te_acc),
        "macro_precision": float(prec_m),
        "macro_recall": float(rec_m),
        "macro_f1": float(f1_m),
        "confusion_matrix": cm.tolist(),
        "classification_report": classification_report(te_targets, te_preds, target_names=CLASS_NAMES, digits=4, output_dict=True),
        "spatial_permutation_test": perm_res,
        "history": history,
    }


def run_classical_baselines(name_prefix: str, X_train, y_train, X_val, y_val, X_test, y_test):
    """
    Évalue Régression Logistique et Arbre de Décision avec sélection d'hyperparamètres
    strictement sur la validation (jamais sur le test set).
    """
    mean = X_train.mean(axis=(0, 2, 3), keepdims=True)
    std = X_train.std(axis=(0, 2, 3), keepdims=True) + 1e-7

    X_tr_n = ((X_train - mean) / std).reshape(len(X_train), -1)
    X_va_n = ((X_val - mean) / std).reshape(len(X_val), -1)
    X_te_n = ((X_test - mean) / std).reshape(len(X_test), -1)

    # 1. Logistic Regression (sélection de C sur Validation uniquement)
    best_lr, best_lr_val_loss, best_c = None, float("inf"), None
    for c_val in [0.01, 0.05, 0.1, 0.5, 1.0, 5.0]:
        clf = LogisticRegression(C=c_val, max_iter=1500, random_state=42)
        clf.fit(X_tr_n, y_train)
        v_loss = log_loss(y_val, clf.predict_proba(X_va_n))
        if v_loss < best_lr_val_loss:
            best_lr_val_loss = v_loss
            best_lr = clf
            best_c = c_val

    lr_tr_acc = accuracy_score(y_train, best_lr.predict(X_tr_n))
    lr_va_acc = accuracy_score(y_val, best_lr.predict(X_va_n))
    lr_te_preds = best_lr.predict(X_te_n)
    lr_te_acc = accuracy_score(y_test, lr_te_preds)
    lr_tr_loss = log_loss(y_train, best_lr.predict_proba(X_tr_n))
    lr_te_loss = log_loss(y_test, best_lr.predict_proba(X_te_n))
    p_m, r_m, f1_m, _ = precision_recall_fscore_support(y_test, lr_te_preds, average="macro")

    lr_res = {
        "experiment": f"{name_prefix} - Logistic Regression (C={best_c})",
        "best_hyperparams": {"C": best_c},
        "train_loss": float(lr_tr_loss),
        "val_loss": float(best_lr_val_loss),
        "test_loss": float(lr_te_loss),
        "train_accuracy": float(lr_tr_acc),
        "val_accuracy": float(lr_va_acc),
        "test_accuracy": float(lr_te_acc),
        "macro_precision": float(p_m),
        "macro_recall": float(r_m),
        "macro_f1": float(f1_m),
        "confusion_matrix": confusion_matrix(y_test, lr_te_preds).tolist(),
    }

    # 2. Decision Tree (sélection max_depth & min_samples_leaf sur Validation uniquement)
    best_dt, best_dt_val_acc, best_dt_cfg = None, -1.0, None
    for depth in [4, 6, 8, 10, 14, None]:
        for msl in [1, 5, 10, 20]:
            clf = DecisionTreeClassifier(max_depth=depth, min_samples_leaf=msl, random_state=42)
            clf.fit(X_tr_n, y_train)
            v_acc = accuracy_score(y_val, clf.predict(X_va_n))
            if v_acc > best_dt_val_acc:
                best_dt_val_acc = v_acc
                best_dt = clf
                best_dt_cfg = {"max_depth": depth, "min_samples_leaf": msl}

    dt_tr_acc = accuracy_score(y_train, best_dt.predict(X_tr_n))
    dt_te_preds = best_dt.predict(X_te_n)
    dt_te_acc = accuracy_score(y_test, dt_te_preds)
    dt_tr_loss = log_loss(y_train, best_dt.predict_proba(X_tr_n))
    dt_va_loss = log_loss(y_val, best_dt.predict_proba(X_va_n))
    dt_te_loss = log_loss(y_test, best_dt.predict_proba(X_te_n))
    p_m2, r_m2, f1_m2, _ = precision_recall_fscore_support(y_test, dt_te_preds, average="macro")

    dt_res = {
        "experiment": f"{name_prefix} - Decision Tree (depth={best_dt_cfg['max_depth']}, leaf={best_dt_cfg['min_samples_leaf']})",
        "best_hyperparams": best_dt_cfg,
        "train_loss": float(dt_tr_loss),
        "val_loss": float(dt_va_loss),
        "test_loss": float(dt_te_loss),
        "train_accuracy": float(dt_tr_acc),
        "val_accuracy": float(best_dt_val_acc),
        "test_accuracy": float(dt_te_acc),
        "macro_precision": float(p_m2),
        "macro_recall": float(r_m2),
        "macro_f1": float(f1_m2),
        "confusion_matrix": confusion_matrix(y_test, dt_te_preds).tolist(),
    }

    return lr_res, dt_res


def run_summary_stats_baselines(name_prefix: str, X_train, y_train, X_val, y_val, X_test, y_test):
    """Baselines sur 7 statistiques scalaires globales (sans information spatiale 2D)."""
    def featurize(X):
        t = X[:, 0]
        h = X[:, 1]
        return np.column_stack([
            t.mean(axis=(1, 2)), t.max(axis=(1, 2)), t.min(axis=(1, 2)), t.std(axis=(1, 2)),
            h.mean(axis=(1, 2)), h.min(axis=(1, 2)), h.max(axis=(1, 2)),
        ])

    F_tr, F_va, F_te = featurize(X_train), featurize(X_val), featurize(X_test)
    mu = F_tr.mean(axis=0, keepdims=True)
    sd = F_tr.std(axis=0, keepdims=True) + 1e-7
    F_tr_n, F_va_n, F_te_n = (F_tr - mu) / sd, (F_va - mu) / sd, (F_te - mu) / sd

    best_lr, best_vloss, best_c = None, float("inf"), None
    for c_val in [0.1, 1.0, 10.0]:
        clf = LogisticRegression(C=c_val, max_iter=1000, random_state=42).fit(F_tr_n, y_train)
        vl = log_loss(y_val, clf.predict_proba(F_va_n))
        if vl < best_vloss:
            best_vloss, best_lr, best_c = vl, clf, c_val

    best_dt, best_vacc, best_d = None, -1.0, None
    for d in [3, 5, 7, 10]:
        clf = DecisionTreeClassifier(max_depth=d, min_samples_leaf=5, random_state=42).fit(F_tr_n, y_train)
        va = accuracy_score(y_val, clf.predict(F_va_n))
        if va > best_vacc:
            best_vacc, best_dt, best_d = va, clf, d

    return {
        f"{name_prefix}_stats_logreg": {
            "C": best_c,
            "val_acc": float(accuracy_score(y_val, best_lr.predict(F_va_n))),
            "test_acc": float(accuracy_score(y_test, best_lr.predict(F_te_n))),
        },
        f"{name_prefix}_stats_dt": {
            "max_depth": best_d,
            "val_acc": float(best_vacc),
            "test_acc": float(accuracy_score(y_test, best_dt.predict(F_te_n))),
        },
    }


def plot_comparative_charts(exp1, exp2, exp3, lr_imp, dt_imp, graphs_dir, out_root):
    os.makedirs(graphs_dir, exist_ok=True)

    # 1. Comparaison des courbes d'apprentissage (Exp1 vs Exp2 vs Exp3)
    fig, axes = plt.subplots(2, 3, figsize=(16, 8.5))
    items = [
        ("Exp 1 : Original Data + Original CNN", exp1["history"]),
        ("Exp 2 : Improved Data + Original CNN", exp2["history"]),
        ("Exp 3 : Improved Data + Optimized CNN", exp3["history"]),
    ]
    for col, (title, h) in enumerate(items):
        ep = range(1, len(h["train_loss"]) + 1)
        axes[0, col].plot(ep, h["train_loss"], "o-", label="Train Loss", lw=1.8, markersize=3.5)
        axes[0, col].plot(ep, h["val_loss"], "s-", label="Val Loss", lw=1.8, markersize=3.5)
        axes[0, col].set_title(title, fontsize=10.5, fontweight="bold")
        axes[0, col].set_xlabel("Époque")
        axes[0, col].set_ylabel("Loss")
        axes[0, col].grid(True, alpha=0.3, linestyle="--")
        axes[0, col].legend(fontsize=8.5)

        axes[1, col].plot(ep, [a * 100 for a in h["train_acc"]], "o-", label="Train Acc", lw=1.8, markersize=3.5)
        axes[1, col].plot(ep, [a * 100 for a in h["val_acc"]], "s-", label="Val Acc", lw=1.8, markersize=3.5)
        axes[1, col].set_xlabel("Époque")
        axes[1, col].set_ylabel("Accuracy (%)")
        axes[1, col].set_ylim(50, 102)
        axes[1, col].grid(True, alpha=0.3, linestyle="--")
        axes[1, col].legend(loc="lower right", fontsize=8.5)

    plt.suptitle("Comparaison des Courbes d'Apprentissage (Original vs Amélioré)", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(graphs_dir, "original_vs_improved_learning_curves.png"), dpi=300, bbox_inches="tight")
    plt.close()

    # 2. Bar chart comparaison des modèles sur Dataset Amélioré + Permutation Spatiale
    labels = [
        "Original Data\n+ Orig CNN",
        "Improved Data\n+ Orig CNN",
        "Improved Data\n+ Optim CNN",
        "Improved Data\n+ LogReg",
        "Improved Data\n+ DecisionTree",
        "Spatial Perm\n(Optim CNN)",
    ]
    test_accs = [
        exp1["test_accuracy"] * 100,
        exp2["test_accuracy"] * 100,
        exp3["test_accuracy"] * 100,
        lr_imp["test_accuracy"] * 100,
        dt_imp["test_accuracy"] * 100,
        exp3["spatial_permutation_test"]["fixed_perm_acc_mean"] * 100,
    ]
    colors = ["#8c8c8c", "#4c72b0", "#2ca02c", "#dd8452", "#c44e52", "#9467bd"]

    plt.figure(figsize=(11, 5.5))
    bars = plt.bar(labels, test_accs, color=colors, edgecolor="#222222", alpha=0.9, width=0.6)
    plt.axhspan(90.0, 93.0, color="#2ca02c", alpha=0.08, label="Zone cible réaliste (90–93%)")
    plt.ylabel("Test Accuracy (%)", fontsize=11)
    plt.title("Comparaison Scientifique des Modèles et Test de Permutation Spatiale", fontsize=12.5, fontweight="bold")
    plt.ylim(0, 108)
    plt.grid(axis="y", alpha=0.3, linestyle="--")
    for bar, val in zip(bars, test_accs):
        plt.text(bar.get_x() + bar.get_width() / 2.0, val + 1.4, f"{val:.2f}%", ha="center", fontweight="bold", fontsize=9.5)
    plt.legend(loc="lower left")
    plt.tight_layout()
    plt.savefig(os.path.join(graphs_dir, "model_comparison.png"), dpi=300, bbox_inches="tight")
    plt.close()

    # 3. Matrices de confusion côte à côte (Exp 2 Orig CNN vs Exp 3 Optim CNN sur Dataset Amélioré)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    for ax, cm_list, ttl in [
        (ax1, exp2["confusion_matrix"], f"CNN Original sur Dataset Amélioré ({exp2['test_accuracy']*100:.2f}%)"),
        (ax2, exp3["confusion_matrix"], f"CNN Optimisé sur Dataset Amélioré ({exp3['test_accuracy']*100:.2f}%)"),
    ]:
        cm = np.array(cm_list)
        cm_n = cm.astype(float) / cm.sum(axis=1)[:, None]
        annot = np.array([[f"{cm_n[i, j]*100:.1f}%\n({cm[i, j]})" for j in range(4)] for i in range(4)])
        sns.heatmap(cm_n, annot=annot, fmt="", cmap="Blues", xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, ax=ax, cbar=False)
        ax.set_title(ttl, fontsize=11, fontweight="bold")
        ax.set_xlabel("Classe Prédite")
        ax.set_ylabel("Vraie Classe")
        ax.tick_params(axis="x", rotation=20)
    plt.tight_layout()
    plt.savefig(os.path.join(out_root, "confusion_matrix_comparison.png"), dpi=300, bbox_inches="tight")
    plt.close()


def main():
    project_root = os.path.dirname(os.path.abspath(__file__))
    orig_data_path = os.path.join(project_root, "dataset", "datacenter_thermal_maps.npz")
    imp_data_path = os.path.join(project_root, "dataset", "improved_thermal_maps.npz")
    saved_dir = os.path.join(project_root, "saved_model_improved")
    results_dir = os.path.join(project_root, "results_improved")
    graphs_dir = os.path.join(results_dir, "graphs")
    metrics_dir = os.path.join(results_dir, "metrics")
    os.makedirs(saved_dir, exist_ok=True)
    os.makedirs(graphs_dir, exist_ok=True)
    os.makedirs(metrics_dir, exist_ok=True)

    print("=" * 72, flush=True)
    print("   ÉTAPE 1 : AUDIT DES DATASETS (ORIGINAL vs AMÉLIORÉ)", flush=True)
    print("=" * 72, flush=True)
    audit_orig = audit_original_dataset(orig_data_path)
    audit_imp = audit_improved_dataset(imp_data_path)
    print("Original unique samples :", audit_orig["exact_unique_samples"], "/", audit_orig["total_samples"], flush=True)
    print("Improved unique samples :", audit_imp["independence"]["exact_unique_samples"], "/", audit_imp["independence"]["total_samples"], flush=True)

    # Chargement des splits Original
    loaded_orig = np.load(orig_data_path)
    X_tv_o, X_te_o, y_tv_o, y_te_o = train_test_split(
        loaded_orig["maps"], loaded_orig["labels"], test_size=0.15, stratify=loaded_orig["labels"], random_state=42
    )
    X_tr_o, X_va_o, y_tr_o, y_va_o = train_test_split(
        X_tv_o, y_tv_o, test_size=0.15 / 0.85, stratify=y_tv_o, random_state=42
    )

    # Chargement des splits Améliorés
    (X_tr_i, y_tr_i), (X_va_i, y_va_i), (X_te_i, y_te_i), _, _ = load_improved_splits(imp_data_path)

    print("\n" + "=" * 72, flush=True)
    print("   EXPÉRIENCE 1 : DATASET ORIGINAL + CNN ORIGINAL", flush=True)
    print("=" * 72, flush=True)
    exp1 = train_original_cnn_experiment(
        "Exp1_OriginalData_OriginalCNN",
        X_tr_o, y_tr_o, X_va_o, y_va_o, X_te_o, y_te_o,
        os.path.join(saved_dir, "exp1_orig_data_orig_cnn.pth"),
        seed=42, epochs=25, patience=7,
    )
    print(f"=> Exp1 Test Acc: {exp1['test_accuracy']*100:.2f}% | Perm Acc: {exp1['spatial_permutation_test']['fixed_perm_acc_mean']*100:.2f}%", flush=True)

    print("\n" + "=" * 72, flush=True)
    print("   EXPÉRIENCE 2 : DATASET AMÉLIORÉ + CNN ORIGINAL", flush=True)
    print("=" * 72, flush=True)
    exp2 = train_original_cnn_experiment(
        "Exp2_ImprovedData_OriginalCNN",
        X_tr_i, y_tr_i, X_va_i, y_va_i, X_te_i, y_te_i,
        os.path.join(saved_dir, "exp2_imp_data_orig_cnn.pth"),
        seed=42, epochs=25, patience=7,
    )
    print(f"=> Exp2 Test Acc: {exp2['test_accuracy']*100:.2f}% | Perm Acc: {exp2['spatial_permutation_test']['fixed_perm_acc_mean']*100:.2f}%", flush=True)

    print("\n" + "=" * 72, flush=True)
    print("   EXPÉRIENCE 3 : DATASET AMÉLIORÉ + CNN OPTIMISÉ (ABLATIONS & SÉLECTION VAL)", flush=True)
    print("=" * 72, flush=True)
    best_opt_path, best_opt_val_res, ablation_summary = run_validation_optimization(data_path=imp_data_path, seed=42)
    exp3_eval = evaluate_improved_model(model_path=best_opt_path, data_path=imp_data_path, batch_size=32)

    # Spatial permutation test sur le CNN optimisé sélectionné
    from cnn_model_improved import OptimizedThermalCNN
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(best_opt_path, map_location=device, weights_only=False)
    opt_model = OptimizedThermalCNN(**ckpt["model_config"]).to(device)
    opt_model.load_state_dict(ckpt["model_state_dict"])
    exp3_perm = run_spatial_permutation_test(opt_model, X_te_i, y_te_i, ckpt["norm_stats"], device)

    exp3 = {
        "experiment": f"Exp3_ImprovedData_OptimizedCNN ({exp3_eval['model_name']})",
        "num_parameters": exp3_eval["num_parameters"],
        "best_epoch": exp3_eval["best_epoch"],
        "train_loss": exp3_eval["train_loss"],
        "val_loss": exp3_eval["val_loss"],
        "test_loss": exp3_eval["test_loss"],
        "train_accuracy": exp3_eval["train_accuracy"],
        "val_accuracy": exp3_eval["val_accuracy"],
        "test_accuracy": exp3_eval["test_accuracy"],
        "macro_precision": exp3_eval["macro_precision"],
        "macro_recall": exp3_eval["macro_recall"],
        "macro_f1": exp3_eval["macro_f1_score"],
        "confusion_matrix": exp3_eval["confusion_matrix"],
        "classification_report": exp3_eval["classification_report"],
        "spatial_permutation_test": exp3_perm,
        "history": best_opt_val_res["history"],
    }
    print(f"=> Exp3 Perm Acc (fixed): {exp3_perm['fixed_perm_acc_mean']*100:.2f}% ± {exp3_perm['fixed_perm_acc_std']*100:.2f}% | (per-sample): {exp3_perm['per_sample_perm_acc_mean']*100:.2f}%", flush=True)

    print("\n" + "=" * 72, flush=True)
    print("   EXPÉRIENCES 4 & 5 : BASELINES CLASSIQUES (LOGREG & DECISION TREE)", flush=True)
    print("=" * 72, flush=True)
    lr_orig, dt_orig = run_classical_baselines("OriginalData", X_tr_o, y_tr_o, X_va_o, y_va_o, X_te_o, y_te_o)
    lr_imp, dt_imp = run_classical_baselines("ImprovedData", X_tr_i, y_tr_i, X_va_i, y_va_i, X_te_i, y_te_i)
    stats_orig = run_summary_stats_baselines("OriginalData", X_tr_o, y_tr_o, X_va_o, y_va_o, X_te_o, y_te_o)
    stats_imp = run_summary_stats_baselines("ImprovedData", X_tr_i, y_tr_i, X_va_i, y_va_i, X_te_i, y_te_i)
    for b in [lr_orig, dt_orig, lr_imp, dt_imp]:
        print(f"  {b['experiment']} -> Train={b['train_accuracy']*100:.2f}% | Val={b['val_accuracy']*100:.2f}% | Test={b['test_accuracy']*100:.2f}% | Macro F1={b['macro_f1']*100:.2f}%", flush=True)
    print("  Summary-Stats Baselines Orig:", stats_orig, flush=True)
    print("  Summary-Stats Baselines Imp :", stats_imp, flush=True)

    # Graphiques comparatifs
    plot_comparative_charts(exp1, exp2, exp3, lr_imp, dt_imp, graphs_dir, results_dir)

    # Tableau CSV comparatif
    table_rows = []
    for row in [exp1, exp2, exp3, lr_orig, dt_orig, lr_imp, dt_imp]:
        perm = row.get("spatial_permutation_test")
        table_rows.append({
            "experiment": row["experiment"],
            "train_loss": row["train_loss"],
            "val_loss": row["val_loss"],
            "test_loss": row["test_loss"],
            "train_accuracy": row["train_accuracy"],
            "val_accuracy": row["val_accuracy"],
            "test_accuracy": row["test_accuracy"],
            "macro_precision": row["macro_precision"],
            "macro_recall": row["macro_recall"],
            "macro_f1": row["macro_f1"],
            "spatial_perm_fixed_acc": perm["fixed_perm_acc_mean"] if perm else None,
            "spatial_perm_per_sample_acc": perm["per_sample_perm_acc_mean"] if perm else None,
        })
    pd.DataFrame(table_rows).to_csv(os.path.join(metrics_dir, "experiments_comparison.csv"), index=False)

    full_summary = {
        "dataset_audit": {"original": audit_orig, "improved": audit_imp},
        "ablation_study_validation": ablation_summary,
        "experiments": {
            "exp1_original_data_original_cnn": {k: v for k, v in exp1.items() if k != "history"},
            "exp2_improved_data_original_cnn": {k: v for k, v in exp2.items() if k != "history"},
            "exp3_improved_data_optimized_cnn": {k: v for k, v in exp3.items() if k != "history"},
            "exp4_original_logreg": lr_orig,
            "exp4_original_decision_tree": dt_orig,
            "exp5_improved_logreg": lr_imp,
            "exp5_improved_decision_tree": dt_imp,
            "summary_stats_baselines": {**stats_orig, **stats_imp},
        },
    }
    with open(os.path.join(metrics_dir, "experiments_summary.json"), "w", encoding="utf-8") as f:
        json.dump(full_summary, f, indent=4)

    print("\n" + "=" * 72, flush=True)
    print("   PIPELINE EXPÉRIMENTAL COMPLET TERMINÉ AVEC SUCCÈS", flush=True)
    print("=" * 72, flush=True)


if __name__ == "__main__":
    main()
