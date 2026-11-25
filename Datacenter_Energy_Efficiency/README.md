# Datacenter Energy Efficiency — CNN Thermal Classification

> **Projet scientifique** : Classification de l'état thermique d'un datacenter à partir de cartes de chaleur 2D (température + humidité relative), par réseau de neurones convolutif (CNN) optimisé.

---

## 🗂️ Structure du projet

```
Datacenter_Energy_Efficiency/
├── dataset/
│   ├── datacenter_thermal_maps.npz       ← Dataset original (3 000 échantillons)
│   ├── improved_thermal_maps.npz         ← Dataset amélioré (3 600 échantillons)
│   └── improved_metadata.csv             ← Métadonnées scénarios physiques
│
├── backup_original/                      ← Sauvegarde avant modifications
│   ├── cnn_model.py, train.py, ...
│   └── results/metrics/test_metrics.json ← 100 % accuracy originale
│
├── results/                              ← Résultats dataset original
├── results_improved/                     ← Résultats dataset amélioré ✅
│   ├── confusion_matrix_optimized.png
│   ├── confusion_matrix_comparison.png
│   ├── graphs/
│   │   ├── optimized_training_curves.png
│   │   ├── ablation_study.png
│   │   ├── model_comparison.png
│   │   └── original_vs_improved_learning_curves.png
│   └── metrics/
│       ├── test_metrics.json             ← Métriques finales
│       ├── experiments_summary.json      ← Toutes expériences
│       ├── experiments_comparison.csv
│       ├── ablation_validation_study.json
│       ├── classification_report.txt
│       ├── roc_auc.png                   ← Courbes ROC-AUC
│       └── roc_auc_results.json
│
├── saved_model_improved/
│   ├── best_optimized_thermal_cnn.pth    ← Meilleur modèle (opt_plateau_wd2e3)
│   ├── exp1_orig_data_orig_cnn.pth
│   └── exp2_imp_data_orig_cnn.pth
│
├── cnn_model.py                          ← Architecture CNN originale
├── cnn_model_improved.py                 ← OptimizedThermalCNN (ResBlock + DualPool)
├── generate_improved_dataset.py          ← Générateur physique v3
├── preprocessing.py / preprocessing_improved.py
├── train.py / train_improved.py
├── evaluate.py / evaluate_improved.py
├── run_experiments.py                    ← Pipeline complet
├── generate_roc_auc.py                   ← Courbes ROC-AUC
└── requirements.txt
```

---

## 🏷️ Classes thermiques

| ID | Nom | Plage de température | Interprétation |
|----|-----|----------------------|----------------|
| 0 | **Optimal / Froid** | < 22 °C | Surrefroidissement, surconsommation électrique |
| 1 | **Nominal ASHRAE** | 22 – 27 °C | Zone de fonctionnement optimale |
| 2 | **Alerte Point Chaud** | 27 – 32 °C | Hotspot localisé, risque de throttling |
| 3 | **Surchauffe Critique** | > 32 °C | Urgence : déclenchement alarme, coupure possible |

---

## ⚠️ Problème du dataset original — 100 % d'accuracy

L'audit scientifique a révélé que le dataset original (`datacenter_thermal_maps.npz`) produit **100 % d'accuracy** sur toutes les expériences, y compris avec une Régression Logistique sur 7 statistiques résumées et un arbre de décision de profondeur 3.

**Cause** : les plages de température moyennes par classe ne se chevauchent **jamais** :
- Classe 0 : moyenne 17–20 °C, max ≤ 21,5 °C
- Classe 1 : moyenne 22–25 °C, min ≥ 18,9 °C  
- Classe 2 : moyenne 23–27 °C
- Classe 3 : moyenne 29–35 °C

Le test de permutation spatiale atteint encore **93,07 %** (vs 100 % normal), confirmant que le CNN exploite des statistiques globales, pas des patterns spatiaux.

---

## 🔬 Dataset amélioré (v3 — physique)

Généré par `generate_improved_dataset.py` avec couplage physique réaliste :

| Propriété | Valeur |
|-----------|--------|
| Échantillons total | 3 600 (2 400 train / 600 val / 600 test) |
| Splits indépendants | Seeds 1001 / 2002 / 3003 |
| Canaux | 2 (température + humidité psychrométrique couplée) |
| Résolution | 16 × 16 pixels |
| Échantillons uniques | 3 600 / 3 600 ✅ |
| NN L2 test→train (min) | 9,60 (très bien séparés) |
| NN L2 test→train (médiane) | 12,17 |

**Améliorations physiques** :
- Panaches thermiques anisotropes (gaussiennes elliptiques orientées)
- Couplage psychrométrique T/RH (humidité relative décroît avec T)
- Gradients de flux d'air (entrée froide / sortie chaude)
- Dérives de capteurs et pannes simulées
- Recouvrement des classes aux frontières (27 ± 2 °C, 32 ± 2 °C)

---

## 🧠 Architecture CNN optimisée

**Modèle** : `OptimizedThermalCNN` (`cnn_model_improved.py`)

```
Input (2, 16, 16)
  → ResidualConvBlock(24, BN, Dropout2D)  → MaxPool2D (8×8)
  → ResidualConvBlock(48, BN, Dropout2D)  → MaxPool2D (4×4)
  → ResidualConvBlock(96, BN, Dropout2D)  → MaxPool2D (2×2)
  → Dual Pooling : GAP(96) + GMP(96) = 192 features
  → FC(192→128) + BN + Dropout(0.35)
  → FC(128→64)  + BN + Dropout(0.20)
  → FC(64→4)    (logits)
```

**Paramètres** : 72 612  
**Configuration sélectionnée** : `opt_plateau_wd2e3`

| Hyperparamètre | Valeur |
|----------------|--------|
| Optimiseur | Adam |
| Learning rate initial | 0.001 |
| Scheduler | ReduceLROnPlateau (patience=5, factor=0.5) |
| Weight decay | 2×10⁻³ |
| Batch size | 32 |
| Early stopping (patience) | 15 |
| Meilleure époque | 27 |
| Augmentation | Flip H/V + micro-shift + bruit Gaussien |

---

## 📊 Résultats expérimentaux

### Tableau comparatif complet

| Expérience | Train Acc | Val Acc | Test Acc | Macro F1 |
|------------|-----------|---------|----------|----------|
| Exp1 : Dataset original + CNN original | 100,00 % | 100,00 % | **100,00 %** ❌ | 100,00 % |
| Exp2 : Dataset amélioré + CNN original | 94,08 % | 89,33 % | **90,50 %** ✅ | 90,46 % |
| **Exp3 : Dataset amélioré + CNN optimisé** | **95,29 %** | **91,67 %** | **91,50 %** ✅ | **91,42 %** |
| Orig + Régression Logistique | 100,00 % | 99,78 % | 100,00 % ❌ | 100,00 % |
| Orig + Arbre de décision (d=14) | 100,00 % | 97,56 % | 95,56 % ❌ | 95,56 % |
| Amélioré + Régression Logistique | 84,29 % | 73,83 % | 77,33 % | 77,10 % |
| Amélioré + Arbre de décision (d=14) | 91,21 % | 72,33 % | 72,17 % | 72,24 % |
| Amélioré + Stats-LogReg | — | 86,33 % | 87,17 % | — |
| Amélioré + Stats-DT (d=7) | — | 82,83 % | 82,50 % | — |

### Métriques finales — Meilleur modèle (Exp3, Test Set 600 échantillons)

| Classe | Précision | Rappel | F1-Score |
|--------|-----------|--------|----------|
| Optimal / Froid (<22 °C) | 97,30 % | 96,00 % | 96,64 % |
| Nominal ASHRAE (22-27 °C) | 83,43 % | 94,00 % | 88,40 % |
| Alerte Point Chaud (27-32 °C) | 92,91 % | 78,67 % | 85,20 % |
| Surchauffe Critique (>32 °C) | 93,59 % | 97,33 % | 95,42 % |
| **Macro-moyenne** | **91,81 %** | **91,50 %** | **91,42 %** |

**Train Loss / Acc** : 0,1351 / 95,29 %  
**Val Loss / Acc** : 0,2029 / 91,67 %  
**Test Loss / Acc** : 0,2346 / 91,50 %

### Test de permutation spatiale

| Modèle | Acc normale | Acc permutée | Δ |
|--------|-------------|--------------|---|
| CNN original (dataset orig.) | 100,00 % | 93,07 % | −6,93 % |
| CNN optimisé (dataset amélioré) | 91,50 % | ≈61,57 % | **−29,93 %** ✅ |

→ Le CNN optimisé apprend **de vrais patterns spatiaux** (Δ ≈ 30 %).

---

## 🔁 Étude d'ablation (8 configurations, sélection sur val uniquement)

| Configuration | Val Acc | Early Stop Epoch |
|---------------|---------|-----------------|
| ablation_no_bn | 90,00 % | 22 |
| ablation_no_dropout | 91,17 % | 17 |
| ablation_no_weight_decay | 91,83 % | 35 |
| ablation_no_augmentation | 91,17 % | — |
| ablation_gap_only | 90,67 % | — |
| opt_cosine_wd1e3 | 91,33 % | — |
| **opt_plateau_wd2e3** ← **SÉLECTIONNÉ** | **91,67 %** | **27** |
| opt_cosine_strong_reg | 90,50 % | — |

Critère de sélection : **val loss minimale** (0,2029).

---

## ▶️ Reproduction

```bash
# 1. Installer les dépendances
pip install torch torchvision numpy pandas scikit-learn matplotlib seaborn scipy

# 2. Générer le dataset amélioré
python generate_improved_dataset.py

# 3. Lancer tous les experiments
python run_experiments.py

# 4. Évaluation finale du meilleur modèle
python evaluate_improved.py

# 5. Courbes ROC-AUC
python generate_roc_auc.py
```

**Python requis** : 3.9 ou 3.13+  
**GPU** : optionnel (CPU utilisé lors des expériences, ~40–115 s par configuration)  
**Seed** : fixé à 42 pour reproductibilité complète

---

## 🔒 Intégrité scientifique

- ✅ Dataset de test strictement non utilisé pour l'entraînement ou la sélection d'hyperparamètres  
- ✅ Normalisation calculée sur le train uniquement (appliquée au val et test)  
- ✅ Splits indépendants par seed différent (1001 / 2002 / 3003)  
- ✅ 3 600 / 3 600 échantillons uniques (aucun doublon exact)  
- ✅ Métriques issues d'une exécution réelle, non fabriquées  
- ✅ Dataset original préservé (`dataset/datacenter_thermal_maps.npz` intact)  
- ✅ Sauvegarde complète dans `backup_original/`

---

## 📋 Dépendances

```
torch>=2.0.0
numpy>=1.24
pandas>=2.0
scikit-learn>=1.3
matplotlib>=3.7
seaborn>=0.12
scipy>=1.10
```

---

*Rapport scientifique complet disponible dans le fichier artifact `rapport_scientifique_final.md`.*
