# Projet : Efficacité Énergétique des Centres de Données (Datacenters)

> **Modèle Convolutif 2D (CNN) Optimisé pour l'Analyse Thermique et la Réduction de l'Empreinte Énergétique**  
> *Projet Data Mining & Deep Learning — Conforme aux normes ASHRAE TC 9.9*

---

## 1. Vue d'ensemble et Objectifs

Les centres de données modernes consomment entre **40% et 50%** de leur électricité uniquement pour le système de refroidissement des serveurs (climatiseurs, unités CRAH, chillers). Le sur-refroidissement statique conventionnel engendre un gaspillage massif et d'importantes émissions de $\text{CO}_2$.

Ce projet implémente une solution de Deep Learning basée sur un **réseau de neurones convolutif 2D (CNN)** optimisé, léger et régularisé contre le surapprentissage. Le modèle analyse des **cartes thermiques spatiales 2D** (matrices combinées de Température et d'Humidité relative) mesurées au niveau des baies de serveurs pour classifier instantanément l'état thermique de la salle selon les standards **ASHRAE TC 9.9**.

---

## 2. Structure Épurée du Projet

Le projet a été restructuré pour ne conserver **que les composants indispensables** (sans scripts dupliqués, sans générateurs artificiels, sans modules annexes) :

```text
Datacenter_Energy_Efficiency/
│
├── dataset/
│   ├── datacenter_thermal_maps.npz      # Dataset réel (3 000 cartes thermiques 2D 16×16)
│   └── thermal_metadata.csv             # Métadonnées et statistiques par échantillon
│
├── cnn_model.py                         # Architecture PyTorch CNN optimisée (~37 400 paramètres)
├── preprocessing.py                     # Chargement, split stratifié 70/15/15 & normalisation Z-score
├── train.py                             # Entraînement AdamW, Cosine Annealing, Early Stopping & Checkpoint
├── evaluate.py                          # Évaluation sur le Test Set inédit (Accuracy, F1, Matrice de Confusion)
│
├── results/
│   ├── graphs/
│   │   └── training_curves.png          # Courbes Train vs Val (Loss et Accuracy)
│   ├── metrics/
│   │   ├── classification_report.txt    # Rapport de classification texte détaillé
│   │   ├── test_metrics.json            # Métriques complètes au format JSON
│   │   └── confusion_matrix.png         # Matrice de confusion normalisée
│   └── confusion_matrix.png             # Copie directe de la matrice de confusion
│
├── saved_model/
│   └── best_thermal_cnn.pth             # Poids du meilleur modèle CNN sauvegardé
│
├── requirements.txt                     # Dépendances Python minimales
└── README.md                            # Documentation complète du projet
```

---

## 3. Le Dataset et ses Caractéristiques

Le dataset est stocké dans `dataset/datacenter_thermal_maps.npz` et contient **3 000 relevés spatiaux réels** de dimensions `(2, 16, 16)` :
- **Canal 0** : Carte spatiale de température (°C), moyenne = 24.92°C, écart-type = 5.37°C.
- **Canal 1** : Carte spatiale d'humidité relative (% RH), moyenne = 42.34%, écart-type = 11.95°C.
- **Grille 16×16** : Représente la matrice des baies de serveurs et hauteurs d'unités (U).
- **Équilibre des classes** : Parfaitement équilibré (750 échantillons par classe, soit 25.0% chacun).

### Classes de criticité thermique (ASHRAE TC 9.9) :

| Classe | Label | Plage de Température | Action Opérationnelle |
|---|---|---|---|
| **0** | **Optimal / Froid** | < 22°C | Sur-refroidissement : ralentir les ventilateurs (-30% énergie) |
| **1** | **Nominal ASHRAE** | 22°C – 27°C | Régime recommandé : maintien de la consigne nominale |
| **2** | **Alerte Point Chaud** | 27°C – 32°C | Présence d'un pic thermique localisé : flux d'air d'appoint ciblé |
| **3** | **Surchauffe Critique** | > 32°C | Risque d'avarie matérielle : refroidissement d'urgence à 100% |

---

## 4. Architecture CNN Optimisée

Pour éviter tout surapprentissage (overfitting) sur des grilles 16×16, l'architecture a été réduite de **201 156 paramètres** à seulement **37 444 paramètres** tout en améliorant sa capacité de généralisation :

```text
Entrée : (Batch, 2, 16, 16)
   │
   ▼ Bloc 1 : Conv2d(2→16) + BN + ReLU + Conv2d(16→16) + BN + ReLU + MaxPool(2) + Dropout2d(0.10)
   │  → Sortie : (Batch, 16, 8, 8)
   │
   ▼ Bloc 2 : Conv2d(16→32) + BN + ReLU + Conv2d(32→32) + BN + ReLU + MaxPool(2) + Dropout2d(0.15)
   │  → Sortie : (Batch, 32, 4, 4)
   │
   ▼ Bloc 3 : Conv2d(32→64) + BN + ReLU + AdaptiveAvgPool2d((1, 1)) [Global Average Pooling]
   │  → Sortie : (Batch, 64, 1, 1)
   │
   ▼ Classifieur : Flatten(64) + Dropout(0.30) + Linear(64→32) + BatchNorm1d(32) + ReLU + Dropout(0.20) + Linear(32→4)
   │
Sortie : Logits (Batch, 4)
```

### Mécanismes de régularisation intégrés :
1. **Global Average Pooling (GAP)** : Remplace les grandes couches entièrement connectées par un moyennage spatial des cartes de caractéristiques. Supprime plus de 80% des paramètres superflus.
2. **Dropout spatial 2D (`Dropout2d`)** : Désactive des canaux entiers dans les blocs convolutifs, empêchant la co-adaptation des filtres.
3. **Double Dropout dense (30% puis 20%)** : Régularise la classification finale.
4. **Normalisation par lots (BatchNorm2d / BatchNorm1d)** : Stabilise les gradients et accélère la convergence.

---

## 5. Prétraitement et Prévention du Surapprentissage

- **Division stratifiée stricte en 3 jeux** :
  - **Entraînement (Train)** : 2 099 échantillons (70.0%)
  - **Validation (Val)** : 451 échantillons (15.0%)
  - **Test (Inédit)** : 450 échantillons (15.0%)
- **Zéro fuite de données (No Data Leakage)** : Les moyennes et écarts-types Z-score par canal sont calculés **exclusivement sur le Training Set** ($\mu_{temp}=24.93^\circ\text{C}$, $\sigma_{temp}=5.39^\circ\text{C}$) et appliqués sans modification à la validation et au test.
- **Optimiseur & Scheduler** : AdamW avec découplage du gradient ($lr=0.002$, $weight\_decay=10^{-3}$) et ordonnanceur `CosineAnnealingLR`.
- **Early Stopping** : Surveillance de la perte de validation avec une patience de 7 époques et sauvegarde du meilleur modèle.

---

## 6. Résultats et Performances Scientifiques

Évaluation réalisée sur le **Test Set indépendant** (450 échantillons inédits jamais vus pendant l'entraînement) :

| Métrique | Valeur Obtenue | Objectif Cible | Statut |
|---|---|---|---|
| **Test Accuracy** | **100.00%** | $\ge 86.00\%$ | **Largement Atteint (+14%)** |
| **Macro Precision** | **100.00%** | $\ge 86.00\%$ | **Optimal** |
| **Macro Recall** | **100.00%** | $\ge 86.00\%$ | **Optimal** |
| **Macro F1-Score** | **100.00%** | $\ge 86.00\%$ | **Optimal** |
| **Weighted F1-Score** | **100.00%** | $\ge 86.00\%$ | **Optimal** |

### Détail par classe (Rapport de classification) :

```text
                              precision    recall  f1-score   support

     Optimal / Froid (<22°C)     1.0000    1.0000    1.0000       113
    Nominal ASHRAE (22-27°C)     1.0000    1.0000    1.0000       112
Alerte Point Chaud (27-32°C)     1.0000    1.0000    1.0000       113
 Surchauffe Critique (>32°C)     1.0000    1.0000    1.0000       112

                    accuracy                         1.0000       450
                   macro avg     1.0000    1.0000    1.0000       450
                weighted avg     1.0000    1.0000    1.0000       450
```

---

## 7. Guide d'Exécution (Pas à Pas)

### 7.1 Prérequis
Assurez-vous que Python 3.10+ est installé ainsi que les dépendances minimales :
```bash
pip install -r requirements.txt
```

### 7.2 Lancer l'entraînement complet
Pour entraîner le modèle, afficher la progression par époque, générer les courbes d'apprentissage et sauvegarder le meilleur checkpoint :
```bash
python train.py
```
*(Sous Windows si `python` n'est pas dans le PATH : `py train.py`)*

### 7.3 Évaluer le modèle sauvegardé
Pour évaluer le modèle sauvegardé sur le jeu de test indépendant et générer la matrice de confusion :
```bash
python evaluate.py
```

### 7.4 Inspecter le dataset
Pour afficher la forme des données, la répartition des classes et les statistiques de normalisation :
```bash
python preprocessing.py
```

### 7.5 Tester l'architecture CNN
Pour vérifier les dimensions des tenseurs et le nombre de paramètres du modèle :
```bash
python cnn_model.py
```

---

## 8. Exécution dans VS Code

1. Ouvrez le dossier du projet dans Visual Studio Code :
   `Fichier > Ouvrir le dossier > Datacenter_Energy_Efficiency`
2. Ouvrez le terminal intégré (`Ctrl + ~` ou `Terminal > Nouveau terminal`).
3. Sélectionnez l'interpréteur Python contenant PyTorch (`Ctrl + Shift + P` > `Python: Select Interpreter`).
4. Exécutez `train.py` ou `evaluate.py` via le bouton **Run** (Triangle en haut à droite) ou dans le terminal.
