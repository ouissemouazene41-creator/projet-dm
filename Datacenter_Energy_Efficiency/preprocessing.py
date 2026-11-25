"""
Projet : Efficacité Énergétique des Centres de Données (Datacenters)
Fichier : preprocessing.py
Description : Chargement, inspection et prétraitement rigoureux du dataset réel.

Règles méthodologiques strictes :
  - Utilise EXCLUSIVEMENT le dataset existant (dataset/datacenter_thermal_maps.npz).
  - Aucune génération de données artificielles.
  - Séparation stratifiée stricte en 3 sous-ensembles indépendants :
      * Entraînement  : 70% (2 100 échantillons)
      * Validation    : 15% (450 échantillons)
      * Test (Inédit) : 15% (450 échantillons)
  - Zéro fuite de données (No Data Leakage) : la moyenne et l'écart-type de
    normalisation Z-score par canal sont calculés UNIQUEMENT sur l'ensemble
    d'entraînement, puis appliqués à l'identique sur la validation et le test.
"""

import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

# Noms normalisés des 4 classes ASHRAE TC 9.9
CLASS_NAMES = [
    "Optimal / Froid (<22°C)",
    "Nominal ASHRAE (22-27°C)",
    "Alerte Point Chaud (27-32°C)",
    "Surchauffe Critique (>32°C)"
]


class DatacenterThermalDataset(Dataset):
    """
    Dataset PyTorch pour les cartes thermiques 2D spatiales.
    Applique la normalisation Z-score par canal issue du Training Set.
    Supporte une augmentation de données légère et physiquement valide pour l'entraînement.
    """

    def __init__(self, data: np.ndarray, labels: np.ndarray, mean: np.ndarray, std: np.ndarray, augment: bool = False):
        """
        Args:
            data    : Tableau numpy (N, 2, 16, 16)
            labels  : Tableau numpy (N,)
            mean    : Moyenne par canal calculée sur le train set, shape (1, 2, 1, 1)
            std     : Écart-type par canal calculé sur le train set, shape (1, 2, 1, 1)
            augment : Activation de l'augmentation de données (flip horizontal symétrique)
        """
        self.labels = torch.tensor(labels, dtype=torch.long)
        self.mean = mean
        self.std = std
        self.augment = augment

        # Normalisation Z-score par canal
        normalized = (data - self.mean) / self.std
        self.data = torch.tensor(normalized, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int):
        x = self.data[idx]
        y = self.labels[idx]

        # Augmentation de données sur l'entraînement uniquement (flip horizontal aléatoire)
        # Physiquement valide car la disposition des baies en rangée conserve les lois de transfert thermique
        if self.augment and torch.rand(1).item() > 0.5:
            x = torch.flip(x, dims=[-1])

        return x, y


def load_and_preprocess_data(
    data_path: str = None,
    batch_size: int = 32,
    test_size: float = 0.15,
    val_size: float = 0.15,
    random_state: int = 42,
    num_workers: int = 0
):
    """
    Charge le dataset existant, réalise la division stratifiée sans fuite de données,
    calcule les statistiques de normalisation et retourne les DataLoaders PyTorch.

    Returns:
        train_loader, val_loader, test_loader : DataLoaders PyTorch
        norm_stats : Dictionnaire {'mean': ..., 'std': ...}
        summary : Dictionnaire récapitulatif des caractéristiques du dataset
    """
    project_root = os.path.dirname(os.path.abspath(__file__))
    if data_path is None:
        data_path = os.path.join(project_root, "dataset", "datacenter_thermal_maps.npz")

    if not os.path.exists(data_path):
        raise FileNotFoundError(
            f"Erreur : Le fichier dataset est introuvable au chemin : {data_path}\n"
            f"Veuillez vous assurer que 'dataset/datacenter_thermal_maps.npz' est présent."
        )

    # 1. Chargement des matrices brutes
    loaded = np.load(data_path)
    if "maps" not in loaded or "labels" not in loaded:
        raise KeyError(f"Le fichier {data_path} doit contenir les clés 'maps' et 'labels'.")

    maps = loaded["maps"]      # Shape: (N, 2, 16, 16)
    labels = loaded["labels"]  # Shape: (N,)

    num_samples = len(labels)
    unique_classes, class_counts = np.unique(labels, return_counts=True)

    # 2. Division stratifiée Train+Val / Test (15% pour le test indépendant)
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        maps, labels,
        test_size=test_size,
        stratify=labels,
        random_state=random_state
    )

    # 3. Division stratifiée Train / Validation (15% du total)
    val_ratio_adjusted = val_size / (1.0 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val,
        test_size=val_ratio_adjusted,
        stratify=y_train_val,
        random_state=random_state
    )

    # 4. Calcul des statistiques de normalisation STRICTEMENT sur le Train Set
    train_mean = X_train.mean(axis=(0, 2, 3), keepdims=True)
    train_std = X_train.std(axis=(0, 2, 3), keepdims=True) + 1e-7

    # 5. Instanciation des Datasets PyTorch
    train_dataset = DatacenterThermalDataset(X_train, y_train, mean=train_mean, std=train_std, augment=True)
    val_dataset = DatacenterThermalDataset(X_val, y_val, mean=train_mean, std=train_std, augment=False)
    test_dataset = DatacenterThermalDataset(X_test, y_test, mean=train_mean, std=train_std, augment=False)

    # 6. Instanciation des DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    norm_stats = {
        "mean": train_mean,
        "std": train_std
    }

    summary = {
        "total_samples": int(num_samples),
        "input_shape": list(maps.shape[1:]),
        "num_classes": len(unique_classes),
        "class_distribution": {int(c): int(count) for c, count in zip(unique_classes, class_counts)},
        "train_size": len(train_dataset),
        "val_size": len(val_dataset),
        "test_size": len(test_dataset),
        "channel_0_temp_mean": float(train_mean[0, 0, 0, 0]),
        "channel_0_temp_std": float(train_std[0, 0, 0, 0]),
        "channel_1_hum_mean": float(train_mean[0, 1, 0, 0]),
        "channel_1_hum_std": float(train_std[0, 1, 0, 0]),
    }

    return train_loader, val_loader, test_loader, norm_stats, summary


if __name__ == "__main__":
    train_loader, val_loader, test_loader, stats, summary = load_and_preprocess_data()
    print("=" * 65)
    print("   Inspection et Prétraitement du Dataset Réel")
    print("=" * 65)
    print(f"Échantillons totaux        : {summary['total_samples']}")
    print(f"Forme d'un échantillon     : {summary['input_shape']} (Canaux, Hauteur, Largeur)")
    print(f"Répartition par classe     : {summary['class_distribution']}")
    print(f"  - Ensemble d'entraînement: {summary['train_size']} ({summary['train_size']/summary['total_samples']*100:.1f}%)")
    print(f"  - Ensemble de validation : {summary['val_size']} ({summary['val_size']/summary['total_samples']*100:.1f}%)")
    print(f"  - Ensemble de test       : {summary['test_size']} ({summary['test_size']/summary['total_samples']*100:.1f}%)")
    print(f"Statistiques Normalisation :")
    print(f"  - Canal 0 (Température)  : Moyenne={summary['channel_0_temp_mean']:.2f}°C, Écart-type={summary['channel_0_temp_std']:.2f}°C")
    print(f"  - Canal 1 (Humidité)     : Moyenne={summary['channel_1_hum_mean']:.2f}%, Écart-type={summary['channel_1_hum_std']:.2f}%")
    print("=" * 65)
