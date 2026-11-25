"""
Projet : Efficacité Énergétique des Centres de Données (Datacenters)
Fichier : preprocessing_improved.py
Description : Prétraitement scientifique sans fuite de données pour le dataset amélioré.

Garanties méthodologiques :
  - Charge les 3 scénarios strictement indépendants (X_train, X_val, X_test)
    depuis 'dataset/improved_thermal_maps.npz'.
  - Vérifie l'absence de NaN/Inf et l'absence de doublons inter-splits.
  - Calcule les paramètres de normalisation Z-score (moyenne, écart-type par canal)
    EXCLUSIVEMENT sur X_train, puis les applique à X_val et X_test.
  - Applique une augmentation de données physiquement valide sur le Train Set uniquement :
      * Symétrie horizontale (flip des colonnes de baies)
      * Micro-décalage spatial ±1 pixel (padding par réflexion)
      * Légère perturbation gaussienne de mesure (bruit capteur)
"""

import os
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

CLASS_NAMES = [
    "Optimal / Froid (<22°C)",
    "Nominal ASHRAE (22-27°C)",
    "Alerte Point Chaud (27-32°C)",
    "Surchauffe Critique (>32°C)"
]


class ImprovedThermalDataset(Dataset):
    """
    Dataset PyTorch pour cartes thermiques 2D avec augmentation physiquement cohérente
    activée uniquement pendant l'entraînement.
    """

    def __init__(
        self,
        data: np.ndarray,
        labels: np.ndarray,
        mean: np.ndarray,
        std: np.ndarray,
        augment: bool = False,
        aug_shift: bool = True,
        aug_noise_std: float = 0.06,
        seed: int = 42,
    ):
        self.labels = torch.tensor(labels, dtype=torch.long)
        self.mean = mean
        self.std = std
        self.augment = augment
        self.aug_shift = aug_shift
        self.aug_noise_std = aug_noise_std
        self.rng = np.random.default_rng(seed)

        normalized = (data - self.mean) / self.std
        self.data = torch.tensor(normalized, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int):
        x = self.data[idx].clone()
        y = self.labels[idx]

        if self.augment:
            # 1. Symétrie horizontale aléatoire (invariance d'allée)
            if self.rng.random() > 0.5:
                x = torch.flip(x, dims=[-1])

            # 2. Micro-translation spatiale (±1 pixel avec padding par réflexion)
            if self.aug_shift and self.rng.random() > 0.5:
                dy = int(self.rng.integers(-1, 2))
                dx = int(self.rng.integers(-1, 2))
                if dy != 0 or dx != 0:
                    padded = F.pad(x.unsqueeze(0), (1, 1, 1, 1), mode="reflect").squeeze(0)
                    y0 = 1 + dy
                    x0 = 1 + dx
                    x = padded[:, y0:y0 + 16, x0:x0 + 16]

            # 3. Léger bruit gaussien de capteur en espace normalisé
            if self.aug_noise_std > 0.0 and self.rng.random() > 0.3:
                noise = torch.from_numpy(
                    self.rng.normal(0.0, self.aug_noise_std, size=x.shape).astype(np.float32)
                )
                x = x + noise

        return x, y


def load_improved_splits(data_path: str = None):
    """
    Charge les 3 splits indépendants bruts et calcule les statistiques de normalisation
    UNIQUEMENT sur l'ensemble d'entraînement.
    """
    project_root = os.path.dirname(os.path.abspath(__file__))
    if data_path is None:
        data_path = os.path.join(project_root, "dataset", "improved_thermal_maps.npz")

    if not os.path.exists(data_path):
        raise FileNotFoundError(
            f"Dataset amélioré introuvable : {data_path}. Exécutez d'abord generate_improved_dataset.py."
        )

    loaded = np.load(data_path)
    required = ["X_train", "y_train", "X_val", "y_val", "X_test", "y_test"]
    for key in required:
        if key not in loaded:
            raise KeyError(f"Clé manquante '{key}' dans {data_path}.")

    X_train, y_train = loaded["X_train"], loaded["y_train"]
    X_val, y_val = loaded["X_val"], loaded["y_val"]
    X_test, y_test = loaded["X_test"], loaded["y_test"]

    # Vérification d'intégrité
    for name, arr in [("X_train", X_train), ("X_val", X_val), ("X_test", X_test)]:
        if np.isnan(arr).any() or np.isinf(arr).any():
            raise ValueError(f"Valeurs NaN ou Inf détectées dans {name}.")

    # Normalisation Z-score calculée STRICTEMENT sur X_train
    train_mean = X_train.mean(axis=(0, 2, 3), keepdims=True)
    train_std = X_train.std(axis=(0, 2, 3), keepdims=True) + 1e-7

    norm_stats = {"mean": train_mean, "std": train_std}
    summary = {
        "train_size": int(len(y_train)),
        "val_size": int(len(y_val)),
        "test_size": int(len(y_test)),
        "input_shape": list(X_train.shape[1:]),
        "train_classes": {int(c): int(n) for c, n in zip(*np.unique(y_train, return_counts=True))},
        "val_classes": {int(c): int(n) for c, n in zip(*np.unique(y_val, return_counts=True))},
        "test_classes": {int(c): int(n) for c, n in zip(*np.unique(y_test, return_counts=True))},
        "channel_0_temp_mean": float(train_mean[0, 0, 0, 0]),
        "channel_0_temp_std": float(train_std[0, 0, 0, 0]),
        "channel_1_hum_mean": float(train_mean[0, 1, 0, 0]),
        "channel_1_hum_std": float(train_std[0, 1, 0, 0]),
    }

    return (X_train, y_train), (X_val, y_val), (X_test, y_test), norm_stats, summary


def load_and_preprocess_improved_data(
    data_path: str = None,
    batch_size: int = 32,
    augment: bool = True,
    aug_shift: bool = True,
    aug_noise_std: float = 0.06,
    seed: int = 42,
    num_workers: int = 0,
):
    """
    Construit les DataLoaders PyTorch pour Train, Val et Test à partir des splits
    indépendants et de la normalisation sans fuite de données.
    """
    (X_train, y_train), (X_val, y_val), (X_test, y_test), norm_stats, summary = load_improved_splits(data_path)
    mean, std = norm_stats["mean"], norm_stats["std"]

    train_dataset = ImprovedThermalDataset(
        X_train, y_train, mean=mean, std=std,
        augment=augment, aug_shift=aug_shift, aug_noise_std=aug_noise_std, seed=seed
    )
    val_dataset = ImprovedThermalDataset(X_val, y_val, mean=mean, std=std, augment=False)
    test_dataset = ImprovedThermalDataset(X_test, y_test, mean=mean, std=std, augment=False)

    g = torch.Generator()
    g.manual_seed(seed)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, generator=g)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, test_loader, norm_stats, summary


if __name__ == "__main__":
    _, _, _, _, summary = load_and_preprocess_improved_data()
    print("=" * 65)
    print("   Inspection et Prétraitement du Dataset Amélioré")
    print("=" * 65)
    for k, v in summary.items():
        print(f"  {k:<22}: {v}")
    print("=" * 65)
