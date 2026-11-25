"""
Projet : Efficacité Énergétique des Centres de Données (Datacenters)
Fichier : cnn_model.py
Description : Architecture optimisée du Réseau de Neurones Convolutif 2D (CNN).

Conception architecturale anti-surapprentissage (Anti-Overfitting) :
  - 3 Blocs Convolutifs hiérarchiques avec Batch Normalization & ReLU.
  - Dropout spatial 2D (Dropout2d) au sein des extracteurs de caractéristiques.
  - Pooling mixte : MaxPool2d (invariance spatiale) + Global Adaptive Average Pooling (GAP).
  - Réduction drastique des paramètres denses via GAP (~37 400 paramètres vs >200 000 auparavant).
  - Tête de classification compacte avec régularisation Dropout (30% et 20%) et BatchNorm1D.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DatacenterThermalCNN(nn.Module):
    """
    Réseau de neurones convolutif 2D optimisé pour l'analyse des cartes thermiques
    spatiales de baies de serveurs et la classification ASHRAE TC 9.9.

    Entrée :
      Tenseur de dimension (Batch_size, 2, 16, 16)
        - Canal 0 : Carte spatiale de température (°C)
        - Canal 1 : Carte spatiale d'humidité relative (% RH)

    Sortie :
      Logits bruts pour les 4 classes de criticité thermique ASHRAE :
        - Classe 0 : Optimal / Froid (< 22°C)
        - Classe 1 : Nominal Recommandé ASHRAE (22 - 27°C)
        - Classe 2 : Alerte Point Chaud Localisé (27 - 32°C)
        - Classe 3 : Surchauffe Critique (> 32°C)
    """

    def __init__(self, in_channels: int = 2, num_classes: int = 4):
        super(DatacenterThermalCNN, self).__init__()

        # =====================================================================
        # BLOC 1 : Extraction des micro-gradients thermiques locaux (16x16 -> 8x8)
        # Receptive field : capture les contrastes immédiats entre serveurs adjacents.
        # =====================================================================
        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 16, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 16x16 -> 8x8
            nn.Dropout2d(p=0.10)                    # Régularisation spatiale légère
        )

        # =====================================================================
        # BLOC 2 : Détection des panaches thermiques et points chauds (8x8 -> 4x4)
        # Capture les signatures spatiales moyennes (taches chaudes, flux d'allée).
        # =====================================================================
        self.block2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 8x8 -> 4x4
            nn.Dropout2d(p=0.15)                    # Prévient la mémorisation de motifs
        )

        # =====================================================================
        # BLOC 3 : Synthèse globale de la baie de serveurs (4x4 -> 1x1)
        # Global Adaptive Average Pooling (GAP) : élimine le risque de surapprentissage
        # des couches denses en moyennant chaque carte de caractéristiques spatiales.
        # =====================================================================
        self.block3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1))            # Réduit à 1x1 indépendamment de l'entrée
        )

        # =====================================================================
        # CLASSIFIEUR : Tête dense compacte et fortement régularisée
        # 64 caractéristiques d'entrée -> 32 neurones cachés -> 4 classes.
        # =====================================================================
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=0.30),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.20),
            nn.Linear(32, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Passe avant complète."""
        features = self.block1(x)
        features = self.block2(features)
        features = self.block3(features)
        logits = self.classifier(features)
        return logits

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Calcule les probabilités Softmax normalisées pour l'inférence."""
        logits = self.forward(x)
        return F.softmax(logits, dim=-1)

    def count_parameters(self) -> int:
        """Retourne le nombre total de paramètres entraînables."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


if __name__ == "__main__":
    model = DatacenterThermalCNN(in_channels=2, num_classes=4)
    dummy_input = torch.randn(8, 2, 16, 16)
    logits = model(dummy_input)
    probs = model.predict_proba(dummy_input)

    print("=" * 60)
    print("   Architecture DatacenterThermalCNN (Optimisée)")
    print("=" * 60)
    print(f"Entrée factice       : {tuple(dummy_input.shape)}")
    print(f"Sortie logits        : {tuple(logits.shape)}")
    print(f"Sortie probabilités  : {tuple(probs.shape)}")
    print(f"Paramètres totaux    : {model.count_parameters():,}")
    print("=" * 60)
