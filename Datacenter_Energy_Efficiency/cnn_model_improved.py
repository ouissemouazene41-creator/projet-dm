"""
Projet : Efficacité Énergétique des Centres de Données (Datacenters)
Fichier : cnn_model_improved.py
Description : Architecture CNN 2D optimisée pour la détection conjointe des régimes
              thermiques globaux et des panaches localisés en présence de pannes capteurs.

Améliorations architecturales par rapport au CNN initial :
  1. Filtrage local hiérarchique avec bloc résiduel (préserve les panaches 2D tout en
     rejetant les pics isolés sur 1 pixel dus aux pannes de capteurs).
  2. Dual Global Pooling (GAP + GMP) :
       - Global Average Pooling (GAP) : capture l'état thermique global de la salle
         (essentiel pour séparer Optimal <22°C et Nominal 22-27°C).
       - Global Max Pooling (GMP) : détecte la présence d'un point chaud localisé
         indépendamment de sa position spatiale (corrige le sous-rappel sur la Classe 2).
  3. Régularisation paramétrable (Batch Normalization, Dropout2d spatial, Dropout dense)
     pour permettre une étude d'ablation rigoureuse sur la validation uniquement.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualConvBlock(nn.Module):
    """Bloc convolutif 2D avec connexion résiduelle et Batch Normalization optionnelle."""

    def __init__(self, in_ch: int, out_ch: int, use_bn: bool = True, dropout2d: float = 0.0):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=not use_bn)
        self.bn1 = nn.BatchNorm2d(out_ch) if use_bn else nn.Identity()
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=not use_bn)
        self.bn2 = nn.BatchNorm2d(out_ch) if use_bn else nn.Identity()
        self.shortcut = (
            nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=not use_bn),
                nn.BatchNorm2d(out_ch) if use_bn else nn.Identity()
            )
            if in_ch != out_ch
            else nn.Identity()
        )
        self.relu = nn.ReLU(inplace=True)
        self.drop = nn.Dropout2d(p=dropout2d) if dropout2d > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.relu(out + self.shortcut(x))
        out = self.drop(out)
        return out


class OptimizedThermalCNN(nn.Module):
    """
    CNN 2D optimisé pour l'analyse thermique de datacenters (16x16x2 -> 4 classes ASHRAE).
    Combine extraction résiduelle multi-échelle, Dual Pooling (GAP + GMP) et
    tête de classification régularisée.
    """

    def __init__(
        self,
        in_channels: int = 2,
        num_classes: int = 4,
        use_bn: bool = True,
        dropout_conv1: float = 0.10,
        dropout_conv2: float = 0.15,
        dropout_fc1: float = 0.35,
        dropout_fc2: float = 0.20,
        pooling_mode: str = "dual",
    ):
        super().__init__()
        self.pooling_mode = pooling_mode
        self.use_bn = use_bn

        # Bloc 1 : 16x16 -> 8x8 (filtrage des pics isolés 1px + micro-gradients)
        self.block1 = ResidualConvBlock(in_channels, 24, use_bn=use_bn, dropout2d=dropout_conv1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Bloc 2 : 8x8 -> 4x4 (détection des panaches thermiques et corrélation T/RH)
        self.block2 = ResidualConvBlock(24, 48, use_bn=use_bn, dropout2d=dropout_conv2)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Bloc 3 : 4x4 (synthèse des motifs spatiaux avant pooling global)
        self.block3 = nn.Sequential(
            nn.Conv2d(48, 64, kernel_size=3, padding=1, bias=not use_bn),
            nn.BatchNorm2d(64) if use_bn else nn.Identity(),
            nn.ReLU(inplace=True),
        )

        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.gmp = nn.AdaptiveMaxPool2d((1, 1))

        feat_dim = 128 if pooling_mode == "dual" else 64

        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout_fc1) if dropout_fc1 > 0.0 else nn.Identity(),
            nn.Linear(feat_dim, 48),
            nn.BatchNorm1d(48) if use_bn else nn.Identity(),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_fc2) if dropout_fc2 > 0.0 else nn.Identity(),
            nn.Linear(48, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool1(self.block1(x))
        x = self.pool2(self.block2(x))
        x = self.block3(x)

        avg_feat = self.gap(x).flatten(1)
        if self.pooling_mode == "dual":
            max_feat = self.gmp(x).flatten(1)
            feats = torch.cat([avg_feat, max_feat], dim=1)
        else:
            feats = avg_feat

        return self.classifier(feats)

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        return F.softmax(self.forward(x), dim=-1)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


if __name__ == "__main__":
    model = OptimizedThermalCNN()
    dummy = torch.randn(8, 2, 16, 16)
    out = model(dummy)
    print(f"OptimizedThermalCNN | out={tuple(out.shape)} | params={model.count_parameters():,}")
