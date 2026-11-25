"""
Projet : Efficacité Énergétique des Centres de Données (Datacenters)
Fichier : generate_improved_dataset.py
Description : Générateur amélioré et physiquement réaliste de cartes thermiques 2D.

Améliorations scientifiques suite à l'audit du dataset initial :
  1. Scénarios strictement indépendants pour Train, Validation et Test
     (graines disjointes, dérives saisonnières distinctes, taux de pannes distincts,
     vérification quantitative d'absence de doublons et quasi-doublons inter-splits).
  2. Champ thermique latent physiquement cohérent :
     - Gradient aéraulique d'allée d'orientation arbitraire (proximité CRAH)
     - Stratification thermique verticale (montée de l'air chaud sur la hauteur U)
     - Turbulence aéraulique 2D spatialement corrélée (filtrage spectral gaussien)
     - Panaches thermiques anisotropes variés (positions, tailles, rotations, intensités)
     - Recouvrement réaliste aux frontières ASHRAE (22°C, 27°C, 32°C)
  3. Couplage psychrométrique réaliste de l'humidité relative (RH) :
     - L'humidité ambiante varie indépendamment de la classe (zéro fuite triviale par classe)
     - Variation locale couplée physiquement à la température (approximation de Clausius-Clapeyron)
  4. Imperfections réalistes de mesure des capteurs (ajoutées APRÈS détermination
     du régime physique réel, préservant des étiquettes physiquement valides) :
     - Dérive temporelle/scénario + dérive spatiale lente + biais par colonne de baies
     - Bruit de mesure blanc + bruit de capteur faiblement corrélé
     - Pannes de capteurs : pics isolés (+4.5 à +9.5°C), sondes bloquées, erreurs négatives,
       rares rafales de 2 pixels adjacents.
  5. Préservation intégrale du dataset original ('dataset/datacenter_thermal_maps.npz').

NOTE SCIENTIFIQUE : Ce dataset reste une simulation synthétique réaliste (jumeau numérique),
et non des mesures physiques brutes issues de capteurs d'un datacenter réel.
"""

import os
import json
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist


def _spatially_correlated_noise(H: int, W: int, rng: np.random.Generator, sigma: float = 1.6) -> np.ndarray:
    """Génère un champ de bruit 2D spatialement corrélé (turbulence de flux d'air)."""
    white = rng.normal(0.0, 1.0, size=(H, W))
    ky = np.fft.fftfreq(H)[:, None]
    kx = np.fft.fftfreq(W)[None, :]
    kernel_ft = np.exp(-2.0 * (np.pi ** 2) * (sigma ** 2) * (kx ** 2 + ky ** 2))
    correlated = np.real(np.fft.ifft2(np.fft.fft2(white) * kernel_ft))
    return correlated / (correlated.std() + 1e-8)


def _anisotropic_plume(H: int, W: int, cy: float, cx: float, sx: float, sy: float, theta: float) -> np.ndarray:
    """Génère un panache thermique 2D elliptique orienté (diffusion anisotrope)."""
    y, x = np.mgrid[0:H, 0:W]
    dy = y - cy
    dx = x - cx
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    xr = dx * cos_t + dy * sin_t
    yr = -dx * sin_t + dy * cos_t
    return np.exp(-0.5 * ((xr / sx) ** 2 + (yr / sy) ** 2))


def generate_split(
    num_samples: int,
    seed: int,
    split_name: str,
    grid_size: tuple = (16, 16),
    scenario_temp_shift: float = 0.0,
    scenario_drift_std: float = 0.72,
    fault_prob: float = 0.018,
    sx_range: tuple = (1.15, 2.5),
):
    """
    Génère un sous-ensemble indépendant (Train, Val ou Test) de cartes thermiques 2D.
    Les étiquettes reflètent le régime thermique physique réel (ASHRAE TC 9.9)
    du champ latent avant ajout des dérives, erreurs de mesure et pannes de capteurs.
    """
    rng = np.random.default_rng(seed)
    H, W = grid_size

    maps = np.zeros((num_samples, 2, H, W), dtype=np.float32)
    labels = np.zeros(num_samples, dtype=np.int64)
    metadata = []

    samples_per_class = num_samples // 4

    for i in range(num_samples):
        target_class = min(i // samples_per_class, 3)

        # Gradient spatial d'écoulement d'air (orientation variable selon l'implantation CRAH)
        angle = rng.uniform(0.0, 2.0 * np.pi)
        yy, xx = np.mgrid[0:H, 0:W]
        yy_c = (yy - (H - 1) / 2.0) / H
        xx_c = (xx - (W - 1) / 2.0) / W
        grad_field = np.cos(angle) * xx_c + np.sin(angle) * yy_c
        grad_amp = rng.uniform(0.7, 2.8)
        aisle_gradient = grad_amp * grad_field

        # Stratification verticale (l'air chaud s'accumule en haut des baies)
        vert_amp = rng.uniform(0.3, 1.1)
        vertical_strat = vert_amp * np.linspace(-0.5, 0.5, H)[:, None]

        # Turbulence aéraulique 2D spatialement corrélée
        turb_amp = rng.uniform(0.60, 1.45)
        turbulence = turb_amp * _spatially_correlated_noise(H, W, rng, sigma=rng.uniform(1.2, 2.4))
        bg_pattern = aisle_gradient + vertical_strat + turbulence

        if target_class == 0:
            # Classe 0 : Optimal / Sur-refroidi (< 22°C, avec recouvrement naturel vers 21.5-22.5°C)
            mean_target = rng.uniform(18.2, 22.0)
            temp_true = mean_target + 0.70 * (bg_pattern - bg_pattern.mean())
            if rng.random() < 0.30:
                cy, cx = rng.uniform(2.0, H - 2.0), rng.uniform(2.0, W - 2.0)
                plume = _anisotropic_plume(
                    H, W, cy, cx, rng.uniform(1.3, 2.4), rng.uniform(1.3, 2.4), rng.uniform(0.0, np.pi)
                )
                temp_true += rng.uniform(0.8, 1.8) * plume
            if temp_true.max() > 23.0:
                temp_true -= (temp_true.max() - rng.uniform(21.6, 22.9))

        elif target_class == 1:
            # Classe 1 : Nominal ASHRAE (22-27°C, avec recouvrement 21.5-22.5°C et 26.2-27.2°C)
            mean_target = rng.uniform(21.7, 25.5)
            temp_true = mean_target + (bg_pattern - bg_pattern.mean())
            if rng.random() < 0.45:
                cy, cx = rng.uniform(2.0, H - 2.0), rng.uniform(2.0, W - 2.0)
                plume = _anisotropic_plume(
                    H, W, cy, cx, rng.uniform(1.4, 2.7), rng.uniform(1.4, 2.7), rng.uniform(0.0, np.pi)
                )
                temp_true += rng.uniform(0.9, 2.3) * plume
            if temp_true.max() > 27.2:
                temp_true -= (temp_true.max() - rng.uniform(25.6, 27.1))
            if temp_true.mean() < 21.7:
                temp_true += (rng.uniform(21.8, 22.4) - temp_true.mean())

        elif target_class == 2:
            # Classe 2 : Alerte Point Chaud Localisé (pic physique 27-32°C sur fond nominal)
            bg_mean = rng.uniform(22.0, 25.9)
            temp_bg = bg_mean + 0.85 * (bg_pattern - bg_pattern.mean())

            plume_field = np.zeros((H, W), dtype=np.float64)
            n_hot = int(rng.integers(1, 3))
            for _ in range(n_hot):
                cy, cx = rng.uniform(1.5, H - 1.5), rng.uniform(1.5, W - 1.5)
                sx = rng.uniform(*sx_range)
                sy = rng.uniform(*sx_range)
                theta = rng.uniform(0.0, np.pi)
                w_p = rng.uniform(0.65, 1.0)
                plume_field += w_p * _anisotropic_plume(H, W, cy, cx, sx, sy, theta)
            plume_field /= (plume_field.max() + 1e-8)

            desired_peak = rng.uniform(27.0, 32.3)
            bg_at_peak = temp_bg[np.unravel_index(np.argmax(plume_field), (H, W))]
            amp = max(2.8, desired_peak - bg_at_peak)
            temp_true = temp_bg + amp * plume_field
            if temp_true.max() > 32.4:
                temp_true -= 0.5 * (temp_true.max() - 32.2)

        else:
            # Classe 3 : Surchauffe Critique (> 32°C, incluant zone frontière 31.6-33.5°C et cas sévères)
            bg_mean = rng.uniform(23.2, 28.0)
            temp_bg = bg_mean + (bg_pattern - bg_pattern.mean())

            plume_field = np.zeros((H, W), dtype=np.float64)
            n_plumes = int(rng.integers(1, 4))
            for _ in range(n_plumes):
                cy, cx = rng.uniform(1.8, H - 1.8), rng.uniform(1.8, W - 1.8)
                sx = rng.uniform(1.5, 3.4)
                sy = rng.uniform(1.5, 3.4)
                theta = rng.uniform(0.0, np.pi)
                w_p = rng.uniform(0.60, 1.0)
                plume_field += w_p * _anisotropic_plume(H, W, cy, cx, sx, sy, theta)
            plume_field /= (plume_field.max() + 1e-8)

            desired_peak = rng.uniform(31.6, 38.5)
            bg_at_peak = temp_bg[np.unravel_index(np.argmax(plume_field), (H, W))]
            amp = max(3.2, desired_peak - bg_at_peak)
            temp_true = temp_bg + amp * plume_field

        # Humidité relative couplée psychrométriquement (sans plage fixe par classe)
        rh_ref_24 = rng.uniform(35.0, 56.0)
        hum_turb = rng.uniform(1.3, 3.0) * _spatially_correlated_noise(H, W, rng, sigma=1.8)
        hum_true = rh_ref_24 - 2.0 * (temp_true - 24.0) + hum_turb

        # --- Imperfections de mesure des capteurs et dérives ---
        temp_drift = scenario_temp_shift + rng.normal(0.0, scenario_drift_std)
        rack_col_calib = rng.normal(0.0, 0.30, size=(1, W))
        spatial_drift = rng.normal(0.0, 0.45) * grad_field
        hum_drift = rng.normal(0.0, 2.8)

        meas_noise_t = rng.normal(0.0, 0.75, size=(H, W)) + 0.40 * _spatially_correlated_noise(H, W, rng, sigma=0.9)
        meas_noise_h = rng.normal(0.0, 2.4, size=(H, W))

        temp_obs = temp_true + temp_drift + rack_col_calib + spatial_drift + meas_noise_t
        hum_obs = hum_true + hum_drift + meas_noise_h

        # Pannes de capteurs réalistes
        fault_mask = rng.random((H, W)) < fault_prob
        if np.any(fault_mask):
            ys, xs = np.where(fault_mask)
            for y_f, x_f in zip(ys, xs):
                ft = rng.random()
                if ft < 0.45:
                    # Pic électrique isolé sur 1 pixel
                    temp_obs[y_f, x_f] += rng.uniform(4.5, 9.5)
                elif ft < 0.75:
                    # Capteur bloqué / déconnecté
                    temp_obs[y_f, x_f] = rng.uniform(18.0, 23.0)
                    hum_obs[y_f, x_f] = rng.uniform(42.0, 56.0)
                elif ft < 0.90:
                    # Chute / erreur négative isolée
                    temp_obs[y_f, x_f] -= rng.uniform(3.5, 6.5)
                else:
                    # Courte rafale de bruit sur 2 capteurs voisins
                    spike = rng.uniform(3.5, 6.5)
                    temp_obs[y_f, x_f] += spike
                    ny = min(H - 1, max(0, y_f + int(rng.choice([-1, 0, 1]))))
                    nx = min(W - 1, max(0, x_f + int(rng.choice([-1, 0, 1]))))
                    temp_obs[ny, nx] += spike * 0.7

        temp_obs = np.clip(temp_obs, 14.0, 49.0)
        hum_obs = np.clip(hum_obs, 12.0, 82.0)

        maps[i, 0] = temp_obs
        maps[i, 1] = hum_obs
        labels[i] = target_class

        metadata.append({
            "split": split_name,
            "sample_id": i,
            "target_class": target_class,
            "true_mean_temp": float(temp_true.mean()),
            "true_max_temp": float(temp_true.max()),
            "max_temp": float(temp_obs.max()),
            "mean_temp": float(temp_obs.mean()),
            "min_humidity": float(hum_obs.min()),
            "mean_humidity": float(hum_obs.mean()),
            "temp_drift": float(temp_drift),
            "faulty_sensors": int(fault_mask.sum()),
        })

    perm = rng.permutation(num_samples)
    maps = maps[perm]
    labels = labels[perm]
    df = pd.DataFrame(metadata).iloc[perm].reset_index(drop=True)
    return maps, labels, df


def verify_split_independence(X_train, X_val, X_test) -> dict:
    """Vérifie l'absence de doublons exacts et de quasi-doublons entre splits."""
    mean_tr = X_train.mean(axis=(0, 2, 3), keepdims=True)
    std_tr = X_train.std(axis=(0, 2, 3), keepdims=True) + 1e-7

    tr_n = ((X_train - mean_tr) / std_tr).reshape(len(X_train), -1)
    va_n = ((X_val - mean_tr) / std_tr).reshape(len(X_val), -1)
    te_n = ((X_test - mean_tr) / std_tr).reshape(len(X_test), -1)

    all_n = np.vstack([tr_n, va_n, te_n])
    unique_count = len(np.unique(all_n, axis=0))

    d_va_tr = cdist(va_n, tr_n, metric="euclidean").min(axis=1)
    d_te_tr = cdist(te_n, tr_n, metric="euclidean").min(axis=1)
    d_te_va = cdist(te_n, va_n, metric="euclidean").min(axis=1)

    return {
        "total_samples": int(len(all_n)),
        "exact_unique_samples": int(unique_count),
        "exact_duplicates": int(len(all_n) - unique_count),
        "nn_l2_val_to_train": {
            "min": float(d_va_tr.min()),
            "p01": float(np.percentile(d_va_tr, 1)),
            "median": float(np.median(d_va_tr)),
        },
        "nn_l2_test_to_train": {
            "min": float(d_te_tr.min()),
            "p01": float(np.percentile(d_te_tr, 1)),
            "median": float(np.median(d_te_tr)),
        },
        "nn_l2_test_to_val": {
            "min": float(d_te_va.min()),
            "p01": float(np.percentile(d_te_va, 1)),
            "median": float(np.median(d_te_va)),
        },
    }


def build_and_save_improved_dataset(output_dir: str = None):
    """Construit et sauvegarde les 3 splits indépendants du dataset amélioré."""
    project_root = os.path.dirname(os.path.abspath(__file__))
    if output_dir is None:
        output_dir = os.path.join(project_root, "dataset")
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 70)
    print("   GÉNÉRATION DU DATASET AMÉLIORÉ (JUMEAU NUMÉRIQUE RÉALISTE)")
    print("=" * 70)

    X_train, y_train, df_train = generate_split(
        num_samples=2400, seed=1001, split_name="train",
        scenario_temp_shift=0.0, scenario_drift_std=0.72, fault_prob=0.018, sx_range=(1.15, 2.4)
    )
    X_val, y_val, df_val = generate_split(
        num_samples=600, seed=2002, split_name="val",
        scenario_temp_shift=0.25, scenario_drift_std=0.75, fault_prob=0.020, sx_range=(1.15, 2.5)
    )
    X_test, y_test, df_test = generate_split(
        num_samples=600, seed=3003, split_name="test",
        scenario_temp_shift=-0.20, scenario_drift_std=0.75, fault_prob=0.022, sx_range=(1.20, 2.6)
    )

    npz_path = os.path.join(output_dir, "improved_thermal_maps.npz")
    np.savez_compressed(
        npz_path,
        X_train=X_train, y_train=y_train,
        X_val=X_val, y_val=y_val,
        X_test=X_test, y_test=y_test,
    )

    df_all = pd.concat([df_train, df_val, df_test], ignore_index=True)
    csv_path = os.path.join(output_dir, "improved_metadata.csv")
    df_all.to_csv(csv_path, index=False)

    indep_report = verify_split_independence(X_train, X_val, X_test)

    print(f"Dataset amélioré sauvegardé : {npz_path}")
    print(f"  - Train : {X_train.shape} | Classes : {dict(zip(*np.unique(y_train, return_counts=True)))}")
    print(f"  - Val   : {X_val.shape}   | Classes : {dict(zip(*np.unique(y_val, return_counts=True)))}")
    print(f"  - Test  : {X_test.shape}  | Classes : {dict(zip(*np.unique(y_test, return_counts=True)))}")
    print(f"Métadonnées : {csv_path} ({len(df_all)} lignes)")
    print(f"Vérification anti-doublon : {indep_report['exact_unique_samples']}/{indep_report['total_samples']} uniques")
    print(f"  - NN L2 Val->Train  : min={indep_report['nn_l2_val_to_train']['min']:.3f}, median={indep_report['nn_l2_val_to_train']['median']:.3f}")
    print(f"  - NN L2 Test->Train : min={indep_report['nn_l2_test_to_train']['min']:.3f}, median={indep_report['nn_l2_test_to_train']['median']:.3f}")
    print("=" * 70)
    return npz_path, csv_path, indep_report


if __name__ == "__main__":
    build_and_save_improved_dataset()
