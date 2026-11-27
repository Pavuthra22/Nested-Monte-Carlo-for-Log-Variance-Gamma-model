from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler

FEATURES = ["T", "K", "kappa", "theta", "sigma"]
TARGET = "price"
PARAM_TOLERANCE = 2e-6


def _read_csv(path):
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(
            f"Fichier introuvable : {path.resolve()}"
        )

    df = pd.read_csv(path)

    #Conserver majuscules de T et K
    df.columns = (
        df.columns.astype(str)
        .str.strip()
        .str.replace(" ", "_", regex=False)
    )

    return df


def load_dataset(input_file="data/entree.csv", output_file="data/sorties.csv"):
    """Renvoie un tableau avec les paramètres (T, K, kappa, theta, sigma) et le prix"""

    inputs, outputs = _read_csv(input_file), _read_csv(output_file)
    #vérification si colonnes bien présentes
    missing_in = [c for c in FEATURES if c not in inputs.columns]
    missing_out = [c for c in FEATURES + [TARGET] if c not in outputs.columns]
    if missing_in:
        raise ValueError(f"Colonnes absentes de entree.csv : {missing_in}")
    if missing_out:
        raise ValueError(f"Colonnes absentes de sorties.csv : {missing_out}")
    #vérification taille
    if len(inputs) != len(outputs):
        raise ValueError(f"Nombre de lignes différent : entree={len(inputs)}, sorties={len(outputs)}")

    #Conversion en numérique
    for col in FEATURES:
        inputs[col] = pd.to_numeric(inputs[col], errors="coerce")
        outputs[col] = pd.to_numeric(outputs[col], errors="coerce")
    for col in [TARGET, "ci_low", "ci_high"]:
        if col in outputs:
            outputs[col] = pd.to_numeric(outputs[col], errors="coerce")

    if inputs[FEATURES].isna().any().any() or outputs[FEATURES + [TARGET]].isna().any().any():
        raise ValueError("Valeurs manquantes ou non numériques dans les paramètres/prix.")

    #vérification alignement des lignes
    a = inputs[FEATURES].to_numpy(dtype=np.float64)
    b = outputs[FEATURES].to_numpy(dtype=np.float64)
    bad = np.argwhere(np.abs(a-b) > PARAM_TOLERANCE)
    if len(bad):
        row, col_idx = bad[0]
        raise ValueError(
            f"Décalage entrée/sortie à la ligne CSV {row+2}, colonne {FEATURES[col_idx]} : "
            f"{a[row, col_idx]:.12g} contre {b[row, col_idx]:.12g}. "
            "Les deux fichiers doivent provenir de la même simulation et garder le même ordre."
        )

    #construction tableau final
    data = inputs[FEATURES].copy()
    data[TARGET] = outputs[TARGET].to_numpy()
    for col in ["ci_low", "ci_high"]:
        if col in outputs:
            data[col] = outputs[col].to_numpy()
    for col in ["model_id", "product_id", "stress"]:
        if col in inputs:
            data[col] = inputs[col].to_numpy()

    data = data.replace([np.inf, -np.inf], np.nan)
    before = len(data)
    data = data.dropna(subset=FEATURES + [TARGET]).reset_index(drop=True)
    print(f"Association validée : {len(data)} lignes conservées ({before-len(data)} supprimées).")

    for col in ["T", "K", "kappa", "sigma"]:
        if (data[col] <= 0).any():
            raise ValueError(f"La colonne {col} doit être strictement positive.")
    if (data[TARGET] < 0).any():
        raise ValueError("Des prix négatifs ont été détectés.")
    if {"ci_low", "ci_high"}.issubset(data.columns) and (data["ci_low"] > data["ci_high"]).any():
        raise ValueError("Certaines bornes ci_low sont supérieures à ci_high.")
 
    return data

def split_and_scale(data, seed=42):
    """Sépare les données en train/validation/test et renvoie un dictionnaire"""

    X, y = data[FEATURES].copy(), data[TARGET].copy()

    #regroupe les 5 paramètres en une seule chaîne
    groups = data[FEATURES].round(10).astype(str).agg("|".join, axis=1)
    if groups.nunique() < 10:
        raise ValueError("Pas assez de combinaisons distinctes de paramètres pour un split fiable.")
    
    s1 = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=seed)
    train_val, test_idx = next(s1.split(X, y, groups))

    #sépare train et validation
    #redécoupe uniquement les 85 % de train_val, pour que la validation représente 15 %
    s2 = GroupShuffleSplit(n_splits=1, test_size=0.15/0.85, random_state=seed)
    train_rel, val_rel = next(s2.split(X.iloc[train_val], y.iloc[train_val], groups.iloc[train_val]))

    #traduit les indices relatifs en indices du tableau complet
    train_idx, val_idx = train_val[train_rel], train_val[val_rel]

    #Extrait les données
    X_train, X_val, X_test = X.iloc[train_idx], X.iloc[val_idx], X.iloc[test_idx]
    y_train, y_val, y_test = y.iloc[train_idx], y.iloc[val_idx], y.iloc[test_idx]

    #centre et réduit
    sx, sy = StandardScaler(), StandardScaler()

    Xtr = sx.fit_transform(X_train)
    Xva, Xte = sx.transform(X_val), sx.transform(X_test)

    ytr = sy.fit_transform(y_train.to_numpy().reshape(-1,1))
    yva = sy.transform(y_val.to_numpy().reshape(-1,1))

    print(f"Split commun : train={len(train_idx)} ({len(train_idx)/len(data):.1%}), "
          f"validation={len(val_idx)} ({len(val_idx)/len(data):.1%}), test={len(test_idx)} ({len(test_idx)/len(data):.1%})")
    
    return dict(X_train=X_train, X_val=X_val, X_test=X_test,
                y_train=y_train, y_val=y_val, y_test=y_test,
                X_train_scaled=Xtr, X_val_scaled=Xva, X_test_scaled=Xte,
                y_train_scaled=ytr, y_val_scaled=yva, scaler_X=sx, scaler_y=sy,
                train_idx=train_idx, val_idx=val_idx, test_idx=test_idx)
