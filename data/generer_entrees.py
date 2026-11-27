"""Génère entree.csv : python generer_entree.py [chemin_du_csv]."""
from pathlib import Path
import sys

import numpy as np
import pandas as pd


def generer():
    """Construit les 1 000 modèles et les 100 produits associés."""
    rng = np.random.default_rng(2026)
    modeles = []
    for i in range(1000):
        stress = i >= 900
        while True:
            sigma = rng.uniform(0.04, 0.65) if stress else rng.uniform(0.08, 0.45)
            kappa = rng.uniform(0.05, 0.80) if stress else rng.uniform(0.03, 0.50)
            theta = rng.uniform(-0.50, 0.35) if stress else rng.uniform(-0.35, 0.15)
            # Correction martingale et second moment fini, avec une marge.
            if (1 - theta*kappa - 0.5*sigma**2*kappa > 0.05
                    and 1 - 2*theta*kappa - 2*sigma**2*kappa > 0.05):
                break
        modeles.append((i, int(stress), kappa, theta, sigma))

    modeles = pd.DataFrame(
        modeles, columns=["model_id", "stress", "kappa", "theta", "sigma"]
    )
    maturites = np.array([21, 42, 63, 126, 189, 252, 378, 504, 630, 756]) / 252
    produits = pd.DataFrame(
        [(T, np.exp(m*T)) for T in maturites for m in np.linspace(-0.2, 0.2, 10)],
        columns=["T", "K"],
    )
    produits.insert(0, "product_id", np.arange(len(produits)))
    dataset = modeles.merge(produits, how="cross")
    # Ordre des colonnes attendu par le lecteur CSV CUDA du TP2.
    return dataset[["model_id", "product_id", "stress", "T", "K", "kappa", "theta", "sigma"]]


if __name__ == "__main__":
    if len(sys.argv) > 2:
        raise SystemExit("Usage : python generer_entree.py [chemin_du_csv]")
    sortie = Path(sys.argv[1]) if len(sys.argv) == 2 else Path("entree.csv")
    generer().to_csv(sortie, index=False, float_format="%.9g", lineterminator="\n")
    print(f"100 000 lignes écrites dans {sortie.resolve()} (S0=1, r=q=0).")
