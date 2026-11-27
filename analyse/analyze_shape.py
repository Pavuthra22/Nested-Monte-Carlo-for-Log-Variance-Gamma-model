from pathlib import Path
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from neural_network.data import FEATURES
from neural_network.model import ARCHITECTURES, VGNeuralNetwork

OUT = Path("outputs")
MODELS = OUT / "models"
SCALERS = OUT / "scalers"
FIGURES = OUT / "figures"
RESULTS = OUT / "results"

NB_COUPES = 12
POINTS = 120

def charger_modele(nom, largeurs, device):
    net = VGNeuralNetwork(len(FEATURES), largeurs).to(device)
    net.load_state_dict(torch.load(
        MODELS / f"{nom}.pth",
        map_location=device,
        weights_only=True
    ))
    net.eval()

    return net


def predire(net, sx, sy, matrice, device):
    matrice = pd.DataFrame(matrice, columns=FEATURES)
    x = torch.tensor(sx.transform(matrice), dtype=torch.float32, device=device)

    with torch.no_grad():
        p = net(x).cpu().numpy()

    return sy.inverse_transform(p).ravel()


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    test = pd.read_csv(RESULTS / "test_set.csv")

    sx = joblib.load(SCALERS / "scaler_X.joblib")
    sy = joblib.load(SCALERS / "scaler_y.joblib")

    modeles = test[["kappa", "theta", "sigma"]].drop_duplicates()

    if modeles.empty:
        raise ValueError("Jeu de test vide")

    positions = np.linspace(
        0,
        len(modeles) - 1,
        min(NB_COUPES, len(modeles)),
        dtype=int
    )

    configurations = modeles.iloc[positions]

    #Construction des grille
    tmin = test["T"].min()
    tmax = test["T"].max()
    kmin = test["K"].min()
    kmax = test["K"].max()

    if tmax <= tmin or kmax <= kmin:
        raise ValueError("T ou K constant dans le jeu de test")

    tgrid = np.linspace(tmin, tmax, POINTS)
    kgrid = np.linspace(kmin, kmax, POINTS)

    tfix = test["T"].median()
    kfix = test["K"].median()

    rows = []
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.5))

    #Analyse des proprietes financières
    for name, widths in ARCHITECTURES.items():

        net = charger_modele(name, widths, device)

        for icoupe, (_, cfg) in enumerate(configurations.iterrows()):

            kappa = float(cfg["kappa"])
            theta = float(cfg["theta"])
            sigma = float(cfg["sigma"])

            XT = np.column_stack((
                tgrid,
                np.full(POINTS, kfix),
                np.full(POINTS, kappa),
                np.full(POINTS, theta),
                np.full(POINTS, sigma)
            ))

            XK = np.column_stack((
                np.full(POINTS, tfix),
                kgrid,
                np.full(POINTS, kappa),
                np.full(POINTS, theta),
                np.full(POINTS, sigma)
            ))

            pt = predire(net, sx, sy, XT, device)
            pk = predire(net, sx, sy, XK, device)

            #Differences premieres et secondes
            dt = np.diff(pt)
            dk = np.diff(pk)
            d2k = np.diff(pk, n=2)

            tol = 1e-6 * max(1.0, np.abs(pt).max(), np.abs(pk).max())

            vT = int(np.sum(dt < -tol))
            vK = int(np.sum(dk > tol))
            vC = int(np.sum(d2k < -tol))

            rows.append({
                "model": name,
                "coupe": icoupe,
                "kappa": kappa,
                "theta": theta,
                "sigma": sigma,
                "T_violations": vT,
                "K_violations": vK,
                "convexite_violations": vC,
                "T_intervalles": len(dt),
                "K_intervalles": len(dk),
                "convexite_points": len(d2k)
            })

            # Representation de la premiere coupe
            if icoupe == 0:
                etiquette = name.replace("mlp_", "")

                axes[0].plot(tgrid, pt, label=etiquette)
                axes[1].plot(kgrid, pk, label=etiquette)
                axes[2].plot(kgrid[1:-1], d2k, label=etiquette)

    
    axes[0].set(
        title="Prix selon T",
        xlabel="Maturite T",
        ylabel="Prix predit"
    )
    axes[1].set(
        title="Prix selon K",
        xlabel="Strike K",
        ylabel="Prix predit"
    )
    axes[2].set(
        title="Convexite selon K",
        xlabel="Strike K",
        ylabel="Difference seconde"
    )

    axes[2].axhline(0, color="black", linestyle="--", linewidth=1)

    for ax in axes:
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)

    fig.suptitle("Comparaison sur une coupe representative (diagnostic sur plusieurs coupes)")
    fig.tight_layout()
    fig.savefig(FIGURES / "shape_comparison_before_after.png", dpi=160)
    plt.close(fig)

    #Sauvegarde diagnostics par coupe
    diagnostics = pd.DataFrame(rows)
    diagnostics.to_csv(RESULTS / "shape_diagnostics_coupes.csv", index=False)

    #Regroupement resultats par modele
    resume = diagnostics.groupby("model", as_index=False).agg(
        T_violations=("T_violations", "sum"),
        K_violations=("K_violations", "sum"),
        convexite_violations=("convexite_violations", "sum"),
        T_intervalles=("T_intervalles", "sum"),
        K_intervalles=("K_intervalles", "sum"),
        convexite_points=("convexite_points", "sum")
    )

    #Calcul taux de violations
    resume["T_taux_violations"] = resume["T_violations"] / resume["T_intervalles"]
    resume["K_taux_violations"] = resume["K_violations"] / resume["K_intervalles"]
    resume["convexite_taux_violations"] = resume["convexite_violations"] / resume["convexite_points"]

    resume.to_csv(RESULTS / "shape_diagnostics_all_models.csv", index=False)

    print(resume.to_string(index=False))


if __name__ == "__main__":
    main()