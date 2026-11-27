from pathlib import Path
import copy, json, random, time
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from neural_network.data import FEATURES, load_dataset, split_and_scale
from neural_network.model import ARCHITECTURES, VGNeuralNetwork

SEED, BATCH_SIZE, EPOCHS = 42, 64, 500
LEARNING_RATE, WEIGHT_DECAY, PATIENCE = 1e-3, 1e-5, 40 #si la validation ne s'améliore plus pendant 40 époques, on arrête.

DATA_IN = Path("data/entree.csv")
DATA_OUT = Path("data/sorties.csv")

OUT = Path("outputs")
MODELS = OUT / "models"
SCALERS = OUT / "scalers"
HISTORIES = OUT / "histories"
FIGURES = OUT / "figures"
RESULTS = OUT / "results"

#Poids des pénalités
LAMBDA_T = 0.1
LAMBDA_K = 0.1
LAMBDA_CONVEX = 0.02


def penalite_financiere(model, xb, pas_T, pas_K, bornes):
    """Punit les prix qui violent les règles financières (T, K, convexité)"""

    #fait varier T, en gardant le reste fixe
    x_base = xb.clone()
    x_base[:, 0] = torch.clamp(x_base[:, 0], bornes[0][0], bornes[0][1] - pas_T)
    x_base[:, 1] = torch.clamp(x_base[:, 1], bornes[1][0] + pas_K, bornes[1][1] - pas_K)
    x_T = x_base.clone()
    x_T[:, 0] += pas_T

    p0 = model(x_base) #prix départ
    pT = model(x_T)

    #deux prix pour la monotonie en K et un troisieme pour la convexite.
    x_K_plus = x_base.clone()
    x_K_moins = x_base.clone()
    x_K_plus[:, 1] += pas_K
    x_K_moins[:, 1] -= pas_K
    p_plus = model(x_K_plus)
    p_moins = model(x_K_moins)

    #penaliser quand c'est pas cohérent
    #prix doit augmenter avec T
    pen_T = torch.relu((p0 - pT) / pas_T).square().mean()
    #prix doit baisser avec K
    pen_K = torch.relu((p_plus - p0) / pas_K).square().mean()
    #prix doit être convexe en K
    pen_conv = torch.relu((2 * p0 - p_plus - p_moins) / (pas_K ** 2)).square().mean()
    return pen_T, pen_K, pen_conv


def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

def predict(model, X_scaled, device):
    model.eval()
    with torch.no_grad():
        p = model(torch.tensor(X_scaled, dtype=torch.float32, device=device)).cpu().numpy()
    return p

def train_one(name, hidden_sizes, split, device):

    set_seed(SEED)
    train_ds = TensorDataset(torch.tensor(split["X_train_scaled"], dtype=torch.float32),
                             torch.tensor(split["y_train_scaled"], dtype=torch.float32))
    
    val_ds = TensorDataset(torch.tensor(split["X_val_scaled"], dtype=torch.float32),
                           torch.tensor(split["y_val_scaled"], dtype=torch.float32))
    
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              generator=torch.Generator().manual_seed(SEED))
    
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

    model = VGNeuralNetwork(len(FEATURES), hidden_sizes).to(device)
    loss_fn = nn.MSELoss()
    contraint = name.endswith("_contraint")

    #pas correspond à 1 % de l'etendue train
    sx = split["scaler_X"]
    t_range = float(split["X_train"]["T"].max() - split["X_train"]["T"].min())
    k_range = float(split["X_train"]["K"].max() - split["X_train"]["K"].min())
    pas_T = 0.01 * t_range / float(sx.scale_[0])
    pas_K = 0.01 * k_range / float(sx.scale_[1])
    bornes = []
    for j, col in enumerate(["T", "K"]):
        lo = (float(split["X_train"][col].min()) - float(sx.mean_[j])) / float(sx.scale_[j])
        hi = (float(split["X_train"][col].max()) - float(sx.mean_[j])) / float(sx.scale_[j])
        bornes.append((lo, hi))
    if contraint and (pas_T <= 0 or pas_K <= 0):
        raise ValueError("Les donnees doivent avoir plusieurs T et K distincts.")
    
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=.5, patience=10)

    best_val, best_weights, wait = float("inf"), None, 0
    history = []
    start = time.perf_counter()
    for epoch in range(1, EPOCHS+1):
        model.train(); train_sum = 0.
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            mse = loss_fn(model(xb), yb)
            loss = mse
            if contraint:
                pt, pk, pc = penalite_financiere(model, xb, pas_T, pas_K, bornes)
                loss = mse + LAMBDA_T * pt + LAMBDA_K * pk + LAMBDA_CONVEX * pc
            loss.backward(); optimizer.step()
            train_sum += loss.item()*len(xb)
        train_loss = train_sum/len(train_ds)

        model.eval(); val_sum = 0.
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                val_sum += loss_fn(model(xb), yb).item()*len(xb)
        val_loss = val_sum/len(val_ds)
        scheduler.step(val_loss)
        history.append({"epoch":epoch, "train_mse_scaled":train_loss,
                        "validation_mse_scaled":val_loss, "learning_rate":optimizer.param_groups[0]["lr"]})
        if val_loss < best_val:
            best_val, best_weights, wait = val_loss, copy.deepcopy(model.state_dict()), 0
        else:
            wait += 1
        if epoch == 1 or epoch % 25 == 0:
            print(f"  {name} | époque {epoch:03d} | train={train_loss:.6g} | val={val_loss:.6g}")
        if wait >= PATIENCE:
            print(f"  Early stopping à l'époque {epoch}.")
            break
    elapsed = time.perf_counter()-start
    model.load_state_dict(best_weights)

    model_path = MODELS / f"{name}.pth"
    torch.save(model.state_dict(), model_path)
    pd.DataFrame(history).to_csv(
        HISTORIES / f"{name}_history.csv",
        index=False
    )

    # Scores de validation en unités de prix, pour choisir sans toucher au test.
    val_pred_scaled = predict(model, split["X_val_scaled"], device)
    val_pred = split["scaler_y"].inverse_transform(val_pred_scaled).ravel()
    yval = split["y_val"].to_numpy()
    val_mse = mean_squared_error(yval, val_pred)
    metrics = {"model":name, "architecture":"-".join(map(str,hidden_sizes)),
               "parameters":sum(p.numel() for p in model.parameters()),
               "best_validation_MSE":float(val_mse),
               "best_validation_RMSE":float(np.sqrt(val_mse)),
               "best_validation_MAE":float(mean_absolute_error(yval,val_pred)),
               "epochs":len(history), "training_seconds":elapsed,
               "contraint":contraint, "lambda_T":LAMBDA_T if contraint else 0.,
               "lambda_K":LAMBDA_K if contraint else 0.,
               "lambda_convex":LAMBDA_CONVEX if contraint else 0.}
    
    return metrics, history

def main():

    for folder in [MODELS, SCALERS, HISTORIES, FIGURES, RESULTS]:
        folder.mkdir(exist_ok=True, parents=True)

    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Calcul sur : {device}")
    data = load_dataset(DATA_IN, DATA_OUT)
    split = split_and_scale(data, SEED)

    joblib.dump(split["scaler_X"], SCALERS / "scaler_X.joblib")
    joblib.dump(split["scaler_y"], SCALERS / "scaler_y.joblib")

    # Fichier de test commun, sauvegardé une seule fois.
    test = split["X_test"].copy()
    test["price"] = split["y_test"].to_numpy()
    original_test_meta = data.iloc[split["test_idx"]].reset_index(drop=True)
    
    for col in ["ci_low","ci_high","model_id","product_id","stress"]:
        if col in original_test_meta: test[col] = original_test_meta[col].to_numpy()
        
    test.to_csv(RESULTS / "test_set.csv", index=False)

    all_metrics, histories = [], {}
    for name, widths in ARCHITECTURES.items():
        print(f"\n===== ENTRAÎNEMENT : {name} ({widths}) =====")
        metrics, history = train_one(name, widths, split, device)
        all_metrics.append(metrics); histories[name] = pd.DataFrame(history)
        print(f"  Validation RMSE = {metrics['best_validation_RMSE']:.8g} | "
              f"temps = {metrics['training_seconds']:.1f}s")

    benchmark = pd.DataFrame(all_metrics).sort_values("best_validation_RMSE")
    benchmark.to_csv(RESULTS / "benchmark_validation.csv", index=False)
    winner = benchmark.iloc[0]["model"]
    (RESULTS / "selected_model.json").write_text(json.dumps({
        "selected_model":winner,
        "selection_rule":"Plus faible RMSE sur validation (le test n'intervient pas dans la sélection).",
        "architectures":ARCHITECTURES
    }, indent=2, ensure_ascii=False), encoding="utf-8")

    # Courbes de pertes comparatives.
    plt.figure(figsize=(10,6))

    for name, hist in histories.items():
        plt.plot(hist["epoch"], hist["validation_mse_scaled"], label=name)

    plt.xlabel("Époque"); plt.ylabel("MSE validation (cible normalisée)")
    plt.title("Comparaison des pertes de validation")
    plt.legend()
    plt.grid(alpha=.3)
    plt.tight_layout()
    plt.savefig(FIGURES / "validation_curves.png", dpi=160); 
    plt.close()

    print("\n===== CLASSEMENT VALIDATION =====")
    print(benchmark.to_string(index=False))
    print(f"\nModèle retenu d'après la validation : {winner}")
    print("Lancez maintenant : python -m analyse.evaluate")

if __name__ == "__main__": main()
