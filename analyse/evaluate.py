from pathlib import Path
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from neural_network.data import FEATURES
from neural_network.model import ARCHITECTURES, VGNeuralNetwork

OUT = Path("outputs")
MODELS = OUT / "models"
SCALERS = OUT / "scalers"
FIGURES = OUT / "figures"
RESULTS = OUT / "results"

def main():
    test_path = RESULTS / "test_set.csv"
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test = pd.read_csv(test_path)
    X = test[FEATURES]
    y = test["price"].to_numpy(float)
    rows = []
    predictions = test.copy()

    sx = joblib.load(SCALERS / "scaler_X.joblib")
    sy = joblib.load(SCALERS / "scaler_y.joblib")
    X_tensor = torch.tensor(sx.transform(X), dtype=torch.float32, device=device)

    for name, widths in ARCHITECTURES.items():

        model_path = MODELS / f"{name}.pth"
        
        model = VGNeuralNetwork(len(FEATURES), widths).to(device)
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
        model.eval()

        with torch.no_grad():
            ps = model(X_tensor).cpu().numpy()

        pred = sy.inverse_transform(ps).ravel()
        mse = mean_squared_error(y, pred)
        rows.append({
            "model": name,
            "architecture": "-".join(map(str, widths)),
            "MSE": mse,
            "RMSE": np.sqrt(mse),
            "MAE": mean_absolute_error(y, pred),
            "R2": r2_score(y, pred),
            "n_test": len(y)
        })
        predictions[f"prediction_{name}"] = pred
        predictions[f"error_{name}"] = pred - y

    metrics = pd.DataFrame(rows).sort_values("RMSE")
    metrics.to_csv(RESULTS/"benchmark_test.csv", index=False)
    predictions.to_csv(RESULTS/"test_predictions_all_models.csv", index=False)

    print(metrics.to_string(index=False))

    lo = min(y.min(), predictions.filter(like="prediction_").min().min())
    hi = max(y.max(), predictions.filter(like="prediction_").max().max())

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.ravel()

    for ax, name in zip(axes, ARCHITECTURES):
        ax.scatter(y, predictions[f"prediction_{name}"], alpha=.45, s=14)
        ax.plot([lo,hi],[lo,hi],"k--",linewidth=1)
        ax.set_title(name.replace("mlp_",""))
        ax.set_xlabel("Prix Monte Carlo")
        ax.set_ylabel("Prix prédit")
        ax.grid(alpha=0.25)

    fig.tight_layout()
    fig.savefig(FIGURES/"actual_vs_predicted_comparison.png",dpi=160)
    plt.close(fig)

if __name__ == "__main__": 
    main()
