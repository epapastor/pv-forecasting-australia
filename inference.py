# ======================================================
# ===================== IMPORTS ========================
# ======================================================
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from torch.utils.data import DataLoader
import pickle

import Pipeline
from models.LSTM import LSTM_two_layers
from utils.graph_pipeline import (
    plot_continuous_horizon0,
    plot_one_day,
    plot_scatter_real_vs_pred,
)

# ======================================================
# =================== HELPERS ==========================
# ======================================================
def load_trained_model(model_class, checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)

    cfg = checkpoint["config"]

    model = model_class(
        input_size=cfg["input_size"],
        hidden_size=cfg["hidden_size"],
        output_size=cfg["output_size"],
        dropout=cfg["dropout"],
    )

    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.to(device)
    model.eval()

    return model, cfg


def inference_model(model, dataloader, device):
    preds, targets = [], []

    with torch.no_grad():
        for x, y in dataloader:
            x = x.to(device)
            preds.append(model(x).cpu().numpy())
            targets.append(y.numpy())

    return (
        np.concatenate(preds, axis=0),
        np.concatenate(targets, axis=0),
    )


# ======================================================
# =================== INFERENCE ========================
# ======================================================
def run_inference():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    MODEL_PATH = "./checkpoints/lstm_model.pt"
    DATA_PATH = "./data/Processed"

    # ==================================================
    # =============== LOAD STATS =======================
    # ==================================================
    with open(f"{DATA_PATH}/stats.pkl", "rb") as f:
        stats = pickle.load(f)

    y_mu = stats["Y"]["mu"]["Energy"]
    y_std = stats["Y"]["std"]["Energy"]

    mase_scale = stats["mase"]["scale"]

    # ------------------ LOAD MODEL ---------------------
    model, cfg = load_trained_model(
        LSTM_two_layers,
        MODEL_PATH,
        device
    )

    # ------------------ LOAD DATA ---------------------
    inf_x, inf_y = Pipeline.load_split("inference", DATA_PATH)

    ds_inf = Pipeline.TimeSeriesDataset(
        inf_x,
        inf_y,
        length=cfg["length"],
        lag=cfg["lag"],
        output_window=cfg["output_window"],
        stride=1,
    )

    dl_inf = DataLoader(ds_inf, batch_size=64, shuffle=False)

    # ------------------ INFERENCE ----------------------
    preds, targets = inference_model(model, dl_inf, device)

    # ------------------ DESNORMALIZAR ------------------
    preds_real = preds * y_std + y_mu
    targets_real = targets * y_std + y_mu

    # ------------------ METRICS ------------------------
    rmse_h0 = np.sqrt(
        np.mean((targets_real[:, 0] - preds_real[:, 0]) ** 2)
    )

    mase_h0 = np.mean(
        np.abs(targets_real[:, 0] - preds_real[:, 0])
    ) / (mase_scale + 1e-8)

    print("\n========== INFERENCE METRICS ==========")
    print(f"RMSE horizon=0 (kWh): {rmse_h0:.3f}")
    print(f"MASE horizon=0:       {mase_h0:.3f}")

    # ==================================================
    # =================== PLOTS ========================
    # ==================================================
    plot_continuous_horizon0(
        targets_real,
        preds_real,
        start_idx=0,
        n_days=7,
    )

    plot_one_day(
        targets_real,
        preds_real,
        day_idx=10
    )

    plot_scatter_real_vs_pred(
        targets_real,
        preds_real
    )

    # ------------------ SAVE ---------------------------
    df = pd.DataFrame(
        preds_real,
        columns=[f"horizon_{i}" for i in range(preds_real.shape[1])]
    )

    out_path = "./outputs/inference_predictions.xlsx"
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_excel(out_path, index=False)

    print(f"\nInference saved to {out_path}")


if __name__ == "__main__":
    run_inference()
