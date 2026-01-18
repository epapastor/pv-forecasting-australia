import torch
import numpy as np
import yaml
import pandas as pd

from torch.utils.data import DataLoader
from pathlib import Path
from train import TimeSeriesDataset
from models.LSTM import LSTM_two_layers
from models.GRU import GRU_two_layers
from models.LSTM_FCN import LSTM_FCN
from models.TransformerForecast import TransformerForecast

from utils.plots import (
    plot_continuous_horizon0,
    plot_one_day,
    plot_scatter_real_vs_pred,
)
from utils.metrics import rmse, mase


def load_split(name):
    df = pd.read_excel(Path("data/Processed") / f"{name}.xlsx", index_col=0)
    y = df["Energy"].to_numpy(dtype=np.float32)
    X = df.drop(columns=["Energy"]).to_numpy(dtype=np.float32)
    return X, y


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    with open("config/timeseries.yaml") as f:
        cfg = yaml.safe_load(f)["model"]

    checkpoint = torch.load("best_model.pth", map_location=device)
    name = checkpoint["model_name"]

    model = {
        "LSTM": LSTM_two_layers,
        "GRU": GRU_two_layers,
        "LSTM_FCN": LSTM_FCN,
        "Transformer": TransformerForecast,
    }[name](
        cfg["input_size"], cfg["hidden_size"], cfg["output_window"], cfg["dropout"]
    )

    model.load_state_dict(checkpoint["state_dict"])
    model.to(device)
    model.eval()

    X, y = load_split("new_test")
    ds = TimeSeriesDataset(X, y, cfg["length"], cfg["lag"], cfg["output_window"], stride=24)
    dl = DataLoader(ds, batch_size=cfg["batch_size"], shuffle=False)

    preds, targets = [], []

    with torch.no_grad():
        for x, t in dl:
            preds.append(model(x.to(device)).cpu().numpy())
            targets.append(t.numpy())

    preds = np.expm1(np.concatenate(preds))
    targets = np.expm1(np.concatenate(targets))

    plot_continuous_horizon0(targets, preds)
    plot_one_day(targets, preds)
    plot_scatter_real_vs_pred(targets, preds)


if __name__ == "__main__":
    main()
