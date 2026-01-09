import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split

SCRIPTS_ROOT = Path(__file__).resolve().parents[1]
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

from _paths import data_dir, outputs_dir, resolve_path  # noqa: E402


def main():
    default_data = resolve_path(data_dir() / "scoring_results.csv")
    default_output = outputs_dir() / "models" / "fly_score_model.pth"

    ap = argparse.ArgumentParser(description="Train a simple PyTorch regressor on fly behavior metrics.")
    ap.add_argument(
        "-d",
        "--data",
        default=str(default_data),
        help="Labeled CSV from label_videos.py",
    )
    ap.add_argument("-e","--epochs", type=int, default=100, help="Training epochs")
    ap.add_argument(
        "-o",
        "--output-model",
        default=str(default_output),
        help="Where to save model",
    )
    ap.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    args = ap.parse_args()

    df = pd.read_csv(args.data)
    if "combined_score" not in df.columns:
        print("CSV must contain 'combined_score'.")
        return

    drop_cols = {"fly_id","trial_id","video_file","csv_file","user_score","data_score","combined_score"}
    feature_cols = [c for c in df.columns if c not in drop_cols and np.issubdtype(df[c].dtype, np.number)]
    if not feature_cols:
        print("No numeric feature columns found.")
        return

    X = df[feature_cols].astype(np.float32).values
    y = df["combined_score"].astype(np.float32).values.reshape(-1,1)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X_train = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_train = torch.tensor(y_train, dtype=torch.float32).to(device)
    X_test  = torch.tensor(X_test, dtype=torch.float32).to(device)
    y_test  = torch.tensor(y_test, dtype=torch.float32).to(device)

    model = nn.Sequential(
        nn.Linear(X_train.shape[1], 32),
        nn.ReLU(),
        nn.Linear(32, 16),
        nn.ReLU(),
        nn.Linear(16, 1)
    ).to(device)

    opt = optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.MSELoss()

    model.train()
    for ep in range(1, args.epochs+1):
        opt.zero_grad()
        pred = model(X_train)
        loss = loss_fn(pred, y_train)
        loss.backward()
        opt.step()
        if ep % 10 == 0 or ep == args.epochs:
            print(f"Epoch {ep}/{args.epochs} - train MSE: {loss.item():.4f}")

    model.eval()
    with torch.no_grad():
        pred = model(X_test)
        mse = loss_fn(pred, y_test).item()
        mae = torch.mean(torch.abs(pred - y_test)).item()
    print(f"Test MSE: {mse:.4f}, MAE: {mae:.4f}")

    output_path = Path(args.output_model)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model, output_path)
    print(f"Saved model to {output_path}")

if __name__ == "__main__":
    main()
