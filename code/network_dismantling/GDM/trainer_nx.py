"""
GDM training script with per-epoch subsampling for large datasets.
"""
import argparse
import logging
import random
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch

from network_dismantling.GDM.models.GAT import GAT_Model
from network_dismantling.GDM.dataset_providers_nx import load_npz_dataset

logger = logging.getLogger(__name__)


class SimpleArgs:
    def __init__(self, features, conv_layers, heads, fc_layers,
                 concat=None, negative_slope=None, dropout=None, bias=None,
                 seed_train=0):
        self.features = features
        self.num_features = len(features)
        self.conv_layers = conv_layers
        self.heads = heads
        self.fc_layers = fc_layers
        self.concat = concat if concat is not None else [True] * len(conv_layers)
        self.negative_slope = negative_slope if negative_slope is not None else [0.2] * len(conv_layers)
        self.dropout = dropout if dropout is not None else [0.3] * len(conv_layers)
        self.bias = bias if bias is not None else [True] * len(conv_layers)
        self.seed_train = seed_train


def train_model(
    model: GAT_Model,
    train_data: List,
    val_data: List,
    epochs: int = 100,
    lr: float = 0.003,
    weight_decay: float = 1e-5,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    seed: int = 0,
    subsample: int = 800,
) -> Tuple[GAT_Model, List[float], List[float]]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=max(1, epochs // 3), gamma=0.5)
    loss_op = torch.nn.MSELoss()

    best_val_loss = float("inf")
    best_state = None
    train_losses = []
    val_losses = []

    for epoch in range(1, epochs + 1):
        # Subsample training data each epoch
        if subsample and subsample < len(train_data):
            epoch_train = random.sample(train_data, subsample)
        else:
            epoch_train = train_data

        model.train()
        total_loss = 0.0
        count = 0
        for data in epoch_train:
            data = data.to(device)
            optimizer.zero_grad()
            pred = model(data.x, data.edge_index)
            loss = loss_op(pred, data.y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            count += 1

        scheduler.step()
        avg_train_loss = total_loss / max(count, 1)
        train_losses.append(avg_train_loss)

        model.eval()
        val_loss = 0.0
        val_count = 0
        with torch.no_grad():
            for data in val_data:
                data = data.to(device)
                pred = model(data.x, data.edge_index)
                loss = loss_op(pred, data.y)
                val_loss += loss.item()
                val_count += 1

        avg_val_loss = val_loss / max(val_count, 1)
        val_losses.append(avg_val_loss)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if epoch % 10 == 0 or epoch == 1:
            logger.info(f"Epoch {epoch:03d}: train_loss={avg_train_loss:.6f}, val_loss={avg_val_loss:.6f}, lr={scheduler.get_last_lr()[0]:.6f}")

    if best_state is not None:
        model.load_state_dict(best_state)
        model = model.to(device)

    return model, train_losses, val_losses


def grid_search_train(
    data_dir: str,
    features: List[str],
    param_configs: List[dict],
    epochs: int = 100,
    lr: float = 0.003,
    weight_decay: float = 1e-5,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    seed: int = 0,
    val_split: float = 0.10,
    subsample: int = 800,
) -> Tuple[GAT_Model, dict, Tuple[List[float], List[float]]]:
    logger.info(f"Loading training data from {data_dir}")
    train_data, val_data = load_npz_dataset(data_dir, val_split=val_split, seed=seed)
    logger.info(f"Train graphs: {len(train_data)}, Val graphs: {len(val_data)}")

    if len(train_data) == 0:
        raise RuntimeError("No training data found!")

    best_model = None
    best_config = None
    best_val_loss = float("inf")
    best_history = None

    for cfg in param_configs:
        logger.info(f"Training config: {cfg}")
        args = SimpleArgs(features=features, **cfg)
        try:
            model = GAT_Model(args)
        except Exception as e:
            logger.warning(f"Failed to create model with config {cfg}: {e}")
            continue

        model, train_hist, val_hist = train_model(
            model, train_data, val_data,
            epochs=epochs, lr=lr, weight_decay=weight_decay,
            device=device, seed=seed, subsample=subsample
        )

        final_val_loss = min(val_hist)
        logger.info(f"Config {cfg} -> best val_loss={final_val_loss:.6f}")

        if final_val_loss < best_val_loss:
            best_val_loss = final_val_loss
            best_model = model
            best_config = cfg
            best_history = (train_hist, val_hist)

    logger.info(f"Best config: {best_config}, val_loss={best_val_loss:.6f}")
    return best_model, best_config, best_history


def main():
    parser = argparse.ArgumentParser(description="Train GDM (networkx version)")
    parser.add_argument("--data_dir", type=str, default="network_dismantling/GDM/dataset/full_train")
    parser.add_argument("--output_model", type=str, default="network_dismantling/GDM/models_newpg/gdm_nx_best.pth")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=0.003)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--subsample", type=int, default=800, help="Subsample training graphs per epoch")
    parser.add_argument("--features", nargs="+", default=["degree", "clustering_coefficient", "kcore", "chi_degree"])
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    param_configs = [
        {"conv_layers": [10], "heads": [1], "fc_layers": [50]},
        {"conv_layers": [20, 20], "heads": [1, 1], "fc_layers": [50, 30]},
        {"conv_layers": [5, 5, 5], "heads": [1, 1, 1], "fc_layers": [100]},
        {"conv_layers": [30, 20], "heads": [1, 1], "fc_layers": [50, 30]},
        {"conv_layers": [40, 30, 20], "heads": [1, 1, 1], "fc_layers": [100]},
    ]

    best_model, best_config, _ = grid_search_train(
        data_dir=args.data_dir,
        features=args.features,
        param_configs=param_configs,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        device=args.device,
        seed=args.seed,
        subsample=args.subsample,
    )

    out_path = Path(args.output_model)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_state_dict": best_model.state_dict(),
        "config": best_config,
        "features": args.features,
    }, out_path)
    logger.info(f"Saved best model to {out_path}")


if __name__ == "__main__":
    main()
