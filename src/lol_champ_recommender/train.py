"""Training loop for ChampionRecommender."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

from .dataset import ChampionDataset
from .model import ChampionRecommender


def train(
    dataset: ChampionDataset,
    *,
    embedding_dim: int = 32,
    hidden_dims: tuple[int, ...] = (64, 32),
    dropout: float = 0.2,
    lr: float = 1e-3,
    epochs: int = 10,
    batch_size: int = 256,
    val_split: float = 0.1,
    device: str | torch.device | None = None,
) -> ChampionRecommender:
    """Train a :class:`ChampionRecommender` on *dataset*.

    Parameters
    ----------
    dataset:
        A :class:`ChampionDataset` instance.
    embedding_dim:
        Dimensionality of embedding vectors.
    hidden_dims:
        Hidden layer sizes for the MLP.
    dropout:
        Dropout probability.
    lr:
        Learning rate for the Adam optimiser.
    epochs:
        Number of full passes over the training data.
    batch_size:
        Mini-batch size.
    val_split:
        Fraction of data reserved for validation.
    device:
        Device to run training on.  Defaults to CUDA if available, else CPU.

    Returns
    -------
    ChampionRecommender
        The trained model (moved to CPU and set to eval mode).
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)

    # Split into train / val
    val_size = max(1, int(len(dataset) * val_split))
    train_size = len(dataset) - val_size
    train_set, val_set = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)

    model = ChampionRecommender(
        num_players=dataset.num_players,
        num_champions=dataset.num_champions,
        embedding_dim=embedding_dim,
        hidden_dims=hidden_dims,
        dropout=dropout,
    ).to(device)

    criterion = nn.BCELoss()
    optimiser = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(1, epochs + 1):
        # ---- training ------------------------------------------------
        model.train()
        train_loss = 0.0
        for player_ids, champ_ids, scores in train_loader:
            player_ids = player_ids.to(device)
            champ_ids = champ_ids.to(device)
            scores = scores.to(device)

            optimiser.zero_grad()
            preds = model(player_ids, champ_ids)
            loss = criterion(preds, scores)
            loss.backward()
            optimiser.step()
            train_loss += loss.item() * len(scores)

        train_loss /= train_size

        # ---- validation ----------------------------------------------
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for player_ids, champ_ids, scores in val_loader:
                player_ids = player_ids.to(device)
                champ_ids = champ_ids.to(device)
                scores = scores.to(device)
                preds = model(player_ids, champ_ids)
                val_loss += criterion(preds, scores).item() * len(scores)
        val_loss /= val_size

        print(
            f"Epoch {epoch:3d}/{epochs}  "
            f"train_loss={train_loss:.4f}  val_loss={val_loss:.4f}"
        )

    model.to("cpu").eval()
    return model
