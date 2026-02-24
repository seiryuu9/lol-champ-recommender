"""PyTorch Dataset for champion pick/performance data."""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class ChampionDataset(Dataset):
    """Dataset of (player_id, champion_id, score) triples.

    ``score`` is a float representing how well a player performs on a given
    champion (e.g. normalised win-rate, KDA, or any preference signal in
    [0, 1]).

    Parameters
    ----------
    data:
        A :class:`pandas.DataFrame` with at least the columns
        ``player_id``, ``champion_id``, and ``score``.  Integer IDs are
        expected to start from 0 and be contiguous.
    """

    REQUIRED_COLUMNS = {"player_id", "champion_id", "score"}

    def __init__(self, data: pd.DataFrame) -> None:
        missing = self.REQUIRED_COLUMNS - set(data.columns)
        if missing:
            raise ValueError(f"DataFrame is missing columns: {missing}")

        self.player_ids = torch.tensor(
            data["player_id"].to_numpy(dtype=np.int64), dtype=torch.long
        )
        self.champion_ids = torch.tensor(
            data["champion_id"].to_numpy(dtype=np.int64), dtype=torch.long
        )
        self.scores = torch.tensor(
            data["score"].to_numpy(dtype=np.float32), dtype=torch.float32
        )

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def num_players(self) -> int:
        """Number of unique players."""
        return int(self.player_ids.max().item()) + 1

    @property
    def num_champions(self) -> int:
        """Number of unique champions."""
        return int(self.champion_ids.max().item()) + 1

    # ------------------------------------------------------------------
    # Dataset protocol
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.scores)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.player_ids[idx], self.champion_ids[idx], self.scores[idx]
