"""Tests for ChampionDataset."""

import pytest
import pandas as pd

from lol_champ_recommender.dataset import ChampionDataset


def _make_df(n: int = 10, num_players: int = 3, num_champions: int = 5) -> pd.DataFrame:
    import numpy as np

    rng = np.random.default_rng(0)
    return pd.DataFrame(
        {
            "player_id": rng.integers(0, num_players, size=n),
            "champion_id": rng.integers(0, num_champions, size=n),
            "score": rng.uniform(0, 1, size=n).astype("float32"),
        }
    )


class TestChampionDataset:
    def test_len(self):
        df = _make_df(10)
        ds = ChampionDataset(df)
        assert len(ds) == 10

    def test_getitem_types(self):
        import torch

        ds = ChampionDataset(_make_df(5))
        player_id, champ_id, score = ds[0]
        assert player_id.dtype == torch.long
        assert champ_id.dtype == torch.long
        assert score.dtype == torch.float32

    def test_num_players_and_champions(self):
        df = pd.DataFrame(
            {
                "player_id": [0, 1, 2],
                "champion_id": [0, 1, 4],
                "score": [0.5, 0.7, 0.3],
            }
        )
        ds = ChampionDataset(df)
        assert ds.num_players == 3
        assert ds.num_champions == 5

    def test_missing_columns_raises(self):
        with pytest.raises(ValueError, match="missing columns"):
            ChampionDataset(pd.DataFrame({"player_id": [0], "score": [0.5]}))
