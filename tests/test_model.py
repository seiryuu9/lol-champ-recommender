"""Tests for ChampionRecommender model and recommend_champions helper."""

import torch
import pandas as pd
import pytest

from lol_champ_recommender.model import ChampionRecommender
from lol_champ_recommender.dataset import ChampionDataset
from lol_champ_recommender.recommend import recommend_champions


NUM_PLAYERS = 4
NUM_CHAMPIONS = 6


@pytest.fixture()
def model() -> ChampionRecommender:
    return ChampionRecommender(
        num_players=NUM_PLAYERS,
        num_champions=NUM_CHAMPIONS,
        embedding_dim=8,
        hidden_dims=(16,),
        dropout=0.0,
    ).eval()


class TestChampionRecommender:
    def test_output_shape(self, model):
        batch = 5
        player_ids = torch.randint(0, NUM_PLAYERS, (batch,))
        champ_ids = torch.randint(0, NUM_CHAMPIONS, (batch,))
        scores = model(player_ids, champ_ids)
        assert scores.shape == (batch,)

    def test_output_range(self, model):
        player_ids = torch.randint(0, NUM_PLAYERS, (20,))
        champ_ids = torch.randint(0, NUM_CHAMPIONS, (20,))
        scores = model(player_ids, champ_ids)
        assert (scores >= 0).all() and (scores <= 1).all()

    def test_single_sample(self, model):
        p = torch.tensor([0])
        c = torch.tensor([0])
        score = model(p, c)
        assert score.shape == (1,)
        assert 0 <= score.item() <= 1


class TestRecommendChampions:
    def test_returns_top_k(self, model):
        recs = recommend_champions(model, player_id=0, num_champions=NUM_CHAMPIONS, top_k=3)
        assert len(recs) == 3

    def test_sorted_descending(self, model):
        recs = recommend_champions(model, player_id=0, num_champions=NUM_CHAMPIONS, top_k=4)
        scores = [s for _, s in recs]
        assert scores == sorted(scores, reverse=True)

    def test_exclude(self, model):
        exclude = [0, 1, 2]
        recs = recommend_champions(
            model, player_id=0, num_champions=NUM_CHAMPIONS, top_k=10, exclude=exclude
        )
        champ_ids = [c for c, _ in recs]
        for ex in exclude:
            assert ex not in champ_ids

    def test_empty_when_all_excluded(self, model):
        recs = recommend_champions(
            model,
            player_id=0,
            num_champions=NUM_CHAMPIONS,
            top_k=5,
            exclude=list(range(NUM_CHAMPIONS)),
        )
        assert recs == []
