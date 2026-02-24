# lol-champ-recommender

A League of Legends champion recommender built with **PyTorch**.

The model uses Neural Collaborative Filtering (NCF) — player and champion
embeddings are concatenated and fed through a small MLP that predicts a
preference score for each (player, champion) pair.

## Project structure

```
src/lol_champ_recommender/
    __init__.py      – public API
    dataset.py       – PyTorch Dataset (player_id, champion_id, score)
    model.py         – ChampionRecommender nn.Module
    train.py         – training loop helper
    recommend.py     – top-k inference helper
tests/
    test_dataset.py
    test_model.py
```

## Installation

```bash
pip install -r requirements.txt
pip install -e .
```

## Quick start

```python
import pandas as pd
from lol_champ_recommender import ChampionDataset, recommend_champions
from lol_champ_recommender.train import train

# Build a dataset from your data
df = pd.read_csv("champion_scores.csv")   # columns: player_id, champion_id, score
dataset = ChampionDataset(df)

# Train the model
model = train(dataset, epochs=20, embedding_dim=32)

# Get top-5 recommendations for player 42
recs = recommend_champions(model, player_id=42, num_champions=dataset.num_champions)
for champ_id, score in recs:
    print(f"champion {champ_id}: {score:.3f}")
```

## Running tests

```bash
pytest tests/
```