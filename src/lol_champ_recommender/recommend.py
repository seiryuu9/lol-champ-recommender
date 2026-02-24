"""Champion recommendation inference helpers."""

from __future__ import annotations

import torch

from .model import ChampionRecommender


def recommend_champions(
    model: ChampionRecommender,
    player_id: int,
    num_champions: int,
    *,
    top_k: int = 5,
    exclude: list[int] | None = None,
) -> list[tuple[int, float]]:
    """Return the top-*k* champion recommendations for a player.

    Parameters
    ----------
    model:
        A trained :class:`ChampionRecommender` (in eval mode).
    player_id:
        The integer ID of the player to generate recommendations for.
    num_champions:
        Total number of champions the model was trained on.
    top_k:
        How many recommendations to return.
    exclude:
        Optional list of champion IDs to exclude (e.g. already picked).

    Returns
    -------
    list[tuple[int, float]]
        Sorted list of ``(champion_id, score)`` pairs, highest score first.
    """
    exclude_set: set[int] = set(exclude or [])
    candidate_ids = [i for i in range(num_champions) if i not in exclude_set]

    if not candidate_ids:
        return []

    model.eval()
    with torch.no_grad():
        player_tensor = torch.tensor(
            [player_id] * len(candidate_ids), dtype=torch.long
        )
        champ_tensor = torch.tensor(candidate_ids, dtype=torch.long)
        scores = model(player_tensor, champ_tensor).tolist()

    ranked = sorted(zip(candidate_ids, scores), key=lambda x: x[1], reverse=True)
    return ranked[:top_k]
