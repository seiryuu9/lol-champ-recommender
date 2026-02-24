"""Neural Collaborative Filtering model for champion recommendation."""

from __future__ import annotations

import torch
import torch.nn as nn


class ChampionRecommender(nn.Module):
    """Neural Collaborative Filtering model.

    Learns latent embeddings for players and champions, then predicts a
    preference score via a small multi-layer perceptron (MLP).

    Parameters
    ----------
    num_players:
        Total number of distinct players in the dataset.
    num_champions:
        Total number of distinct champions in the dataset.
    embedding_dim:
        Dimensionality of the player / champion embedding vectors.
    hidden_dims:
        Sizes of the hidden layers of the MLP that follows concatenation of
        the two embeddings.
    dropout:
        Dropout probability applied after each hidden layer.
    """

    def __init__(
        self,
        num_players: int,
        num_champions: int,
        embedding_dim: int = 32,
        hidden_dims: tuple[int, ...] = (64, 32),
        dropout: float = 0.2,
    ) -> None:
        super().__init__()

        self.player_embedding = nn.Embedding(num_players, embedding_dim)
        self.champion_embedding = nn.Embedding(num_champions, embedding_dim)

        # Build MLP: input is the concatenation of both embeddings.
        layers: list[nn.Module] = []
        in_features = embedding_dim * 2
        for out_features in hidden_dims:
            layers.extend(
                [
                    nn.Linear(in_features, out_features),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                ]
            )
            in_features = out_features
        layers.append(nn.Linear(in_features, 1))
        layers.append(nn.Sigmoid())

        self.mlp = nn.Sequential(*layers)

        self._init_weights()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _init_weights(self) -> None:
        nn.init.normal_(self.player_embedding.weight, std=0.01)
        nn.init.normal_(self.champion_embedding.weight, std=0.01)
        for module in self.mlp.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def forward(
        self, player_ids: torch.Tensor, champion_ids: torch.Tensor
    ) -> torch.Tensor:
        """Predict preference scores.

        Parameters
        ----------
        player_ids:
            1-D ``LongTensor`` of shape ``(batch,)``.
        champion_ids:
            1-D ``LongTensor`` of shape ``(batch,)``.

        Returns
        -------
        torch.Tensor
            Float tensor of shape ``(batch,)`` with values in ``(0, 1)``.
        """
        player_emb = self.player_embedding(player_ids)
        champ_emb = self.champion_embedding(champion_ids)
        x = torch.cat([player_emb, champ_emb], dim=-1)
        return self.mlp(x).squeeze(-1)
