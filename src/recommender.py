"""
PyTorch model that recommends a champion for a given role and team composition.
Each champion is represented as a learned embedding vector — champions with
similar playstyles end up with similar vectors.
These are concatenated with a role embedding and passed through fully
connected layers to output a recommendation.

Input:  role + 4 allies + 5 enemies + lane opponent (encoded as integers)
Output: recommended champion (decoded via label_encoder.pkl)
"""

import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import Dataset, DataLoader

class ChampionRecommender(nn.Module):
    def __init__(self, num_champions, num_roles, champ_embed_dim=16, role_embed_dim=8):
        super().__init__()

        # one shared embedding table for all champion inputs
        self.champ_embedding = nn.Embedding(num_champions, champ_embed_dim)
        self.role_embedding = nn.Embedding(num_roles, role_embed_dim)

        # 10 champion inputs + 1 role input
        input_size = (champ_embed_dim * 10) + role_embed_dim

        self.network = nn.Sequential(
            nn.Linear(input_size, 256), # more space for the model learning
            nn.ReLU(), # removes negatives
            nn.Linear(256, 128), # cut down number for more refinement
            nn.ReLU(),
            nn.Linear(128, num_champions)  # gives a score to every champion - 172
        )

    def forward(self, role, allies, enemies, lane_opponent):
        # allies shape: (batch, 4), enemies shape: (batch, 5)
        role_emb = self.role_embedding(role)                    # (batch, 8)
        ally_emb = self.champ_embedding(allies)                 # (batch, 4, 16), e.g. 64 rows, 4 allies each, 16 numbers each
        enemy_emb = self.champ_embedding(enemies)               # (batch, 5, 16)
        opponent_emb = self.champ_embedding(lane_opponent)      # (batch, 16)

        # flatten ally and enemy embeddings
        ally_emb = ally_emb.view(ally_emb.size(0), -1)         # (batch, 64), -1 could be replaced by 64 to hardcode it
        enemy_emb = enemy_emb.view(enemy_emb.size(0), -1)      # (batch, 80)

        # concatenate everything into one vector
        x = torch.cat([role_emb, ally_emb, enemy_emb, opponent_emb], dim=1)

        return self.network(x)

class ChampionDataset(Dataset):
    def __init__(self, csv_path):
        df = pd.read_csv(csv_path)

        self.roles = torch.tensor(df["role"].values, dtype=torch.long)
        self.allies = torch.tensor(df[["ally_1", "ally_2", "ally_3", "ally_4"]].values, dtype=torch.long)
        self.enemies = torch.tensor(
            df[["enemy_top", "enemy_jungle", "enemy_mid", "enemy_bot", "enemy_support"]].values, dtype=torch.long)
        self.opponents = torch.tensor(df["lane_opponent"].values, dtype=torch.long)
        self.labels = torch.tensor(df["champion_name"].values, dtype=torch.long)

    # returns the number for training rows
    def __len__(self):
        return len(self.roles)

    # returns complete training sample at given index - 1 batch
    def __getitem__(self, idx):
        return (
            self.roles[idx],
            self.allies[idx],
            self.enemies[idx],
            self.opponents[idx],
            self.labels[idx]
        )

# prediction -> measure loss -> backpropagate blame -> nudge weights -> repeat
# lr - learning rate, epochs - 30 passes through the data
def train(epochs=30, batch_size=64, lr=0.001):
    train_dataset = ChampionDataset("../data/processed_train.csv")
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    model = ChampionRecommender(num_champions=172, num_roles=5) # creates model
    criterion = torch.nn.CrossEntropyLoss() # loss function
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # training loop
    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for batch in train_loader:
            role, allies, enemies, opponent, label = batch # unpacks the batch, label is the champ we are trying to predict

            # calls our forward method - returns the 172 scores for champs (vector)
            output = model(role, allies, enemies, opponent)

            # calculate loss
            loss = criterion(output, label)

            # backpropagation
            optimizer.zero_grad() # clears gradients from previous batch
            loss.backward() # calculates how much each weight contributed to the loss (gradients)
            optimizer.step() # adjust the weights

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch + 1}/{epochs} — Loss: {avg_loss:.4f}")

    # save model
    torch.save(model.state_dict(), "../data/model.pth")
    print("Model saved to data/model.pth")

if __name__ == "__main__":
    train()
