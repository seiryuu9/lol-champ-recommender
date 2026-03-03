"""
Cleans raw match data collected from the Riot API, encodes
champion names and roles as integers, and splits into train/test sets
ready for PyTorch training.

Input:  data/matches.csv
Output: data/processed_train.csv, data/processed_test.csv, data/label_encoder.pkl
"""

import pandas as pd
from sklearn.preprocessing import LabelEncoder
import pickle
from sklearn.model_selection import train_test_split

# cleaning data
df = pd.read_csv("../data/matches.csv")
print(f"Loaded {len(df):,} rows")

# ensure win column is boolean
df["win"] = df["win"].astype(bool)

# fixes role name
df["role"] = df["role"].replace("UTILITY", "SUPPORT")

# excludes games under 10 mins
df = df[df["game_duration"] >= 600]
print(f"After removing short games: {len(df):,} rows")

# excludes incomplete rows
critical_cols = ["champion_name", "role", "win", "ally_1", "ally_2", "ally_3", "ally_4",
                 "enemy_top", "enemy_jungle", "enemy_mid", "enemy_bot", "enemy_support", "lane_opponent"]

df = df.dropna(subset=critical_cols)
print(f"After dropping missing values: {len(df):,} rows")

# sanity check
print("\nRole distribution:")
print(df["role"].value_counts())


# encoding champion names
champion_cols = ["champion_name", "ally_1", "ally_2", "ally_3", "ally_4",
                 "enemy_top", "enemy_jungle", "enemy_mid", "enemy_bot", "enemy_support", "lane_opponent"]

# get all unique champion names
all_champions = pd.unique(df[champion_cols].values.ravel())
print(f"\nUnique champions found: {len(all_champions)}")

# works alphabetically
le = LabelEncoder()
le.fit(all_champions)

# apply encoding to every champion column - replaces the champ name with its number
for col in champion_cols:
    df[col] = le.transform(df[col])

# save encoder for later use
with open("../data/label_encoder.pkl", "wb") as f:
    pickle.dump(le, f)
print("Label encoder saved to data/label_encoder.pkl")

# sanity check
print(f"\nSample encoded values:")
print(df[["champion_name", "ally_1", "enemy_top"]].head(3))
print(le.inverse_transform([161, 151, 76, 38]))


# encoding roles
role_map = {"TOP": 0, "JUNGLE": 1, "MIDDLE": 2, "BOTTOM": 3, "SUPPORT": 4}
df["role"] = df["role"].map(role_map)

print("\nRole encoding:", role_map)


# train/test split (by unique match IDs)
match_ids = df["match_id"].unique()
train_ids, test_ids = train_test_split(match_ids, test_size=0.2, random_state=42)

train_df = df[df["match_id"].isin(train_ids)]
test_df = df[df["match_id"].isin(test_ids)]

print(f"\nTrain rows: {len(train_df):,}")
print(f"Test rows:  {len(test_df):,}")

train_df.to_csv("../data/processed_train.csv", index=False)
test_df.to_csv("../data/processed_test.csv", index=False)
print("\nSaved processed_train.csv and processed_test.csv")