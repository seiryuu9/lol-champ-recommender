"""
Collects ranked match data from the Riot API for Diamond+ EUW players.
Saves 10 rows per match (one per participant) with champion, role, team comp, and performance stats.
Supports checkpointing — can be stopped and resumed without losing progress.

Output: data/matches.csv
"""

import requests
import time
import pandas as pd
import os
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.getenv("RIOT_API_KEY")
HEADERS = {"X-Riot-Token": API_KEY}
REGION = "europe"
PLATFORM = "euw1"

# config
TARGET_MATCHES = 40_000
MATCHES_PER_PLAYER = 50
SLEEP = 1.2  # seconds between requests (keeps you under rate limit)

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data") # create data folder at root

CHECKPOINT_FILE = os.path.join(DATA_DIR, "matches.csv")
SEEN_FILE = os.path.join(DATA_DIR, "seen_matches.txt")
PLAYERS_FILE = os.path.join(DATA_DIR, "seen_players.txt")


def get_players_by_tier(tier, division, page=1):
    url = (
        f"https://{PLATFORM}.api.riotgames.com/lol/league/v4/entries/"
        f"RANKED_SOLO_5x5/{tier}/{division}?page={page}"
    )
    r = requests.get(url, headers=HEADERS)
    if r.status_code != 200:
        print(f"  Error fetching {tier} {division} page {page}: {r.status_code}")
        return []
    return [e["puuid"] for e in r.json() if "puuid" in e]


def get_challenger_puuids():
    url = f"https://{PLATFORM}.api.riotgames.com/lol/league/v4/challengerleagues/by-queue/RANKED_SOLO_5x5"
    r = requests.get(url, headers=HEADERS)
    entries = [e for e in r.json().get("entries", []) if "puuid" in e]
    return [e["puuid"] for e in entries]


def get_grandmaster_puuids():
    url = f"https://{PLATFORM}.api.riotgames.com/lol/league/v4/grandmasterleagues/by-queue/RANKED_SOLO_5x5"
    r = requests.get(url, headers=HEADERS)
    entries = [e for e in r.json().get("entries", []) if "puuid" in e]
    return [e["puuid"] for e in entries]


def get_master_puuids():
    url = f"https://{PLATFORM}.api.riotgames.com/lol/league/v4/masterleagues/by-queue/RANKED_SOLO_5x5"
    r = requests.get(url, headers=HEADERS)
    entries = [e for e in r.json().get("entries", []) if "puuid" in e]
    return [e["puuid"] for e in entries]


def get_match_ids(puuid, count=50):
    url = (
        f"https://{REGION}.api.riotgames.com/lol/match/v5/matches/by-puuid/"
        f"{puuid}/ids?count={count}&queue=420"
    )
    r = requests.get(url, headers=HEADERS)
    if r.status_code != 200:
        return []
    return r.json()


def get_match_data(match_id):
    url = f"https://{REGION}.api.riotgames.com/lol/match/v5/matches/{match_id}"
    r = requests.get(url, headers=HEADERS)
    if r.status_code != 200:
        return None
    return r.json()


def extract_participants(match_data):
    info = match_data["info"]
    match_id = match_data["metadata"]["matchId"]
    game_version = info.get("gameVersion", "")
    game_duration = info.get("gameDuration", 0)

    # build team composition maps so each row knows its allies and enemies
    # team_id 100 = blue, 200 = red
    team_champs = {100: [], 200: []}
    team_roles  = {100: {}, 200: {}}  # role -> champion

    for p in info["participants"]:
        tid  = p["teamId"]
        role = p["teamPosition"]
        champ = p["championName"]
        team_champs[tid].append(champ)
        if role:
            team_roles[tid][role] = champ

    rows = []
    for p in info["participants"]:
        tid   = p["teamId"]
        enemy_tid = 200 if tid == 100 else 100
        role  = p["teamPosition"]

        # ally champions (excluding self)
        allies = [c for c in team_champs[tid] if c != p["championName"]]

        # enemy champion in the same role (direct lane opponent)
        lane_opponent = team_roles[enemy_tid].get(role, "")

        rows.append({
            "match_id":         match_id,
            "puuid":            p["puuid"],
            "game_version":     game_version,   # patch filtering later
            "game_duration":    game_duration,

            "champion_name":    p["championName"],
            "role":             role,
            "team_id":          tid,             # 100=blue, 200=red

            "win":              p["win"],

            "kills":            p["kills"],
            "deaths":           p["deaths"],
            "assists":          p["assists"],
            "cs":               p["totalMinionsKilled"],
            "gold_earned":      p["goldEarned"],
            "total_damage":     p["totalDamageDealtToChampions"],
            "vision_score":     p["visionScore"],
            "summoner1_id":     p["summoner1Id"],  # flash/ignite/tp — signals playstyle
            "summoner2_id":     p["summoner2Id"],

            "ally_1":           allies[0] if len(allies) > 0 else "",
            "ally_2":           allies[1] if len(allies) > 1 else "",
            "ally_3":           allies[2] if len(allies) > 2 else "",
            "ally_4":           allies[3] if len(allies) > 3 else "",

            "enemy_top":        team_roles[enemy_tid].get("TOP", ""),
            "enemy_jungle":     team_roles[enemy_tid].get("JUNGLE", ""),
            "enemy_mid":        team_roles[enemy_tid].get("MIDDLE", ""),
            "enemy_bot":        team_roles[enemy_tid].get("BOTTOM", ""),
            "enemy_support":    team_roles[enemy_tid].get("UTILITY", ""),
            "lane_opponent":    lane_opponent,   # direct counter matchup
        })

    return rows


def load_checkpoints():
    all_rows = []
    seen_matches = set()
    seen_players = set()

    os.makedirs(DATA_DIR, exist_ok=True)

    if os.path.exists(CHECKPOINT_FILE):
        df = pd.read_csv(CHECKPOINT_FILE)
        all_rows = df.to_dict("records")
        print(f"  Loaded {len(all_rows):,} existing rows from checkpoint")

    if os.path.exists(SEEN_FILE):
        with open(SEEN_FILE, "r") as f:
            seen_matches = set(f.read().splitlines())
        print(f"  Skipping {len(seen_matches):,} already collected matches")

    if os.path.exists(PLAYERS_FILE):
        with open(PLAYERS_FILE, "r") as f:
            seen_players = set(f.read().splitlines())
        print(f"  Skipping {len(seen_players):,} already processed players")

    return all_rows, seen_matches, seen_players


def save_checkpoints(all_rows, seen_matches, seen_players):
    df = pd.DataFrame(all_rows)
    df.to_csv(CHECKPOINT_FILE, index=False)

    with open(SEEN_FILE, "w") as f:
        f.write("\n".join(seen_matches))

    with open(PLAYERS_FILE, "w") as f:
        f.write("\n".join(seen_players))


def collect():
    print("=" * 55)
    print("  LoL Match Collector — Diamond+ EUW")
    print(f"  Target: {TARGET_MATCHES:,} matches")
    print("=" * 55)

    print("\nLoading checkpoints...")
    all_rows, seen_matches, seen_players = load_checkpoints()
    collected_matches = len(seen_matches)

    if collected_matches >= TARGET_MATCHES:
        print(f"\nAlready reached target ({collected_matches:,} matches). Done!")
        return

    print("\nFetching player lists...")
    summoner_ids = []

    print("  Fetching Challenger...")
    summoner_ids += get_challenger_puuids()
    time.sleep(SLEEP)

    print("  Fetching Grandmaster...")
    summoner_ids += get_grandmaster_puuids()
    time.sleep(SLEEP)

    print("  Fetching Master...")
    summoner_ids += get_master_puuids()
    time.sleep(SLEEP)

    # Diamond I and II for extra volume
    for division in ["I", "II"]:
        for page in range(1, 4):  # 3 pages each
            print(f"  Fetching Diamond {division} page {page}...")
            summoner_ids += get_players_by_tier("DIAMOND", division, page)
            time.sleep(SLEEP)

    # Deduplicate
    summoner_ids = list(set(summoner_ids))
    print(f"\nTotal players fetched: {len(summoner_ids):,}")

    print(f"\nStarting collection (target: {TARGET_MATCHES:,} matches)...\n")

    for i, sid in enumerate(summoner_ids):
        if collected_matches >= TARGET_MATCHES:
            print(f"\nTarget reached! {collected_matches:,} matches collected.")
            break

        if sid in seen_players:
            continue

        try:
            puuid = sid

            match_ids = get_match_ids(puuid, count=MATCHES_PER_PLAYER)
            time.sleep(SLEEP)

            new_ids = [m for m in match_ids if m not in seen_matches]

            for match_id in new_ids:
                if collected_matches >= TARGET_MATCHES:
                    break
                try:
                    match = get_match_data(match_id)
                    time.sleep(SLEEP)

                    if not match:
                        continue

                    rows = extract_participants(match)
                    all_rows.extend(rows)
                    seen_matches.add(match_id)
                    collected_matches += 1

                except Exception as e:
                    print(f"  Match error ({match_id}): {e}")
                    time.sleep(5)

            seen_players.add(sid)

            save_checkpoints(all_rows, seen_matches, seen_players)

            print(
                f"[{collected_matches:,}/{TARGET_MATCHES:,}] "
                f"Player {i+1}/{len(summoner_ids)} done — "
                f"{len(all_rows):,} total rows"
            )

        except KeyboardInterrupt:
            print("\n\nStopped by user. Saving progress...")
            save_checkpoints(all_rows, seen_matches, seen_players)
            print(f"Progress saved. {collected_matches:,} matches collected so far.")
            print("Run the script again to resume.")
            return

        except Exception as e:
            print(f"  Player error ({sid}): {e}")
            time.sleep(5)
            continue

    save_checkpoints(all_rows, seen_matches, seen_players)
    print("\n" + "=" * 55)
    print(f"  Collection complete!")
    print(f"  Matches collected : {collected_matches:,}")
    print(f"  Total rows        : {len(all_rows):,}")
    print(f"  Saved to          : {CHECKPOINT_FILE}")
    print("=" * 55)


if __name__ == "__main__":
    collect()