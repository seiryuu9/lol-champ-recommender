# LoL Champion Recommender

 A PyTorch champion recommendation system for League of Legends, trained on 40,000 ranked matches (Diamond+ elo), pulled directly from the Riot Games API.
The model's job: given a role + 4 allies + 5 enemies + lane opponent, predict which champion to play based on ally synergies and enemy counters.  
 
 ---

 <br>
 
## How to run

> Please note that the data is not included in this repository. Collecting 40,000 matches directly from the Riot API takes tens of hours due to Riot's rate limits.  

### 1. Install dependencies

```bash
pip install -r requirements.txt
```
### 2. Get a Riot API key
- Go to [developer.riotgames.com](https://developer.riotgames.com)
- Log in and generate your API key (note that it expires every 24 hours - you need to regenerate it)
  

### 3. Set up your `.env` file in the project root `lol-champ-recommender/`
Inside should look like this:

```
RIOT_API_KEY=RGAPI-your-key-here
```

### 4. Collect match data
Run the script at `src/collect_matches.py`

This script targets 40,000 matches. Progress is automatically saved after every player — you can stop and resume at any time without losing data. Your data will be saved at `/data`.

### 5. Clean match data
Run the script at `src/prepare_data.py`
This script normalizes data and prepares it for further use. We also encode the champions and roles. Lastly we split the set into test/training data.

### 6. Model
Run the script at `src/recommender.py`
This is where the model actually trains to be able to recommend champions. It runs through the training data 30 times, printing loss each epoch, and saves the learned
model weights.