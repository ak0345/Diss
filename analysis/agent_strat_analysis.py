import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df_games = []
for c in range(1000):
    df = pd.read_csv(r"C:\Users\foosh\OneDrive\Desktop\projects\DIss\games_ep_diff\epsilonagent_epsilonagent_game_"+f"{c+1}.csv")
    df_games.append(df)

# Each DataFrame has columns: id, obs, action, reward, winner, current_player.
# Here we combine them into one DataFrame for analysis.
all_games = pd.concat(
    [df.assign(game_id=i) for i, df in enumerate(df_games, start=1)],
    ignore_index=True
)

# Convert the winner column to numeric (if not already), and fill missing winner values with -1.
all_games["winner"] = pd.to_numeric(all_games["winner"], errors="coerce").fillna(-1).astype(int)

# Convert the action column to numeric if it isn't already.
all_games["action"] = pd.to_numeric(all_games["action"], errors="coerce")

# Create a summary for each game:
game_summary = all_games.groupby("game_id").agg(
    moves=("id", "max"),
    total_reward=("reward", "sum"),
    final_winner=("winner", lambda x: x.iloc[-1])  # assuming the last move's winner value indicates the game outcome
).reset_index()

player_summary = all_games.groupby(["game_id", "current_player"]).agg(
    moves=("id", "count"),
    total_reward=("reward", "sum")
).reset_index()

# Plot 0: Distribution of total reward for each player
colors = {0: 'blue', 1: 'orange'}

# Loop through each player, filter the data, and plot.
for player, color in colors.items():
    player_df = player_summary[player_summary["current_player"] == player]
    plt.scatter(player_df["moves"], player_df["total_reward"],
                color=color, label=f"Player {player + 1}")

plt.xlabel("Total Moves in Game")
plt.ylabel("Total Reward in Game")
plt.title("Distribution of Total Reward for Each Player")
plt.legend(title="Players")
plt.show()


# Plot 1: Distribution of Game Lengths (number of moves per game)
plt.figure(figsize=(8, 5))
plt.hist(game_summary["moves"], bins=20, edgecolor="black")
plt.xlabel("Number of Moves")
plt.ylabel("Frequency")
plt.title("Distribution of Game Lengths")
plt.show()

# Plot 2: Distribution of Actions Across All Moves
plt.figure(figsize=(8, 5))
action_counts = all_games["action"].value_counts().sort_index()
action_counts.plot(kind="bar", edgecolor="black")
plt.xlabel("Action")
plt.ylabel("Count")
plt.title("Distribution of Actions")
plt.show()

# Plot 3: Winning Frequency (only for games that ended with a win)
finished_games = game_summary[game_summary["final_winner"] > -1]
winner_counts = finished_games["final_winner"].value_counts().sort_index()
plt.figure(figsize=(8, 5))
winner_counts.plot(kind="bar", edgecolor="black")
plt.xlabel("Winner (Player Number)")
plt.ylabel("Number of Wins")
plt.title("Winning Frequency")
plt.show()

# Plot the cumulative reward over moves for first 5 games
for i in range(5):
    selected_game = all_games[all_games["game_id"] == (i+1)].copy()
    selected_game["cumulative_reward"] = selected_game["reward"].cumsum()

    plt.figure(figsize=(8, 5))
    plt.plot(selected_game["id"], selected_game["cumulative_reward"], marker="o")
    plt.xlabel("Move ID")
    plt.ylabel("Cumulative Reward")
    plt.title(f"Cumulative Reward over Moves (Game {i+1})")
    plt.show()


