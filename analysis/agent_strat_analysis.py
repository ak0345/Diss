import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import glob
import sys
import re


# Plot 1: Distribution of Game Lengths (number of moves per game) 

# Plot 2: Distribution of Actions Across All Moves

# Plot 3: Winning Frequency (including draws)

# Plot 4: Resource Progression

# Plot 5: Trade Analysis

# Plot 6: Victory Points Distribution at W/L/D

# Plot 7: Biome Heatmap

# Plot 8: Longest Road vs Winner

def extract_real_parts(value):
    if not isinstance(value, str):
        return value
    
    if '(' in value:
        # Use regex to extract all real parts before +0j
        pattern = r'\(([+-]?\d+\.?\d*)(?:[+-].*?j)\)'
        matches = re.findall(pattern, value)
        if matches:
            # Return the first match as a float
            return float(matches[0])
    
    # If not a complex string or extraction failed, return original
    return value

# Read the CSV files with error handling
def safe_read_csv(file_path):
    try:
        # First attempt: read without type conversion
        df = pd.read_csv(file_path)
        
        # Apply conversion to any columns that might have complex numbers
        numeric_cols = ['reward', 'action']  # Add other columns if needed
        for col in numeric_cols:
            if col in df.columns:
                df[col] = df[col].apply(extract_real_parts)
        
        return df
    
    except ValueError as e:
        print(f"Handling error in {file_path}: {e}")
        
        # Second attempt: read with explicit converters
        converters = {
            'reward': extract_real_parts,
            'action': extract_real_parts
            # Add other columns as needed
        }
        
        try:
            df = pd.read_csv(file_path, converters=converters)
            return df
        except Exception as e2:
            print(f"Failed to read {file_path} even with converters: {e2}")
            # Return empty DataFrame as a fallback
            return pd.DataFrame()

# Convert the obs column to expanded features
def expand_obs(row):
    """Expand observation vector into separate columns"""
    try:
        # Convert string representation to list if needed
        if isinstance(row['obs'], str):
            obs = eval(row['obs'])  # Convert string representation to actual list
        else:
            obs = row['obs']
        
        # If obs is None or empty, return the original row
        if obs is None or len(obs) == 0:
            return row
        
        # Convert to numpy array if it's not already
        if not isinstance(obs, np.ndarray):
            obs = np.array(obs)
        
        num_players = 2  # Assuming 2 players
        
        # Parse observation vector according to structure
        idx = 0
        inventories = obs[idx:(idx + num_players * 4)].reshape(num_players, 4)
        for p in range(num_players):
            row[f'p{p+1}_wood'] = inventories[p, 0]
            row[f'p{p+1}_brick'] = inventories[p, 1]
            row[f'p{p+1}_sheep'] = inventories[p, 2]
            row[f'p{p+1}_wheat'] = inventories[p, 3]
        idx += num_players * 4
        
        # Edges (settlements)
        row['edges'] = obs[idx:(idx + 24)].tolist()
        idx += 24
        
        # Sides (roads)
        row['sides'] = obs[idx:(idx + 30)].tolist()
        idx += 30
        
        # Longest road owner
        row['longest_road_owner'] = obs[idx]
        idx += 1
        
        # Victory points
        for p in range(num_players):
            row[f'p{p+1}_victory_points'] = obs[idx + p]
        idx += num_players
        
        # Biomes
        row['biomes'] = obs[idx:(idx + 7)].tolist()  
        idx += 7
        
        # Hex numbers
        row['hex_nums'] = obs[idx:(idx + 7)].tolist()  
        idx += 7
        
        # Robber location
        row['robber_loc'] = obs[idx]
        idx += 1
        
        # Turn number
        row['turn_number'] = obs[idx]
        idx += 1
        
        return row
    except Exception as e:
        # If any error occurs, just return the original row
        print(f"Error expanding observation: {e}")
        return row

folder_path = r"C:\Users\foosh\OneDrive\Desktop\projects\DIss\analysis\experiments"

dtype_dict = {
    'action': 'object',  # Use object type for mixed content column
}

all_games = pd.read_csv(folder_path+f"\{sys.argv[1]}.csv", dtype=dtype_dict)

all_games['reward'] = all_games['reward'].apply(extract_real_parts).astype('float64')

# Create a new trade column
all_games['trade'] = np.nan

# Process action column
for i, action in enumerate(all_games['action']):
    # Check if it's a trade representation
    if isinstance(action, str) and ('[' in action and ']' in action):
        # Move the trade data to trade column
        all_games.at[i, 'trade'] = action
        # Set the action to NaN
        all_games.at[i, 'action'] = np.nan
    else:
        # Try to ensure numeric actions are properly typed
        try:
            if not pd.isna(action):
                all_games.at[i, 'action'] = float(action)
        except:
            # If conversion fails, set to NaN
            all_games.at[i, 'action'] = np.nan

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

# Plot 3: Winning Frequency (including draws)
# Count all game results including wins and draws
all_results = game_summary["final_winner"].value_counts().sort_index()

# Calculate percentages
total_games = len(game_summary)
percentages = (all_results / total_games) * 100

plt.figure(figsize=(10, 6))
ax = all_results.plot(kind="bar", edgecolor="black")

# Add labels to the top of each bar
for i, (count, percentage) in enumerate(zip(all_results, percentages)):
    ax.text(i, count + (max(all_results) * 0.02), f"{percentage:.1f}%",
            ha='center', va='bottom', fontweight='bold')

# Update x-tick labels to clarify that -1 means draw
x_labels = [f"Draw" if idx == -1 else f"Player {idx}" for idx in all_results.index]
ax.set_xticklabels(x_labels)

plt.xlabel("Result")
plt.ylabel("Number of Games")
plt.title("Game Outcomes with Win Percentages")
plt.tight_layout()
plt.show()

# Plot 4: Resource Progression
plt.figure(figsize=(12, 8))

# Group by turn number and calculate average resources for each player
resource_progression = all_games.groupby('turn_number').agg({
    'p1_wood': 'mean',
    'p1_brick': 'mean',
    'p1_sheep': 'mean',
    'p1_wheat': 'mean',
    'p2_wood': 'mean',
    'p2_brick': 'mean',
    'p2_sheep': 'mean',
    'p2_wheat': 'mean'
}).reset_index()

# Filter to include only turns up to a reasonable point (e.g., turn 20)
max_turn = min(200, resource_progression['turn_number'].max())
resource_progression = resource_progression[resource_progression['turn_number'] <= max_turn]

# Plot Player 1 resources
plt.subplot(2, 1, 1)
plt.plot(resource_progression['turn_number'], resource_progression['p1_wood'], label='Wood', color='brown')
plt.plot(resource_progression['turn_number'], resource_progression['p1_brick'], label='Brick', color='red')
plt.plot(resource_progression['turn_number'], resource_progression['p1_sheep'], label='Sheep', color='green')
plt.plot(resource_progression['turn_number'], resource_progression['p1_wheat'], label='Wheat', color='yellow')
plt.title('Player 1: Average Resources by Turn')
plt.xlabel('Turn Number')
plt.ylabel('Average Resource Count')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)

# Plot Player 2 resources
plt.subplot(2, 1, 2)
plt.plot(resource_progression['turn_number'], resource_progression['p2_wood'], label='Wood', color='brown')
plt.plot(resource_progression['turn_number'], resource_progression['p2_brick'], label='Brick', color='red')
plt.plot(resource_progression['turn_number'], resource_progression['p2_sheep'], label='Sheep', color='green')
plt.plot(resource_progression['turn_number'], resource_progression['p2_wheat'], label='Wheat', color='yellow')
plt.title('Player 2: Average Resources by Turn')
plt.xlabel('Turn Number')
plt.ylabel('Average Resource Count')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)

plt.tight_layout()
plt.show()

# Plot 5: Trade Analysis by Player
plt.close('all')  # Close any existing figures

# Create a figure with subplots - one row for each player and one for combined
fig, axes = plt.subplots(3, 2, figsize=(15, 15))
fig.suptitle('Trade Analysis', fontsize=18)

# Extract trades from the trade column
trades = all_games[~all_games['trade'].isna()].copy()
print(f"Found {len(trades)} trades")

# Function to parse NumPy array format trades - simplified since format is consistent
def parse_trade(trade_str):
    try:
        if isinstance(trade_str, str):
            # Extract the numbers directly assuming format is [[a b c d] [e f g h]]
            numbers = re.findall(r'\d+', trade_str)
            if len(numbers) == 8:  # 4 for offer, 4 for request
                offer = np.array([int(numbers[i]) for i in range(4)])
                request = np.array([int(numbers[i+4]) for i in range(4)])
                return offer, request
    except Exception as e:
        print(f"Error parsing trade: {e}")
    return None, None

# Resource names
resource_names = ['Wood', 'Brick', 'Sheep', 'Wheat']

# Parse all trades and separate by player
player1_offers = []
player1_requests = []
player2_offers = []
player2_requests = []
all_offers = []
all_requests = []

for i, trade_str in enumerate(trades['trade']):
    offer, request = parse_trade(trade_str)
    if offer is not None and request is not None:
        player = trades.iloc[i]['current_player']
        
        # Add to all trades
        all_offers.append(offer)
        all_requests.append(request)
        
        # Add to player-specific lists
        if player == 0:  # Player 1
            player1_offers.append(offer)
            player1_requests.append(request)
        elif player == 1:  # Player 2
            player2_offers.append(offer)
            player2_requests.append(request)

"""print(f"Successfully parsed {len(all_offers)} trades")
print(f"Player 1 trades: {len(player1_offers)}")
print(f"Player 2 trades: {len(player2_offers)}")"""

# Function to create resource distribution plot
def plot_resource_distribution(ax, offers, requests, title):
    if len(offers) == 0:
        ax.text(0.5, 0.5, 'No trades found', ha='center', va='center', transform=ax.transAxes)
        return
    
    offers_array = np.array(offers)
    requests_array = np.array(requests)
    
    offered_counts = offers_array.sum(axis=0)
    requested_counts = requests_array.sum(axis=0)
    
    ax.bar(np.arange(4), offered_counts, width=0.4, label='Offered', color='green', alpha=0.7)
    ax.bar(np.arange(4) + 0.4, requested_counts, width=0.4, label='Requested', color='blue', alpha=0.7)
    ax.set_title(title)
    ax.set_xlabel('Resource Type')
    ax.set_ylabel('Total Count')
    ax.set_xticks(np.arange(4) + 0.2)
    ax.set_xticklabels(resource_names)
    ax.legend()

# Function to create resource exchange heatmap
def plot_exchange_heatmap(ax, offers, requests, title):
    if len(offers) == 0:
        ax.text(0.5, 0.5, 'No trades found', ha='center', va='center', transform=ax.transAxes)
        return
    
    # Create matrix showing how many of resource i was traded for resource j
    exchange_matrix = np.zeros((4, 4))
    
    for offer, request in zip(offers, requests):
        for i in range(4):
            for j in range(4):
                if offer[i] > 0 and request[j] > 0:
                    exchange_matrix[i, j] += min(offer[i], request[j])
    
    # Plot as heatmap
    im = ax.imshow(exchange_matrix, cmap='YlOrRd')
    ax.set_title(title)
    ax.set_xlabel('Resource Requested')
    ax.set_ylabel('Resource Offered')
    ax.set_xticks(np.arange(4))
    ax.set_yticks(np.arange(4))
    ax.set_xticklabels(resource_names)
    ax.set_yticklabels(resource_names)
    
    # Add text annotations in the heatmap cells
    for i in range(4):
        for j in range(4):
            text = ax.text(j, i, int(exchange_matrix[i, j]),
                          ha="center", va="center", color="black")
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Exchange Frequency')

# Create the plots for each player and combined
# Row 0: Player 1
plot_resource_distribution(axes[0, 0], player1_offers, player1_requests, 'Player 1: Resources in Trades')
plot_exchange_heatmap(axes[0, 1], player1_offers, player1_requests, 'Player 1: Resource Exchange Patterns')

# Row 1: Player 2
plot_resource_distribution(axes[1, 0], player2_offers, player2_requests, 'Player 2: Resources in Trades')
plot_exchange_heatmap(axes[1, 1], player2_offers, player2_requests, 'Player 2: Resource Exchange Patterns')

# Row 2: All Trades
plot_resource_distribution(axes[2, 0], all_offers, all_requests, 'All Players: Resources in Trades')
plot_exchange_heatmap(axes[2, 1], all_offers, all_requests, 'All Players: Resource Exchange Patterns')

plt.tight_layout()
plt.subplots_adjust(top=0.95)
plt.show()


# Plot 6: VP Margin Analysis and Draw Proximity
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))

# Get the last state of each game
last_states = all_games.loc[all_games.groupby('game_id')['id'].idxmax()]

# Analyze non-draw games (margins between winner and loser)
won_games = last_states[last_states['winner'] > 0].copy()  # Using > 0 since -1 is draw

# Create columns for winner and loser victory points and calculate margin
won_games['winner_vp'] = won_games.apply(
    lambda row: row['p1_victory_points'] if row['winner'] == 1 else row['p2_victory_points'], 
    axis=1
)
won_games['loser_vp'] = won_games.apply(
    lambda row: row['p2_victory_points'] if row['winner'] == 1 else row['p1_victory_points'], 
    axis=1
)
won_games['margin'] = won_games['winner_vp'] - won_games['loser_vp']

# Separate by winning player
p1_wins = won_games[won_games['winner'] == 1]
p2_wins = won_games[won_games['winner'] == 2]

# Plot VP margin distribution by winning player
if len(won_games) > 0 and won_games['margin'].max() > 0:
    bins = range(1, int(won_games['margin'].max()) + 2)
    ax1.hist([p1_wins['margin'], p2_wins['margin']], bins=bins, alpha=0.7, 
            label=['Player 1 Wins', 'Player 2 Wins'], color=['blue', 'orange'])

# Add average margin lines
avg_margin_p1 = p1_wins['margin'].mean() if len(p1_wins) > 0 else 0
avg_margin_p2 = p2_wins['margin'].mean() if len(p2_wins) > 0 else 0

if len(p1_wins) > 0:
    ax1.axvline(avg_margin_p1, color='blue', linestyle='--', 
               label=f'Avg P1 Margin: {avg_margin_p1:.1f}')
if len(p2_wins) > 0:
    ax1.axvline(avg_margin_p2, color='orange', linestyle='--', 
               label=f'Avg P2 Margin: {avg_margin_p2:.1f}')

ax1.set_title('Victory Point Margin by Winning Player')
ax1.set_xlabel('VP Margin (Winner - Loser)')
ax1.set_ylabel('Number of Games')
ax1.grid(True, linestyle='--', alpha=0.7)
ax1.legend()

# Analyze draw games
draw_games = last_states[last_states['winner'] == -1].copy()

if len(draw_games) > 0:
    # Calculate how close each player was to winning in draw games
    draw_games['p1_to_win'] = 10 - draw_games['p1_victory_points']  # Assuming 10 VP to win
    draw_games['p2_to_win'] = 10 - draw_games['p2_victory_points']
    
    # Create a scatter plot for draw games
    ax2.scatter(draw_games['p1_victory_points'], draw_games['p2_victory_points'], 
               alpha=0.7, s=100, c='green', edgecolors='black')
    
    # Add count annotations
    for p1_vp in range(int(draw_games['p1_victory_points'].max()) + 1):
        for p2_vp in range(int(draw_games['p2_victory_points'].max()) + 1):
            count = len(draw_games[(draw_games['p1_victory_points'] == p1_vp) & 
                                  (draw_games['p2_victory_points'] == p2_vp)])
            if count > 0:
                ax2.annotate(str(count), (p1_vp, p2_vp), 
                             ha='center', va='center', fontweight='bold')
    
    # Add diagonal line of equal VPs
    max_vp = max(draw_games['p1_victory_points'].max(), draw_games['p2_victory_points'].max())
    ax2.plot([0, max_vp], [0, max_vp], 'r--', label='Equal VPs')
    
    # Calculate average VP in draws
    avg_p1_vp = draw_games['p1_victory_points'].mean()
    avg_p2_vp = draw_games['p2_victory_points'].mean()
    ax2.plot(avg_p1_vp, avg_p2_vp, 'ro', markersize=10, 
            label=f'Avg (P1: {avg_p1_vp:.1f}, P2: {avg_p2_vp:.1f})')
    
    ax2.set_title('Victory Points in Draw Games')
    ax2.set_xlabel('Player 1 Victory Points')
    ax2.set_ylabel('Player 2 Victory Points')
    ax2.grid(True, linestyle='--', alpha=0.7)
    ax2.legend()
else:
    ax2.text(0.5, 0.5, 'No Draw Games in Dataset', 
             ha='center', va='center', fontsize=14, transform=ax2.transAxes)

# Add overall summary
overall_margin = won_games['margin'].mean() if len(won_games) > 0 else 0
fig.suptitle(f'Victory Point Analysis (Overall Average Margin: {overall_margin:.2f} VP)', fontsize=16)

plt.tight_layout()
plt.show()

def plot_settlement_biome_hex_heatmap_for_player(all_games, player=1):
    # Get the final state of each game (assuming 'id' increases during the game)
    #last_states = all_games.loc[all_games.groupby('game_id')['id'].idxmax()].copy()

    first_moves = []
    for game_id, game_data in all_games.groupby('game_id'):
        # Sort by id to ensure moves are in order
        game_data_sorted = game_data.sort_values('id')
        # Take the first 8 moves (or all moves if less than 8)
        first_8_moves = game_data_sorted.head(8)
        first_moves.append(first_8_moves)

    # Combine all the first moves into a single DataFrame
    last_states = pd.concat(first_moves, ignore_index=True)
    
    # Helper function to convert a string array into a numpy array (comma-separated)
    def convert_to_np(s_arr):
        return np.fromstring(s_arr.strip('[]'), dtype=int, sep=",")
    
    # Convert the necessary columns
    last_states['edges'] = last_states['edges'].apply(convert_to_np)
    last_states['biomes'] = last_states['biomes'].apply(convert_to_np)
    last_states['hex_nums'] = last_states['hex_nums'].apply(convert_to_np)
    
    # Mapping from settlement (edge) index to adjacent hex indices
    edges_to_hexes = {
        0: [0], 1: [0, 2], 2: [0, 2, 3], 3: [0, 1, 3], 
        4: [0, 1], 5: [0], 6: [1, 3, 4], 7: [1, 4],
        8: [1], 9: [1], 10: [2], 11: [2], 
        12: [2, 5], 13: [2, 3, 5], 14: [3, 5, 6], 15: [3, 4, 6],
        16: [4, 6], 17: [4], 18: [4], 19: [5],
        20: [5], 21: [5, 6], 22: [6], 23: [6]
    }
    
    # Define biome names; we assume that biome codes are integers 0-4.
    biome_names = ['Desert', 'Forest', 'Hills', 'Fields', 'Pasture']
    
    # This dictionary will hold counts keyed by (hex_number, biome)
    counts = {}
    
    # Loop over each final game state and update counts for each settlement placed by this player.
    for _, row in last_states.iterrows():
        edges_arr = row['edges']       # Array of settlement placements (length 24)
        biomes_arr = row['biomes']     # Array of biome codes for each hex (e.g., length 7)
        hex_nums_arr = row['hex_nums'] # Array of hex numbers (e.g., length 7)
        
        for edge_idx, settlement in enumerate(edges_arr):
            if settlement == player:
                adjacent_hexes = edges_to_hexes.get(edge_idx, [])
                for hex_idx in adjacent_hexes:
                    # Make sure the adjacent hex exists in both arrays.
                    #if hex_idx < len(biomes_arr) and hex_idx <= len(hex_nums_arr):
                    biome_code = biomes_arr[hex_idx]
                    hex_num = hex_nums_arr[hex_idx]
                    #if 0 <= biome_code < len(biome_names):
                    biome_label = biome_names[biome_code]
                    key = (hex_num, biome_label)
                    counts[key] = counts.get(key, 0) + 1
    
    # Get the unique hex numbers observed and sort them (for y-axis)
    hex_nums_unique = sorted({key[0] for key in counts.keys()})
    # Use the defined biome_names order for the x-axis.
    
    # Build a 2D array with rows corresponding to hex numbers and columns to biomes.
    heatmap = np.zeros((len(hex_nums_unique), len(biome_names)))
    for i, hex_num in enumerate(hex_nums_unique):
        for j, biome in enumerate(biome_names):
            heatmap[i, j] = counts.get((hex_num, biome), 0)
    
    # Plot the heatmap
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(heatmap, aspect='auto', cmap='plasma')
    ax.set_xticks(np.arange(len(biome_names)))
    ax.set_xticklabels(biome_names)
    ax.set_yticks(np.arange(len(hex_nums_unique)))
    ax.set_yticklabels(hex_nums_unique)
    ax.set_xlabel('Biome')
    ax.set_ylabel('Hex Number')
    ax.set_title(f'Settlement Adjacent Heatmap for Player {player}\n(Count by (Hex Number, Biome) combination)')
    
    # Annotate each cell with the count
    for i in range(heatmap.shape[0]):
        for j in range(heatmap.shape[1]):
            ax.text(j, i, int(heatmap[i, j]), ha='center', va='center', color='white', fontsize=12)
    
    fig.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.show()

plot_settlement_biome_hex_heatmap_for_player(all_games, player=1)
plot_settlement_biome_hex_heatmap_for_player(all_games, player=2)


# Plot 8: Outcomes when each player owns the longest road
plt.figure(figsize=(12, 7))

# Get the last state of each game
last_states = all_games.loc[all_games.groupby('game_id')['id'].idxmax()]

# Filter out games where no one has longest road (if applicable)
valid_games = last_states[last_states['longest_road_owner'] >= 0]

# Create the outcome categories for each player
outcomes = []
for player in [1, 2]:  # Player 1 and Player 2
    # Games where this player has longest road
    player_games = valid_games[valid_games['longest_road_owner'] == player]
    
    # Count different outcomes
    wins = sum(player_games['winner'] == player)
    losses = sum((player_games['winner'] != player) & (player_games['winner'] >= 0))
    draws = sum(player_games['winner'] == -1)
    total = len(player_games)
    
    # Calculate percentages
    win_pct = (wins / total * 100) if total > 0 else 0
    loss_pct = (losses / total * 100) if total > 0 else 0
    draw_pct = (draws / total * 100) if total > 0 else 0
    
    outcomes.append({
        'player': player,
        'wins': wins,
        'losses': losses,
        'draws': draws,
        'total': total,
        'win_pct': win_pct,
        'loss_pct': loss_pct,
        'draw_pct': draw_pct
    })

# Set up the plot
bar_width = 0.25
r1 = [0, 1.5]  # Positions for player 1 and player 2 groups

# Create the grouped bars for each player
for i, outcome in enumerate(outcomes):
    # Create three bars for each player
    plt.bar(r1[i] - bar_width, outcome['wins'], width=bar_width, color='green', label='Win' if i == 0 else "")
    plt.bar(r1[i], outcome['losses'], width=bar_width, color='red', label='Loss' if i == 0 else "")
    plt.bar(r1[i] + bar_width, outcome['draws'], width=bar_width, color='blue', label='Draw' if i == 0 else "")
    
    # Add count and percentage labels
    plt.text(r1[i] - bar_width, outcome['wins'] + 1, f"{outcome['wins']} ({outcome['win_pct']:.1f}%)", ha='center', fontweight='bold')
    plt.text(r1[i], outcome['losses'] + 1, f"{outcome['losses']} ({outcome['loss_pct']:.1f}%)", ha='center', fontweight='bold')
    plt.text(r1[i] + bar_width, outcome['draws'] + 1, f"{outcome['draws']} ({outcome['draw_pct']:.1f}%)", ha='center', fontweight='bold')

# Customize the plot
plt.ylabel('Number of Games')
plt.title('Outcomes When Player Owns the Longest Road')
plt.xticks([0, 1.5], ['Player 1 has Longest Road', 'Player 2 has Longest Road'])
plt.legend()
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Add overall insight
total_games = sum(outcome['total'] for outcome in outcomes)
total_wins = sum(outcome['wins'] for outcome in outcomes)
overall_win_rate = (total_wins / total_games * 100) if total_games > 0 else 0

plt.figtext(0.5, 0.01, 
            f"Overall, longest road owners win {total_wins} out of {total_games} games ({overall_win_rate:.1f}%)",
            ha='center', fontsize=12, fontweight='bold')

plt.tight_layout()
plt.subplots_adjust(bottom=0.1)  # Make room for the text at the bottom
plt.show()