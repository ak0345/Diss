import re
import pandas as pd
import numpy as np
import glob
import ast

def extract_real_parts(value):
    if not isinstance(value, str):
        return value
    
    if '(' in value:
        # Use regex to extract all real parts before +0j
        pattern = r'\(([+-]?\d+\.?\d*)(?:[+-]?\d+\.?\d*j)\)'
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

def expand_obs_vectorized(df):
    """
    Vectorized expansion of the 'obs' column into separate columns.
    
    Instead of processing each row individually, this function converts the entire 
    'obs' column to a 2D NumPy array (one row per observation) and uses slicing to extract:
    - Inventories (for 2 players, 4 resources each)
    - Edges (settlement placements)
    - Sides (road placements)
    - Longest road owner
    - Victory points for each player
    - Biomes
    - Hex numbers
    - Robber location
    - Turn number
    """
    # Convert the 'obs' column to a list of NumPy arrays.
    # Use ast.literal_eval to safely convert string representations to lists.
    obs_list = df['obs'].apply(lambda x: np.array(ast.literal_eval(x)) if isinstance(x, str) else np.array(x)).tolist()
    
    # Stack into a 2D array with shape (n_rows, obs_length)
    obs_mat = np.stack(obs_list)
    n = obs_mat.shape[0]
    num_players = 2
    idx = 0
    
    # Inventories: first 2*4 = 8 elements, reshape to (n, 2, 4)
    inventories = obs_mat[:, idx:(idx + num_players * 4)].reshape(n, num_players, 4)
    df['p1_wood']   = inventories[:, 0, 0]
    df['p1_brick']  = inventories[:, 0, 1]
    df['p1_sheep']  = inventories[:, 0, 2]
    df['p1_wheat']  = inventories[:, 0, 3]
    df['p2_wood']   = inventories[:, 1, 0]
    df['p2_brick']  = inventories[:, 1, 1]
    df['p2_sheep']  = inventories[:, 1, 2]
    df['p2_wheat']  = inventories[:, 1, 3]
    idx += num_players * 4
    
    # Edges (settlements): next 24 elements; convert slice to list per row
    df['edges'] = list(obs_mat[:, idx:(idx + 24)])
    idx += 24
    
    # Sides (roads): next 30 elements; convert slice to list per row
    df['sides'] = list(obs_mat[:, idx:(idx + 30)])
    idx += 30
    
    # Longest road owner: next 1 element
    df['longest_road_owner'] = obs_mat[:, idx]
    idx += 1
    
    # Victory points: next 2 elements
    df['p1_victory_points'] = obs_mat[:, idx]
    df['p2_victory_points'] = obs_mat[:, idx + 1]
    idx += num_players
    
    # Biomes: next 7 elements; convert slice to list per row
    df['biomes'] = list(obs_mat[:, idx:(idx + 7)])
    idx += 7
    
    # Hex numbers: next 7 elements; convert slice to list per row
    df['hex_nums'] = list(obs_mat[:, idx:(idx + 7)])
    idx += 7
    
    # Robber location: next 1 element
    df['robber_loc'] = obs_mat[:, idx]
    idx += 1
    
    # Turn number: next 1 element
    df['turn_number'] = obs_mat[:, idx]
    idx += 1
    
    # Drop the original 'obs' column
    df.drop("obs", axis=1, inplace=True)
    
    return df

experiment = "games_chi_ra" # Folder name

# Make a CSV containing all the games to use for Analysis
folder_path = r"C:\Users\foosh\OneDrive\Desktop\projects\DIss\analysis\experiments"
# Use glob to create a list of all CSV files in the folder
csv_files = glob.glob(folder_path + f"\{experiment}" + r"\*.csv")
# Read each CSV file into a list of DataFrames
df_games = [safe_read_csv(file) for file in csv_files]

# Each DataFrame has columns: id, obs, action, reward, winner, current_player.
# Here we combine them into one DataFrame for analysis.
all_games = pd.concat(
    [df.assign(game_id=i) for i, df in enumerate(df_games, start=1)],
    ignore_index=True
)

# Instead of applying expand_obs row-by-row, we use our vectorized version.
all_games = expand_obs_vectorized(all_games)

all_games.to_csv(r"C:\Users\foosh\OneDrive\Desktop\projects\DIss\analysis\experiments\df_" + f"{experiment}.csv")
