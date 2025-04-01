import catan_agent
import mini_catan
import gymnasium as gym
import numpy as np
import traceback
import pandas as pd
import os
import sys
import argparse
import importlib
import glob
import ast
import re

def render(render_flag, game):
    """Render the game if the render flag is True"""
    if render_flag:
        game.render()

def create_agent(agent_name, player_index):
    """Create an agent instance based on agent name"""
    try:
        if agent_name.lower() == "none" or agent_name.lower() == "player" or agent_name.lower() == "human":
            return None
        
        # Try to import the agent module if it's not already imported
        if not agent_name.lower() in sys.modules:
            try:
                importlib.import_module(f"catan_agent.{agent_name}")
            except ImportError:
                print(f"Warning: Could not import module catan_agent.{agent_name}")
        
        # Create the agent
        agent = eval(f"catan_agent.{agent_name}.{agent_name}({player_index})")
        return agent
    except (NameError, AttributeError, ImportError) as e:
        print(f"Error creating agent {agent_name}: {e}")
        return None

def human_vs_agent_mode(args):
    """Run in Human vs Agent mode"""
    # Human as player 1, agent as player 2
    agent = create_agent(args.agent2, 1)
    agents = [None, agent]
    
    print("Game mode: Human vs Agent")
    if agent is None:
        print("Warning: Could not create agent. Defaulting to Human vs Human.")
    
    # Create and reset the environment
    game = gym.make("MiniCatanEnv-v0")
    game = game.unwrapped
    game.reset()
    obs, reward, done, trunc, info = None, None, None, None, None
    
    print("Type any valid command (e.g., game.step(2), game.render()) or 'exit' to quit.")
    
    while True:
        # Determine current player (handling initialization phase)
        if game.board.turn_number == 0 and hasattr(game, 'init_phase'):
            if game.init_phase < 4:
                current_player_idx = game.init_phase // 2  # 0, 0, 1, 1
            else:
                current_player_idx = 1 - ((game.init_phase - 4) // 2)  # 1, 1, 0, 0
        else:
            current_player_idx = game.current_player
        
        current_agent = agents[current_player_idx]
        
        if done:
            winner = current_player_idx + 1
            print(f"Player {winner} wins!")
            render(args.render, game)
            break
        elif current_agent is None:
            # Display current player and phase during initialization
            if game.board.turn_number == 0 and hasattr(game, 'init_phase'):
                phase_type = "settlement" if game.init_phase % 2 == 0 else "road"
                print(f"Player {current_player_idx + 1}'s turn to place a {phase_type} (Phase {game.init_phase + 1}/8)")
            else:
                print(f"Player {current_player_idx + 1}'s turn")
            
            user_command = input(">>> ")
            if user_command.strip().lower() == "exit":
                break
            try:
                result = eval(user_command)
                if result is not None:
                    print(result)
            except Exception as e:
                print("Error:", e)
        else:
            print(f"Agent for player {current_player_idx + 1} running turn......")
            try:
                if obs is not None:
                    move = current_agent.act(obs, game.board, game)
                else:
                    move = current_agent.act(game.state, game.board, game)
                print("Agent move:", move)
                obs, reward, done, trunc, info = game.step(move)
                print(obs)
            except AssertionError as e:
                traceback.print_exc()

def agent_vs_human_mode(args):
    """Run in Agent vs Human mode"""
    # Agent as player 1, human as player 2
    agent = create_agent(args.agent1, 0)
    agents = [agent, None]
    
    print("Game mode: Agent vs Human")
    if agent is None:
        print("Warning: Could not create agent. Defaulting to Human vs Human.")
    
    # Create and reset the environment
    game = gym.make("MiniCatanEnv-v0")
    game = game.unwrapped
    game.reset()
    obs, reward, done, trunc, info = None, None, None, None, None
    
    print("Type any valid command (e.g., game.step(2), game.render()) or 'exit' to quit.")
    
    while True:
        # Determine current player (handling initialization phase)
        if game.board.turn_number == 0 and hasattr(game, 'init_phase'):
            if game.init_phase < 4:
                current_player_idx = game.init_phase // 2  # 0, 0, 1, 1
            else:
                current_player_idx = 1 - ((game.init_phase - 4) // 2)  # 1, 1, 0, 0
        else:
            current_player_idx = game.current_player
        
        current_agent = agents[current_player_idx]
        
        if done:
            winner = current_player_idx + 1
            print(f"Player {winner} wins!")
            render(args.render, game)
            break
        elif current_agent is None:
            # Display current player and phase during initialization
            if game.board.turn_number == 0 and hasattr(game, 'init_phase'):
                phase_type = "settlement" if game.init_phase % 2 == 0 else "road"
                print(f"Player {current_player_idx + 1}'s turn to place a {phase_type} (Phase {game.init_phase + 1}/8)")
            else:
                print(f"Player {current_player_idx + 1}'s turn")
            
            user_command = input(">>> ")
            if user_command.strip().lower() == "exit":
                break
            try:
                result = eval(user_command)
                if result is not None:
                    print(result)
            except Exception as e:
                print("Error:", e)
        else:
            print(f"Agent for player {current_player_idx + 1} running turn......")
            try:
                if obs is not None:
                    move = current_agent.act(obs, game.board, game)
                else:
                    move = current_agent.act(game.state, game.board, game)
                print("Agent move:", move)
                obs, reward, done, trunc, info = game.step(move)
                print(obs)
            except AssertionError as e:
                traceback.print_exc()

def human_vs_human_mode(args):
    """Run in Human vs Human mode"""
    agents = [None, None]
    
    print("Game mode: Human vs Human")
    
    # Create and reset the environment
    game = gym.make("MiniCatanEnv-v0")
    game = game.unwrapped
    game.reset()
    obs, reward, done, trunc, info = None, None, None, None, None
    
    print("Type any valid command (e.g., game.step(2), game.render()) or 'exit' to quit.")
    
    while True:
        # Determine current player (handling initialization phase)
        if game.board.turn_number == 0 and hasattr(game, 'init_phase'):
            if game.init_phase < 4:
                current_player_idx = game.init_phase // 2  # 0, 0, 1, 1
            else:
                current_player_idx = 1 - ((game.init_phase - 4) // 2)  # 1, 1, 0, 0
        else:
            current_player_idx = game.current_player
        
        if done:
            winner = current_player_idx + 1
            print(f"Player {winner} wins!")
            render(args.render, game)
            break
        else:
            # Display current player and phase during initialization
            if game.board.turn_number == 0 and hasattr(game, 'init_phase'):
                phase_type = "settlement" if game.init_phase % 2 == 0 else "road"
                print(f"Player {current_player_idx + 1}'s turn to place a {phase_type} (Phase {game.init_phase + 1}/8)")
            else:
                print(f"Player {current_player_idx + 1}'s turn")
            
            user_command = input(">>> ")
            if user_command.strip().lower() == "exit":
                break
            try:
                result = eval(user_command)
                if result is not None:
                    print(result)
            except Exception as e:
                print("Error:", e)

def agent_vs_agent_mode(args):
    """Run in Agent vs Agent mode"""
    # Create agents
    agent1 = create_agent(args.agent1, 0)
    agent2 = create_agent(args.agent2, 1)
    
    if agent1 is None:
        print(f"Error: Could not create agent1 ({args.agent1}). Exiting.")
        return
    
    if agent2 is None:
        print(f"Error: Could not create agent2 ({args.agent2}). Exiting.")
        return
    
    if args.agent1_dqn_model:
        try:
            agent1.load_model(args.agent1_dqn_model)
            agent1.eval()
            print(f"Model at {args.agent1_dqn_model} loaded successfully")
        except Exception:
            print(f"Model not found at {args.agent1_dqn_model}")

    if args.agent2_dqn_model:
        try:
            agent2.load_model(args.agent2_dqn_model)
            agent2.eval()
            print(f"Model at {args.agent2_dqn_model} loaded successfully")
        except Exception:
            print(f"Model not found at {args.agent2_dqn_model}")
    
    agents = [agent1, agent2]
    print(f"Game mode: Agent vs Agent ({args.agent1} vs {args.agent2})")
    print(f"Running {args.num_games} simulations")
    
    # Create folder for storing logs
    os.makedirs(f"analysis/experiments/{args.experiment}", exist_ok=True)

    #import time
    #time.sleep(5)
    
    # Run simulations
    for game_idx in range(1, args.num_games + 1):
        print(f"\n=== Starting simulation game {game_idx}/{args.num_games} ===")
        game = gym.make("MiniCatanEnv-v0")
        game = game.unwrapped
        game.reset()
        log_data = []
        action_id = 0
        obs, reward, done, trunc, info = None, None, None, None, None

        # Run the game simulation until it terminates
        while True:
            # Determine current player (handling initialization phase)
            if game.board.turn_number == 0 and hasattr(game, 'init_phase'):
                if game.init_phase < 4:
                    current_player_idx = game.init_phase // 2
                else:
                    current_player_idx = 1 - ((game.init_phase - 4) // 2)
            else:
                current_player_idx = game.current_player
                
            current_agent = agents[current_player_idx]

            if done:
                winner = current_player_idx + 1  # 1-indexed winner
                print(f"Game {game_idx}: Player {winner} wins!")
                log_data.append({
                    "id": action_id,
                    "obs": obs.tolist() if obs is not None else None,
                    "action": move,
                    "reward": reward,
                    "winner": winner,
                    "current_player": current_player_idx
                })
                render(args.render, game)
                break

            # End Game as a Stalemate if lasts more than 1000 turns
            elif game.board.turn_number == 1000:
                print(f"Game {game_idx}: Stalemate/Draw")
                log_data.append({
                    "id": action_id,
                    "obs": obs.tolist() if obs is not None else None,
                    "action": move,
                    "reward": reward,
                    "winner": -1,
                    "current_player": current_player_idx
                })
                render(args.render, game)
                mini_catan.CatanEnv.print("============================================ Game Concluded ============================================")
                break

            # Get agent move and update environment
            try:
                if obs is not None:
                    move = current_agent.act(obs, game.board, game)
                else:
                    move = current_agent.act(game.state, game.board, game)
                
                # Add more informative logging during initialization
                if game.board.turn_number == 0 and hasattr(game, 'init_phase'):
                    phase_type = "settlement" if game.init_phase % 2 == 0 else "road"
                    print(f"Game {game_idx}, Player {current_player_idx+1} placing {phase_type}: {move}, Phase: {game.init_phase+1}/8")
                else:
                    print(f"Game {game_idx}, Player {current_player_idx+1} move: {move}, Turn: {game.board.turn_number}")
                
                obs, reward, done, trunc, info = game.step(move)
                
                # Log the move
                log_data.append({
                    "id": action_id,
                    "obs": obs.tolist() if obs is not None else None,
                    "action": move,
                    "reward": reward,
                    "winner": -1,  # Not terminal
                    "current_player": current_player_idx
                })
                action_id += 1
            except AssertionError as e:
                #traceback.print_exc()
                #break
                pass

        # After game termination, store the log as a CSV file
        df = pd.DataFrame(log_data, columns=["id", "obs", "action", "reward", "winner", "current_player"])
        agent1_name = args.agent1.lower()
        agent2_name = args.agent2.lower()
        csv_filename = os.path.join(f"analysis/experiments/{args.experiment}", f"{agent1_name}_{agent2_name}_game_{game_idx}.csv")
        df.to_csv(csv_filename, index=False)
        print(f"Game {game_idx} log saved to {csv_filename}")

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

    # Make a CSV containing all the games to use for Analysis
    folder_path = r"C:\Users\foosh\OneDrive\Desktop\projects\DIss\analysis\experiments"
    # Use glob to create a list of all CSV files in the folder
    csv_files = glob.glob(folder_path + f"\{args.experiment}" + r"\*.csv")
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

    all_games.to_csv(r"C:\Users\foosh\OneDrive\Desktop\projects\DIss\analysis\experiments\df_" + f"{args.experiment}.csv")

def main():
    # Configure argument parser
    parser = argparse.ArgumentParser(description='Mini Catan game runner')
    parser.add_argument('experiment', type=str, help='Experiment name (for log directory)')
    parser.add_argument('-m', '--mode', type=str, choices=['human_vs_agent', 'agent_vs_human', 'human_vs_human', 'agent_vs_agent'], 
                        default='agent_vs_agent', help='Game mode')
    parser.add_argument('-a1', '--agent1', type=str, default='RandomAgent', help='Agent for player 1 (e.g., "RandomAgent", "DQNAgent")')
    parser.add_argument('-a1dqn', '--agent1-dqn-model', type=str, default=None, help='If DQN for a1, input model path')
    parser.add_argument('-a2', '--agent2', type=str, default='RandomAgent', help='Agent for player 2 (e.g., "RandomAgent", "DQNAgent")')
    parser.add_argument('-a2dqn', '--agent2-dqn-model', type=str, default=None, help='If DQN for a2, input model path')
    parser.add_argument('-n', '--num-games', type=int, default=10, help='Number of games to simulate (for agent_vs_agent mode)')
    parser.add_argument('-r', '--render', action='store_true', help='Render the game')
    
    args = parser.parse_args()
    
    # Run the selected mode
    if args.mode == 'human_vs_agent':
        human_vs_agent_mode(args)
    elif args.mode == 'agent_vs_human':
        agent_vs_human_mode(args)
    elif args.mode == 'human_vs_human':
        human_vs_human_mode(args)
    elif args.mode == 'agent_vs_agent':
        agent_vs_agent_mode(args)

if __name__ == "__main__":
    main()