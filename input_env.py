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
    
    agents = [agent1, agent2]
    print(f"Game mode: Agent vs Agent ({args.agent1} vs {args.agent2})")
    print(f"Running {args.num_games} simulations")
    
    # Create folder for storing logs
    os.makedirs(f"experiments/{args.experiment}", exist_ok=True)
    
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
        csv_filename = os.path.join(f"experiments/{args.experiment}", f"{agent1_name}_{agent2_name}_game_{game_idx}.csv")
        df.to_csv(csv_filename, index=False)
        print(f"Game {game_idx} log saved to {csv_filename}")
    
    # Run analysis if requested
    if args.analyze:
        print(f"Running analysis on experiment {args.experiment}")
        os.system(f"python analysis/agent_strat_analysis.py experiments/{args.experiment}")

def main():
    # Configure argument parser
    parser = argparse.ArgumentParser(description='Mini Catan game runner')
    parser.add_argument('experiment', type=str, help='Experiment name (for log directory)')
    parser.add_argument('-m', '--mode', type=str, choices=['human_vs_agent', 'agent_vs_human', 'human_vs_human', 'agent_vs_agent'], 
                        default='agent_vs_agent', help='Game mode')
    parser.add_argument('-a1', '--agent1', type=str, default='RandomAgent', help='Agent for player 1 (e.g., "RandomAgent", "DQNAgent")')
    parser.add_argument('-a2', '--agent2', type=str, default='RandomAgent', help='Agent for player 2 (e.g., "RandomAgent", "DQNAgent")')
    parser.add_argument('-n', '--num-games', type=int, default=10, help='Number of games to simulate (for agent_vs_agent mode)')
    parser.add_argument('-r', '--render', action='store_true', help='Render the game')
    parser.add_argument('-a', '--analyze', action='store_true', help='Run analysis after simulation')
    
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