import catan_agent
import mini_catan
import gymnasium as gym
import numpy as np
import traceback
import pandas as pd
import os

# Create folder for game logs if it doesn't exist.
os.makedirs("games", exist_ok=True)

print("Choose game mode:")
print("1. Human vs Agent")
print("2. Agent vs Agent")
print("3. Human vs Human")
mode = input(">>> ").strip()

if mode == "1":
    # Human vs Agent: main player (index 0) is human; opponent (index 1) is agent.
    print("Choose a valid agent to go up against (type 'player' for human):")
    agent_name = input(">>> ").strip()
    try:
        if "agent" in agent_name.lower():
            agent = eval(f"catan_agent.{agent_name}.{agent_name}()")
        elif agent_name.lower() == "player":
            agent = None
    except (NameError, AttributeError) as e:
        print("Incorrect command or Agent Name:", e)
        agent = None
    agents = [None, agent]

elif mode == "2":
    # Agent vs Agent: ask for an agent for each player.
    print("Enter agent for Player 1:")
    agent1_name = input(">>> ").strip()
    try:
        if "agent" in agent1_name.lower():
            agent1 = eval(f"catan_agent.{agent1_name}.{agent1_name}(0)")
        else:
            agent1 = None
    except (NameError, AttributeError) as e:
        print("Incorrect command or Agent Name for Player 1:", e)
        agent1 = None

    print("Enter agent for Player 2:")
    agent2_name = input(">>> ").strip()
    try:
        if "agent" in agent2_name.lower():
            agent2 = eval(f"catan_agent.{agent2_name}.{agent2_name}(1)")
        else:
            agent2 = None
    except (NameError, AttributeError) as e:
        print("Incorrect command or Agent Name for Player 2:", e)
        agent2 = None

    agents = [agent1, agent2]

elif mode == "3":
    # Human vs Human: both players are human.
    agents = [None, None]

else:
    print("Invalid mode selection; defaulting to Human vs Human")
    agents = [None, None]

# If not in Agent vs Agent mode, run interactively.
if mode != "2":
    print("Type any valid command (e.g., game.step(2), game.render()) or 'exit' to quit.")
    print("Game mode selected:")
    if agents[0] is None and agents[1] is None:
        print("Human vs Human")
    elif agents[0] is None and agents[1] is not None:
        print("Human vs Agent")
    elif agents[0] is not None and agents[1] is None:
        print("Agent vs Human")
    else:
        print("Agent vs Agent")
    
    # Create and reset the environment.
    game = gym.make("MiniCatanEnv-v0")
    game = game.unwrapped
    game.reset()
    obs, reward, done, trunc, info = None, None, None, None, None
    while True:
        current_player_idx = game.current_player  # index 0 or 1
        current_agent = agents[current_player_idx]
        if done == True:
            winner = current_player_idx + 1
            print(f"Player {winner} wins!")
            #game.render()
            break
        elif current_agent is None:
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
            print(f"Agent for player {current_player_idx} running turn......")
            try:
                if obs is not None:
                    move = current_agent.act(obs, game.board)
                else:
                    move = current_agent.act(game.state, game.board)
                print("Agent move:", move)
                obs, reward, done, trunc, info = game.step(move)
                print(obs)
            except AssertionError as e:
                traceback.print_exc()

# If Agent vs Agent mode, run n simulations and record logs.
else:
    n_games = int(input("Enter number of games to simulate: ").strip())
    for game_idx in range(1, n_games+1):
        print(f"\n=== Starting simulation game {game_idx} ===")
        game = gym.make("MiniCatanEnv-v0")
        game = game.unwrapped
        game.reset()
        log_data = []
        action_id = 0
        obs, reward, done, trunc, info = None, None, None, None, None

        # Run the game simulation until it terminates.
        while True:
            current_player_idx = game.current_player  # index 0 or 1
            current_agent = agents[current_player_idx]

            if done == True:
                winner = current_player_idx + 1  # 1-indexed winner.
                print(f"Game {game_idx}: Player {winner} wins!")
                #game.render()
                log_data.append({
                    "id": action_id,
                    "obs": obs.tolist() if obs is not None else None,
                    "action": move,
                    "reward": reward,
                    "winner": winner,
                    "current_player": current_player_idx
                })
                break

            #  End Game as a Stalemate if lasts more than 1000
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
                break

            # Since we are in agent vs agent mode, there is no human input.
            try:
                if obs is not None:
                    move = current_agent.act(obs, game.board)
                else:
                    move = current_agent.act(game.state, game.board)
                print(f"Game {game_idx}, Player {current_player_idx} move: {move}, Turn: {game.board.turn_number}")
                obs, reward, done, trunc, info = game.step(move)
                # Log the move.
                log_data.append({
                    "id": action_id,
                    "obs": obs.tolist() if obs is not None else None,
                    "action": move,
                    "reward": reward,
                    "winner": -1,  # Not terminal.
                    "current_player": current_player_idx
                })
                action_id += 1
            except AssertionError as e:
                traceback.print_exc()

        # After game termination, store the log as a CSV file.
        df = pd.DataFrame(log_data, columns=["id", "obs", "action", "reward", "winner", "current_player"])
        s = "player"
        csv_filename = os.path.join("games", f"{agent1_name.lower() if agents[0] is not None else s}_{agent2_name.lower() if agents[1] is not None else s}_game_{game_idx}.csv")
        df.to_csv(csv_filename, index=False)
        print(f"Game {game_idx} log saved to {csv_filename}")
