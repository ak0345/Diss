import catan_agent
import mini_catan
import gymnasium as gym
import numpy as np
import traceback
import pandas as pd

# Create and reset the environment.
game = gym.make("MiniCatanEnv-v0")
game.reset()

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

print("Type any valid command (e.g., game.step(2), game.render()) or 'exit' to quit.")

# Print a quick summary of the selected mode.
print("Game mode selected:")
if agents[0] is None and agents[1] is None:
    print("Human vs Human")
elif agents[0] is None and agents[1] is not None:
    print("Human vs Agent")
elif agents[0] is not None and agents[1] is None:
    print("Agent vs Human")
else:
    print("Agent vs Agent")

# Ask if we want to record actions.
record_choice = input("Record actions to CSV? (y/n): ").strip().lower()
record_csv = (record_choice == "y")
# Prepare an empty list to store log data.
log_data = []
action_id = 0

obs, reward, done, trunc, info = None, None, None, None, None
ep = 0

# Main game loop.
game.reset()
while True:
    # Determine which player's turn it is.
    current_player_idx = game.current_player  # assume index 0 or 1
    current_agent = agents[current_player_idx]

    if done == True:
        winner = current_player_idx + 1  # human-readable winner (1-indexed)
        print(f"Player {winner} wins!")
        game.render()
        # Log the terminal move.
        log_data.append({
            "id": action_id,
            "obs": obs.tolist() if obs is not None else None,
            "action": move,
            "reward": reward,
            "winner": winner,
            "current_player": current_player_idx
        })
        break

    elif current_agent is None:
        # Human player's turn: prompt for a command.
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
        # Agent player's turn.
        print(f"Agent for player {current_player_idx} running turn......")
        while True:
            try:
                if obs is not None:
                    move = current_agent.act(obs)
                else:
                    move = current_agent.act(game.state)
                print("Agent move:", move)
                obs, reward, done, trunc, info = game.step(move)
                print(obs)
                ep += 1
                if ep == 10000:
                    ep = 0
                    game.render()

                # Log the move.
                log_data.append({
                    "id": action_id,
                    "obs": obs.tolist() if obs is not None else None,
                    "action": move,
                    "reward": reward,
                    "winner": -1,
                    "current_player": current_player_idx
                })
                action_id += 1
                break
            except AssertionError as e:
                # If an assertion error occurs (e.g. due to invalid move), try again.
                traceback.print_exc()

# If recording, create a DataFrame and save it to CSV.
if record_csv:
    df = pd.DataFrame(log_data, columns=["id", "obs", "action", "reward", "winner", "current_player"])
    df.to_csv("catan_game_log.csv", index=False)
    print("Game log saved to catan_game_log.csv")