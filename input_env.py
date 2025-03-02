import catan_agent
import mini_catan
import gymnasium as gym
import numpy as np

# Create and reset the environment.
game = gym.make("MiniCatanEnv-v0")
game.reset()
agent = None

print("Interactive Mode for MiniCatanEnv:")
while True:
    print("Choose a valid agent to go up against (pick human if not against any agent)")
    agent_name = input(">>> ")
    try:
        if "agent" in agent_name.lower():
            agent = eval(f"catan_agent.{agent_name}.{agent_name}()")
            break
        elif agent_name.lower() == "player":
            agent = None
            break
    except NameError:
        print("Incorrect command")
    except AttributeError:
        print("Incorrect Agent Name")

print("Type any valid command (e.g., game.step(2), game.render()) or 'exit' to quit.")

if agent:
    print(f"Going Up Against {agent_name}")
    while True:
        
        if game.current_player == game.main_player:
            user_command = input(">>> ")
            if user_command.strip().lower() == "exit":
                break
            try:
                # Evaluate the user's command.
                # For example, if the user types "game.step(2)", it will execute that.
                result = eval(user_command)
                if result is not None:
                    print(result)
            except Exception as e:
                print("Error:", e)
        else:
            print("Agent Running Turn......")
            while True:
                try:
                    move = agent.act(game.board.turn_number)
                    print(move)
                    print(game.step(move))
                    break
                except AssertionError:
                    pass
else:
    while True:
        user_command = input(">>> ")
        if user_command.strip().lower() == "exit":
            break
        try:
            # Evaluate the user's command.
            # For example, if the user types "game.step(2)", it will execute that.
            result = eval(user_command)
            if result is not None:
                print(result)
        except Exception as e:
            print("Error:", e)