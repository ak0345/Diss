# Mini Catan - AI Agent Framework

This repository contains an implementation of a simplified Settlers of Catan board game (Mini Catan) designed for evaluating and training AI agents. The project provides a complete environment including a game engine, custom OpenAI Gym interface, and multiple agent implementations.

## Overview

Mini Catan implements core Catan mechanics including:

- Hexagonal board with resource production
- Settlement and road building
- Resource trading between players and with the bank
- Longest road achievement
- Victory point system

The framework is specifically designed for training and evaluating different AI approaches in a complex, partially observable, multi-agent environment.

## Installation

1. Clone the repository:

```
git clone https://github.com/yourusername/mini-catan.git
cd mini-catan
```

2. Install dependencies:

```
pip install -r requirements.txt
```

## Usage

### Playing Games

You can run games with human players, AI agents, or combinations using the `play_game.py` script:

```
python play_game.py experiment_name [options]
```

Options:

- `-m {human_vs_agent,agent_vs_human,human_vs_human,agent_vs_agent}`: Game mode
- `-a1 AGENT1`: Agent for player 1 (e.g., "RandomAgent", "DQNAgent")
- `-a1dqn AGENT1_DQN_MODEL`: Path to DQN model for agent 1 (if applicable)
- `-a2 AGENT2`: Agent for player 2
- `-a2dqn AGENT2_DQN_MODEL`: Path to DQN model for agent 2 (if applicable)
- `-n NUM_GAMES`: Number of games to simulate (for agent_vs_agent mode)
- `-r`: Render the game visually

Example:

```
python play_game.py experiment1 -m agent_vs_agent -a1 DQNAgent -a1dqn models/best_model.pt -a2 GreedyAgent -n 100
```

### Training Agents

You can train DQN agents with various configurations using the `train_dqnagent.py` script:

```
python train_dqnagent.py [options]
```

Key options:

- `--episodes EPISODES`: Number of training episodes
- `--save-interval SAVE_INTERVAL`: How often to save model checkpoints
- `--eval-interval EVAL_INTERVAL`: How often to evaluate the agent
- `--self-play-start SELF_PLAY_START`: Episode to start incorporating self-play
- `--load-model LOAD_MODEL`: Path to pre-trained model for continued training
- `--eval-only`: Only evaluate a pre-trained model

Example:

```
python train_dqnagent.py --episodes 5000 --save-interval 100 --self-play-start 2000 --agent-name dqn_agent1
```

### Human vs AI Play

For a human vs. AI game, use:

```
python play_game.py experiment_name -m human_vs_agent -a2 GreedyAgent -r
```

This will open an interactive console where you can input game commands. Type `game.render()` to see the board state and `game.step(action)` to take actions.

## Agent Types

The repository includes several agent implementations:

1. **RandomAgent**: Makes completely random legal moves
2. **RandomAgentV2**: Uses weighted probabilities for action selection
3. **GreedyAgent**: Implements sophisticated reward functions for strategic gameplay
4. **DQNAgent**: A deep reinforcement learning agent that can be trained through self-play

## Project Structure

- `mini_catan/`: Core game engine and environment
  - `Board.py`: Main game board logic
  - `Hex.py`: Hexagonal tile implementation
  - `Player.py`: Player logic including inventory and trading
  - `enums.py`: Game constants and enumeration types
  - `CatanEnv.py`: OpenAI Gym environment wrapper
- `catan_agent/`: AI agent implementations
  - `RandomAgent.py`: Basic random agent
  - `RandomAgentV2.py`: Weighted random agent
  - `GreedyAgent.py`: Heuristic-based strategic agent
  - `DQNAgent.py`: Deep Q-Network implementation
- `play_game.py`: Script for running games
- `train_dqnagent.py`: Script for training DQN agents

## Creating Custom Agents

You can create custom agents by implementing the agent interface:

1. Create a new file `catan_agent/YourAgentName.py`
2. Implement a class with the same name as the file
3. Include an `__init__(self, player_index)` method
4. Implement an `act(self, obs, board, game)` method that returns valid actions
5. Use your agent with `-a1 YourAgentName` or `-a2 YourAgentName`

## License

This project is provided for academic and research purposes.
