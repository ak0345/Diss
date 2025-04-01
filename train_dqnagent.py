import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import deque
from tqdm import tqdm
import gymnasium as gym
import mini_catan
import copy

# Import the DQNAgent class from the provided code
# Assuming the DQNAgent class is in the current directory
from catan_agent import DQNAgent, RandomAgent, GreedyAgent

def train_dqn_agent(num_episodes=1000, save_interval=50, log_dir="training_logs", model_dir="models", eval_interval=100, self_play_start=2000, opponent_cycle=3, agent_name="new_agent"):
    """
    Train a DQN agent against different opponents including self-play.
    
    Args:
        num_episodes: Number of episodes to train for
        save_interval: How often to save the model
        log_dir: Directory to save logs
        model_dir: Directory to save models
        eval_interval: How often to evaluate the agent
        self_play_start: Episode number to start incorporating self-play
        opponent_cycle: How often to cycle between different opponents (1=always self-play after start, 2=alternate, 3=self-play, random, greedy)
    """
    # Create directories if they don't exist
    os.makedirs(f"DQN_Data_{agent_name}/{log_dir}", exist_ok=True)
    os.makedirs(f"DQN_Data_{agent_name}/{model_dir}", exist_ok=True)
    
    # Initialize DQN agent as player 1 (index 0)
    dqn_agent = DQNAgent.DQNAgent(player_index=0)
    
    # Initialize random and greedy agents
    random_agent = RandomAgent.RandomAgent(1)
    greedy_agent = GreedyAgent.GreedyAgent(1)
    
    # Initialize self-play agent (will be populated later)
    self_play_agent = None
    
    # Metrics for tracking progress
    win_history = []
    reward_history = []
    episode_lengths = []
    win_rate_history = []
    
    # Track opponent types
    opponent_types = []
    
    # Queue for calculating win rate
    win_queue = deque(maxlen=eval_interval)
    
    # Training loop
    for episode in tqdm(range(1, num_episodes + 1), desc="Training DQN Agent"):
        # Determine opponent for this episode
        if episode < self_play_start:
            # Before self-play starts, always use random agent
            opponent = random_agent
            opponent_type = "random"
        else:
            # After self-play starts, cycle between opponents
            cycle_position = (episode - self_play_start) % opponent_cycle
            
            if cycle_position == 0:
                # Use a previous version of the DQN agent for self-play
                # Target model is from 500 episodes ago, but no earlier than episode 500
                target_episode = max(500, episode - 500)
                # Round to the nearest save interval
                target_episode = (target_episode // save_interval) * save_interval
                
                # Load a previous version of the model for self-play
                model_path = os.path.join(f"DQN_Data_{agent_name}/{model_dir}", f"dqn_agent_episode_{target_episode}.pt")
                
                if os.path.exists(model_path):
                    # Create a new agent and load the old weights
                    self_play_agent = DQNAgent.DQNAgent(player_index=1)
                    self_play_agent.load_model(model_path)
                    self_play_agent.eval()  # Set to evaluation mode
                    
                    opponent = self_play_agent
                    opponent_type = f"self-play-{target_episode}"
                    print(f"\nUsing self-play against model from episode {target_episode}")
                else:
                    # If model doesn't exist, use random agent
                    opponent = random_agent
                    opponent_type = "random"
                    print(f"\nWanted to use self-play but model {model_path} not found, using random agent")
            
            elif cycle_position == 1:
                # Use random agent
                opponent = random_agent
                opponent_type = "random"
            
            else:
                # Use greedy agent (simple heuristic agent)
                opponent = greedy_agent
                opponent_type = "greedy"
        
        # Track opponent type
        opponent_types.append(opponent_type)
        
        # Create agents list
        agents = [dqn_agent, opponent]
        
        # Initialize environment
        env = gym.make("MiniCatanEnv-v0")
        env = env.unwrapped
        env.reset()
        
        # Initialize variables for this episode
        done = False
        total_reward = 0
        step_count = 0
        previous_obs = None  # Track previous observation for learning
        
        # Episode loop
        while True:
            # Determine current player
            if env.board.turn_number == 0 and hasattr(env, 'init_phase'):
                if env.init_phase < 4:
                    current_player_idx = env.init_phase // 2  # 0, 0, 1, 1
                else:
                    current_player_idx = 1 - ((env.init_phase - 4) // 2)  # 1, 1, 0, 0
            else:
                current_player_idx = env.current_player
            
            # Get current agent
            current_agent = agents[current_player_idx]
            
            # Check if game is done
            if done:
                if env.board.get_vp()[0] >= env.max_victory_points:
                    winner = 0  # DQN agent won
                    total_reward += 200#30
                    win_queue.append(1)
                    if previous_obs is not None:
                        dqn_agent.learn(current_state, action, total_reward, None, done, action_type)

                elif env.board.get_vp()[1] >= env.max_victory_points:
                    winner = 1  # Opponent won
                    total_reward -= 100#20
                    win_queue.append(0)
                    if previous_obs is not None:
                        dqn_agent.learn(current_state, action, total_reward, None, done, action_type)

                else:
                    winner = -1  # Draw
                    total_reward -= 40#6
                    win_queue.append(0)
                    if previous_obs is not None:
                        dqn_agent.learn(current_state, action, total_reward, None, done, action_type)
                
                break
            
            # End game if it's too long (stalemate)
            if env.board.turn_number >= 300:
                total_reward -= 40#6
                done = True
                winner = -1  # Draw
                win_queue.append(0)
                if previous_obs is not None:
                    dqn_agent.learn(current_state, action, total_reward, None, done, action_type)
                mini_catan.CatanEnv.print("============================================ Game Concluded ============================================")
                break
            
            # Get action from agent
            try:
                # Determine current state for the agent
                current_state = previous_obs if previous_obs is not None else env.state
                
                # Get action
                action = current_agent.act(current_state, env.board, env)
                
                # Take step in environment
                next_obs, reward, done, trunc, info = env.step(action)
                
                # If current agent is DQN agent, update it
                if current_agent == dqn_agent:
                    # Determine action type for DQN update
                    if env.waiting_for_road_build_followup:
                        action_type = "road"
                    elif env.waiting_for_settlement_build_followup:
                        action_type = "settlement"
                    elif env.waiting_for_b_trade_followup:
                        action_type = "bank_trade"
                    elif env.waiting_for_p_trade_followup_1 or env.waiting_for_p_trade_followup_2 or env.waiting_for_p_trade_followup_3:
                        if env.reply_to_offer:
                            action_type = "player_trade_response"
                        else:
                            action_type = "player_trade"
                    else:
                        action_type = "main"
                    
                    # Learn from experience - only if we have previous observation
                    if previous_obs is not None:
                        dqn_agent.learn(current_state, action, reward, next_obs, done, action_type)
                    
                    # Update total reward
                    total_reward += reward
                
                # Update observation
                previous_obs = next_obs
                step_count += 1
                
            except Exception as e:
                pass
                #print(f"Error during episode {episode}, step {step_count}: {e}")
                #import traceback
                #traceback.print_exc()
                #done = True
                #winner = -1  # Draw due to error
                #win_queue.append(0)
                #break
        
        # Record metrics
        win_history.append(1 if winner == 0 else 0)
        reward_history.append(total_reward)
        episode_lengths.append(step_count)
        
        # Calculate win rate
        if episode % eval_interval == 0:
            win_rate = np.mean(win_queue)
            win_rate_history.append(win_rate)
            
            # Count opponent types in recent episodes
            recent_opponents = opponent_types[-eval_interval:]
            opponent_counts = {}
            for opp in set(recent_opponents):
                opponent_counts[opp] = recent_opponents.count(opp)
            
            print(f"\nEpisode {episode}, Win Rate: {win_rate:.2f}, Avg Reward: {np.mean(reward_history[-eval_interval:]):.2f}")
            print(f"Recent opponents: {opponent_counts}")
        
        # Save model at intervals
        if episode % save_interval == 0:
            model_path = os.path.join(f"DQN_Data_{agent_name}/{model_dir}", f"dqn_agent_episode_{episode}.pt")
            dqn_agent.save_model(model_path)
            print(f"\nSaved model at episode {episode} to {model_path}")
            
            # Save metrics
            metrics = {
                "episode": list(range(1, episode + 1)),
                "win": win_history,
                "reward": reward_history,
                "length": episode_lengths,
                "opponent": opponent_types
            }
            
            metrics_df = pd.DataFrame(metrics)
            metrics_path = os.path.join(f"DQN_Data_{agent_name}/{log_dir}", "training_metrics.csv")
            metrics_df.to_csv(metrics_path, index=False)
            
            # Plot metrics
            plot_metrics(win_history, reward_history, episode_lengths, win_rate_history, 
                         opponent_types, eval_interval, os.path.join(f"DQN_Data_{agent_name}/{log_dir}", f"metrics_episode_{episode}.png"))
    
    # Save final model
    final_model_path = os.path.join(f"DQN_Data_{agent_name}/{model_dir}", "dqn_agent_final.pt")
    dqn_agent.save_model(final_model_path)
    print(f"\nSaved final model to {final_model_path}")
    
    # Plot final metrics
    plot_metrics(win_history, reward_history, episode_lengths, win_rate_history, 
                 opponent_types, eval_interval, os.path.join(f"DQN_Data_{agent_name}/{log_dir}", "metrics_final.png"))
    
    return dqn_agent, win_history, reward_history, episode_lengths

def plot_metrics(win_history, reward_history, episode_lengths, win_rate_history, opponent_types, eval_interval, save_path):
    """
    Plot training metrics and save the figure.
    
    Args:
        win_history: List of binary values indicating whether DQN agent won each episode
        reward_history: List of total rewards for each episode
        episode_lengths: List of episode lengths
        win_rate_history: List of win rates evaluated at intervals
        opponent_types: List of opponent types for each episode
        eval_interval: How often win rate was evaluated
        save_path: Path to save the figure
    """
    fig, axs = plt.subplots(2, 2, figsize=(15, 10))
    
    # Plot running average of win rate
    window_size = min(100, len(win_history))
    win_avg = [np.mean(win_history[max(0, i-window_size):i+1]) for i in range(len(win_history))]
    
    # Plot win rate
    line1, = axs[0, 0].plot(range(1, len(win_avg)+1), win_avg)
    axs[0, 0].set_title("DQN Agent Win Rate (Running Average)")
    axs[0, 0].set_xlabel("Episode")
    axs[0, 0].set_ylabel("Win Rate")
    axs[0, 0].grid(True)
    
    # Plot evaluated win rate
    eval_episodes = list(range(eval_interval, len(win_history) + 1, eval_interval))
    if len(eval_episodes) == len(win_rate_history):
        line2, = axs[0, 1].plot(eval_episodes, win_rate_history)
        axs[0, 1].set_title("DQN Agent Lose Rate (Evaluation)")
        axs[0, 1].set_xlabel("Episode")
        axs[0, 1].set_ylabel("Lose Rate")
        axs[0, 1].grid(True)
    
    # Plot reward history
    window_size = min(100, len(reward_history))
    reward_avg = [np.mean(reward_history[max(0, i-window_size):i+1]) for i in range(len(reward_history))]
    
    line3, = axs[1, 0].plot(range(1, len(reward_avg)+1), reward_avg)
    axs[1, 0].set_title("Average Reward (Running Average)")
    axs[1, 0].set_xlabel("Episode")
    axs[1, 0].set_ylabel("Reward")
    axs[1, 0].grid(True)
    
    # Plot episode length
    window_size = min(100, len(episode_lengths))
    length_avg = [np.mean(episode_lengths[max(0, i-window_size):i+1]) for i in range(len(episode_lengths))]
    
    line4, = axs[1, 1].plot(range(1, len(length_avg)+1), length_avg)
    axs[1, 1].set_title("Average Episode Length (Running Average)")
    axs[1, 1].set_xlabel("Episode")
    axs[1, 1].set_ylabel("Length")
    axs[1, 1].grid(True)
    
    # Add cross markers for self-play episodes
    if len(opponent_types) > 0:
        # Find episode indices where self-play was used
        self_play_episodes = [i+1 for i, opp in enumerate(opponent_types) if opp.startswith("self-play")]
        
        if self_play_episodes:
            # Plot crosses at self-play episodes on win rate graph
            for ep in self_play_episodes:
                if 1 <= ep <= len(win_avg):
                    # Get the exact y-value from the trend line
                    y_val = win_avg[ep-1]  # -1 because arrays are 0-indexed but episodes start at 1
                    axs[0, 0].plot(ep, y_val, 'rx', markersize=8)
            
            # Add a single entry to the legend
            axs[0, 0].plot([], [], 'rx', markersize=8, label='Self-Play Episodes')
            axs[0, 0].legend()
            
            # Plot crosses on evaluation win rate graph if applicable
            if len(eval_episodes) == len(win_rate_history):
                # Find which evaluation periods had self-play
                for i, ep in enumerate(eval_episodes):
                    # Check if any self-play episodes fall within this eval interval
                    start_ep = max(0, ep - eval_interval)
                    if any(start_ep < s_ep <= ep for s_ep in self_play_episodes):
                        axs[0, 1].plot(ep, win_rate_history[i], 'rx', markersize=8)
                
                # Add to legend
                axs[0, 1].plot([], [], 'rx', markersize=8, label='Self-Play Episodes')
                axs[0, 1].legend()
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def evaluate_agent(agent, agent_name="new_agent", num_episodes=1000, verbose=True):
    """
    Evaluate the trained agent against different opponents.
    
    Args:
        agent: The trained DQN agent
        num_episodes: Number of episodes to evaluate
        verbose: Whether to print progress
        
    Returns:
        results: Dictionary with evaluation results against different opponents
    """
    # Switch agent to evaluation mode
    agent.eval()
    
    # Define opponents to evaluate against
    opponents = {
        "random": RandomAgent.RandomAgent(1),
        "greedy": GreedyAgent.GreedyAgent(1)
    }
    
    try:
        # Create a temporary file to save the current model
        temp_model_path = os.path.join(f"DQN_Data_{agent_name}", "temp_self_play_model.pt")
        
        # Save the current model state
        agent.save_model(temp_model_path)
        
        # Create a new agent for self-play
        self_play_agent = DQNAgent.DQNAgent(player_index=1)
        
        # Load the saved model
        self_play_agent.load_model(temp_model_path)
        
        # Add to opponents
        opponents["self"] = self_play_agent
        
        # Remove the temporary file
        os.remove(temp_model_path)
    except Exception as e:
        print(f"Could not create self-play opponent for evaluation: {e}")
    
    results = {}
    
    # Evaluate against each opponent
    for opponent_name, opponent in opponents.items():
        print(f"\nEvaluating against {opponent_name} opponent...")
        
        # Create agents list
        agents = [agent, opponent]
        
        # Metrics for tracking performance
        wins = 0
        total_rewards = []
        episode_lens = []
        
        # Evaluation loop
        iterator = tqdm(range(num_episodes), desc=f"Evaluating vs {opponent_name}") if verbose else range(num_episodes)
        for episode in iterator:
            # Initialize environment
            env = gym.make("MiniCatanEnv-v0")
            env = env.unwrapped
            env.reset()
            
            # Initialize variables for this episode
            done = False
            total_reward = 0
            step_count = 0
            
            # Episode loop
            while True:
                # Determine current player
                if env.board.turn_number == 0 and hasattr(env, 'init_phase'):
                    if env.init_phase < 4:
                        current_player_idx = env.init_phase // 2
                    else:
                        current_player_idx = 1 - ((env.init_phase - 4) // 2)
                else:
                    current_player_idx = env.current_player
                
                # Get current agent
                current_agent = agents[current_player_idx]
                
                # Check if game is done
                if done:
                    if env.board.get_vp()[1] >= env.max_victory_points:
                        wins += 1  # DQN agent won
                    break
                
                # End game if it's too long (stalemate)
                if env.board.turn_number >= 300:
                    done = True
                    break
                
                try:
                    # Get action
                    action = current_agent.act(env.state, env.board, env)
                    
                    # Take step in environment
                    obs, reward, done, trunc, info = env.step(action)
                    
                    # If current agent is DQN agent, update metrics
                    if current_agent == agent:
                        total_reward += reward
                    
                    step_count += 1
                    
                except Exception as e:
                    if verbose:
                        print(f"Error during evaluation episode {episode}, step {step_count}: {e}")
                    #done = True
                    #break
            
            # Record metrics
            total_rewards.append(total_reward)
            episode_lens.append(step_count)
        
        # Calculate metrics
        win_rate = wins / num_episodes
        avg_reward = np.mean(total_rewards)
        avg_length = np.mean(episode_lens)
        
        if verbose:
            print(f"Evaluation results vs {opponent_name} over {num_episodes} episodes:")
            print(f"Win rate: {win_rate:.4f}")
            print(f"Average reward: {avg_reward:.4f}")
            print(f"Average episode length: {avg_length:.2f}")
        
        results[opponent_name] = {
            "win_rate": win_rate,
            "avg_reward": avg_reward,
            "avg_length": avg_length
        }
    
    # Switch agent back to training mode
    agent.train()
    
    return results

if __name__ == "__main__":
    import argparse
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Train DQN Agent for Mini Catan with Self-Play")
    parser.add_argument("--episodes", type=int, default=1000, help="Number of training episodes")
    parser.add_argument("--save-interval", type=int, default=50, help="Interval for saving model checkpoints")
    parser.add_argument("--eval-interval", type=int, default=100, help="Interval for evaluating the agent")
    parser.add_argument("--eval-episodes", type=int, default=50, help="Number of episodes for final evaluation")
    parser.add_argument("--log-dir", type=str, default="training_logs", help="Directory to save logs")
    parser.add_argument("--model-dir", type=str, default="models", help="Directory to save models")
    parser.add_argument("--agent-name", type=str, default="new_agent", help="Name of Agent")
    parser.add_argument("--load-model", type=str, default=None, help="Path to load a pre-trained model")
    parser.add_argument("--eval-only", action="store_true", help="Only evaluate a pre-trained model")
    parser.add_argument("--self-play-start", type=int, default=2000, help="Episode to start incorporating self-play")
    parser.add_argument("--opponent-cycle", type=int, default=3, help="How often to cycle between different opponents")
    
    args = parser.parse_args()
    
    # Create a DQN agent
    dqn_agent = DQNAgent.DQNAgent(player_index=0)
    
    # Load pre-trained model if specified
    if args.load_model and os.path.exists(args.load_model):
        print(f"Loading pre-trained model from {args.load_model}")
        dqn_agent.load_model(args.load_model)
    
    # Evaluate only if specified
    if args.eval_only:
        if not args.load_model:
            print("Warning: Evaluation mode selected but no model specified. Using randomly initialized model.")
        print(f"Evaluating agent over {args.eval_episodes} episodes...")
        evaluate_agent(dqn_agent, num_episodes=args.eval_episodes, agent_name=args.agent_name)
    else:
        # Train the agent
        print(f"Training DQN agent for {args.episodes} episodes...")
        print(f"Saving model every {args.save_interval} episodes to {args.model_dir}")
        print(f"Evaluating agent every {args.eval_interval} episodes")
        print(f"Self-play starts at episode {args.self_play_start}")
        print(f"Opponent cycle: {args.opponent_cycle}")
        
        trained_agent, wins, rewards, lengths = train_dqn_agent(
            num_episodes=args.episodes,
            save_interval=args.save_interval,
            log_dir=args.log_dir,
            model_dir=args.model_dir,
            eval_interval=args.eval_interval,
            self_play_start=args.self_play_start,
            opponent_cycle=args.opponent_cycle, 
            agent_name=args.agent_name
            )
        
        # Final evaluation
        print(f"\nPerforming final evaluation over {args.eval_episodes} episodes...")
        evaluate_agent(trained_agent, num_episodes=args.eval_episodes, agent_name=args.agent_name)