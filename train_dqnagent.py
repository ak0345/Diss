import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import deque
from tqdm import tqdm
import gymnasium as gym
import mini_catan

# Import the DQNAgent class from the provided code
# Assuming the DQNAgent class is in the current directory
from catan_agent import DQNAgent, RandomAgent, GreedyAgent

def train_dqn_agent(num_episodes=1000, save_interval=50, log_dir="training_logs", model_dir="models", eval_interval=100):
    """
    Train a DQN agent against a RandomAgent in the Catan environment.
    
    Args:
        num_episodes: Number of episodes to train for
        save_interval: How often to save the model
        log_dir: Directory to save logs
        model_dir: Directory to save models
        eval_interval: How often to evaluate the agent
    """
    # Create directories if they don't exist
    os.makedirs(f"DQN_Data/{log_dir}", exist_ok=True)
    os.makedirs(f"DQN_Data/{model_dir}", exist_ok=True)
    
    # Initialize DQN agent as player 1 (index 0)
    dqn_agent = DQNAgent.DQNAgent(player_index=0)
    
    # Initialize random agent as player 2 (index 1)
    random_agent = RandomAgent.RandomAgent(1)
    
    # Create agents list
    agents = [dqn_agent, random_agent]
    
    # Metrics for tracking progress
    win_history = []
    reward_history = []
    episode_lengths = []
    win_rate_history = []
    
    # Queue for calculating win rate
    win_queue = deque(maxlen=eval_interval)
    
    # Training loop
    for episode in tqdm(range(1, num_episodes + 1), desc="Training DQN Agent"):
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
                    win_queue.append(1)
                elif env.board.get_vp()[1] >= env.max_victory_points:
                    winner = 1  # Random agent won
                    win_queue.append(0)
                else:
                    winner = -1  # Draw
                    win_queue.append(0)
                
                break
            
            # End game if it's too long (stalemate)
            if env.board.turn_number >= 1000:
                done = True
                winner = -1  # Draw
                win_queue.append(0)
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
            print(f"\nEpisode {episode}, Win Rate: {win_rate:.2f}, Avg Reward: {np.mean(reward_history[-eval_interval:]):.2f}, Avg Length: {np.mean(episode_lengths[-eval_interval:]):.2f}")
        
        # Save model at intervals
        if episode % save_interval == 0:
            model_path = os.path.join(f"DQN_Data/{model_dir}", f"dqn_agent_episode_{episode}.pt")
            dqn_agent.save_model(model_path)
            print(f"\nSaved model at episode {episode} to {model_path}")
            
            # Save metrics
            metrics = {
                "episode": list(range(1, episode + 1)),
                "win": win_history,
                "reward": reward_history,
                "length": episode_lengths
            }
            
            metrics_df = pd.DataFrame(metrics)
            metrics_path = os.path.join(f"DQN_Data/{log_dir}", "training_metrics.csv")
            metrics_df.to_csv(metrics_path, index=False)
            
            # Plot metrics
            plot_metrics(win_history, reward_history, episode_lengths, win_rate_history, 
                         eval_interval, os.path.join(f"DQN_Data/{log_dir}", f"metrics_episode_{episode}.png"))
    
    # Save final model
    final_model_path = os.path.join(f"DQN_Data/{model_dir}", "dqn_agent_final.pt")
    dqn_agent.save_model(final_model_path)
    print(f"\nSaved final model to {final_model_path}")
    
    # Plot final metrics
    plot_metrics(win_history, reward_history, episode_lengths, win_rate_history, 
                 eval_interval, os.path.join(f"DQN_Data/{log_dir}", "metrics_final.png"))
    
    return dqn_agent, win_history, reward_history, episode_lengths

def plot_metrics(win_history, reward_history, episode_lengths, win_rate_history, eval_interval, save_path):
    """
    Plot training metrics and save the figure.
    
    Args:
        win_history: List of binary values indicating whether DQN agent won each episode
        reward_history: List of total rewards for each episode
        episode_lengths: List of episode lengths
        win_rate_history: List of win rates evaluated at intervals
        eval_interval: How often win rate was evaluated
        save_path: Path to save the figure
    """
    fig, axs = plt.subplots(2, 2, figsize=(15, 10))
    
    # Plot running average of win rate
    window_size = min(100, len(win_history))
    win_avg = [np.mean(win_history[max(0, i-window_size):i+1]) for i in range(len(win_history))]
    
    # Plot win rate
    axs[0, 0].plot(win_avg)
    axs[0, 0].set_title("DQN Agent Win Rate (Running Average)")
    axs[0, 0].set_xlabel("Episode")
    axs[0, 0].set_ylabel("Win Rate")
    axs[0, 0].grid(True)
    
    # Plot evaluated win rate
    eval_episodes = list(range(eval_interval, len(win_history) + 1, eval_interval))
    if len(eval_episodes) == len(win_rate_history):
        axs[0, 1].plot(eval_episodes, win_rate_history)
        axs[0, 1].set_title("DQN Agent Win Rate (Evaluation)")
        axs[0, 1].set_xlabel("Episode")
        axs[0, 1].set_ylabel("Win Rate")
        axs[0, 1].grid(True)
    
    # Plot reward history
    window_size = min(100, len(reward_history))
    reward_avg = [np.mean(reward_history[max(0, i-window_size):i+1]) for i in range(len(reward_history))]
    
    axs[1, 0].plot(reward_avg)
    axs[1, 0].set_title("Average Reward (Running Average)")
    axs[1, 0].set_xlabel("Episode")
    axs[1, 0].set_ylabel("Reward")
    axs[1, 0].grid(True)
    
    # Plot episode length
    window_size = min(100, len(episode_lengths))
    length_avg = [np.mean(episode_lengths[max(0, i-window_size):i+1]) for i in range(len(episode_lengths))]
    
    axs[1, 1].plot(length_avg)
    axs[1, 1].set_title("Average Episode Length (Running Average)")
    axs[1, 1].set_xlabel("Episode")
    axs[1, 1].set_ylabel("Length")
    axs[1, 1].grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def evaluate_agent(agent, num_episodes=100, verbose=True):
    """
    Evaluate the trained agent against a RandomAgent for a number of episodes.
    
    Args:
        agent: The trained DQN agent
        num_episodes: Number of episodes to evaluate
        verbose: Whether to print progress
        
    Returns:
        win_rate: Fraction of episodes won by the DQN agent
        avg_reward: Average reward per episode
        avg_length: Average episode length
    """
    # Initialize random agent as player 2 (index 1)
    random_agent = GreedyAgent.GreedyAgent(1)
    
    # Create agents list
    agents = [agent, random_agent]
    
    # Switch agent to evaluation mode
    agent.eval()
    
    # Metrics for tracking performance
    wins = 0
    total_rewards = []
    episode_lens = []
    
    # Evaluation loop
    iterator = tqdm(range(num_episodes), desc="Evaluating DQN Agent") if verbose else range(num_episodes)
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
                if env.board.players[0].victory_points >= env.victory_points_to_win:
                    wins += 1  # DQN agent won
                break
            
            # End game if it's too long (stalemate)
            if env.board.turn_number >= 1000:
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
    
    # Switch agent back to training mode
    agent.train()
    
    # Calculate metrics
    win_rate = wins / num_episodes
    avg_reward = np.mean(total_rewards)
    avg_length = np.mean(episode_lens)
    
    if verbose:
        print(f"Evaluation results over {num_episodes} episodes:")
        print(f"Win rate: {win_rate:.4f}")
        print(f"Average reward: {avg_reward:.4f}")
        print(f"Average episode length: {avg_length:.2f}")
    
    return win_rate, avg_reward, avg_length

if __name__ == "__main__":
    import argparse
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Train DQN Agent for Mini Catan")
    parser.add_argument("--episodes", type=int, default=1000, help="Number of training episodes")
    parser.add_argument("--save-interval", type=int, default=50, help="Interval for saving model checkpoints")
    parser.add_argument("--eval-interval", type=int, default=100, help="Interval for evaluating the agent")
    parser.add_argument("--eval-episodes", type=int, default=50, help="Number of episodes for final evaluation")
    parser.add_argument("--log-dir", type=str, default="training_logs", help="Directory to save logs")
    parser.add_argument("--model-dir", type=str, default="models", help="Directory to save models")
    parser.add_argument("--load-model", type=str, default=None, help="Path to load a pre-trained model")
    parser.add_argument("--eval-only", action="store_true", help="Only evaluate a pre-trained model")
    
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
        evaluate_agent(dqn_agent, num_episodes=args.eval_episodes)
    else:
        # Train the agent
        print(f"Training DQN agent for {args.episodes} episodes...")
        print(f"Saving model every {args.save_interval} episodes to {args.model_dir}")
        print(f"Evaluating agent every {args.eval_interval} episodes")
        
        trained_agent, wins, rewards, lengths = train_dqn_agent(
            num_episodes=args.episodes,
            save_interval=args.save_interval,
            log_dir=args.log_dir,
            model_dir=args.model_dir,
            eval_interval=args.eval_interval
        )
        
        # Final evaluation
        print(f"\nPerforming final evaluation over {args.eval_episodes} episodes...")
        evaluate_agent(trained_agent, num_episodes=args.eval_episodes)