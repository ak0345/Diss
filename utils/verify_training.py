import torch
import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym
from collections import defaultdict
import os
import pickle
from tqdm import tqdm

from catan_agent import DQNAgent, RandomAgent

def verify_model_weights(model_path_before, model_path_after):
    """
    Compare model weights before and after training to confirm weight updates.
    """
    print("\n=== MODEL WEIGHT VERIFICATION ===")
    if not os.path.exists(model_path_before) or not os.path.exists(model_path_after):
        print(f"Error: One or both model paths don't exist")
        return
    
    # Load models
    before_agent = DQNAgent.DQNAgent(player_index=0)
    after_agent = DQNAgent.DQNAgent(player_index=0)
    
    before_agent.load_model(model_path_before)
    after_agent.load_model(model_path_after)
    
    # Compare weights for each network
    networks = [
        ("Main Action", before_agent.main_q_network, after_agent.main_q_network),
        ("Road", before_agent.road_q_network, after_agent.road_q_network),
        ("Settlement", before_agent.settlement_q_network, after_agent.settlement_q_network),
        ("Bank Trade", before_agent.bank_trade_q_network, after_agent.bank_trade_q_network),
        ("Player Trade", before_agent.player_trade_q_network, after_agent.player_trade_q_network),
        ("Trade Response", before_agent.player_trade_response_q_network, after_agent.player_trade_response_q_network)
    ]
    
    for name, before_net, after_net in networks:
        # Get parameters for each network
        before_params = {name: param.data for name, param in before_net.named_parameters()}
        after_params = {name: param.data for name, param in after_net.named_parameters()}
        
        # Calculate total parameter difference
        total_diff = 0
        param_count = 0
        
        for param_name in before_params:
            if param_name in after_params:
                diff = torch.norm(before_params[param_name] - after_params[param_name]).item()
                param_size = before_params[param_name].numel()
                param_count += param_size
                total_diff += diff
                
        avg_diff = total_diff / max(1, len(before_params))
        
        print(f"{name} Network - Avg parameter difference: {avg_diff:.6f}")
        
        # Check if weights significantly changed (arbitrary threshold)
        if avg_diff < 0.001:
            print(f"  WARNING: Very small weight changes detected in {name} network")
        else:
            print(f"  ✓ {name} network shows significant weight updates")
    
    # Compare epsilon values
    if 'epsilon' in before_agent.__dict__ and 'epsilon' in after_agent.__dict__:
        print(f"\nEpsilon before: {before_agent.epsilon:.6f}")
        print(f"Epsilon after: {after_agent.epsilon:.6f}")
        
        if before_agent.epsilon <= after_agent.epsilon:
            print("  WARNING: Epsilon did not decrease during training")
        else:
            print("  ✓ Epsilon decreased properly during training")

def analyze_q_values(model_path, num_samples=10):
    """
    Analyze Q-values output by the model to check for reasonable values.
    """
    print("\n=== Q-VALUE ANALYSIS ===")
    # Load the model
    agent = DQNAgent.DQNAgent(player_index=0)
    agent.load_model(model_path)
    agent.eval()  # Set to evaluation mode
    
    # Create environment to get sample states
    env = gym.make("MiniCatanEnv-v0")
    env = env.unwrapped
    
    # Reset environment multiple times to get diverse states
    all_q_values = defaultdict(list)
    
    for _ in range(num_samples):
        obs, _ = env.reset()
        
        # Get Q-values for main actions
        state_tensor = torch.FloatTensor(obs).unsqueeze(0).to(agent.device)
        with torch.no_grad():
            q_values = agent.main_q_network(state_tensor).cpu().numpy()[0]
        all_q_values["main"].append(q_values)
        
        # Get Q-values for other networks
        all_q_values["road"].append(agent.road_q_network(state_tensor).cpu().numpy()[0])
        all_q_values["settlement"].append(agent.settlement_q_network(state_tensor).cpu().numpy()[0])
        all_q_values["bank_trade"].append(agent.bank_trade_q_network(state_tensor).cpu().numpy()[0])
        all_q_values["player_trade"].append(agent.player_trade_q_network(state_tensor).cpu().numpy()[0])
        all_q_values["player_trade_response"].append(agent.player_trade_response_q_network(state_tensor).cpu().numpy()[0])
    
    # Calculate statistics for each network
    networks = ["main", "road", "settlement", "bank_trade", "player_trade", "player_trade_response"]
    network_names = ["Main Action", "Road", "Settlement", "Bank Trade", "Player Trade", "Trade Response"]
    
    for network, name in zip(networks, network_names):
        # Convert to numpy array
        values = np.array(all_q_values[network])
        
        # Calculate statistics
        mean_vals = np.mean(values)
        std_vals = np.std(values)
        min_vals = np.min(values)
        max_vals = np.max(values)
        val_range = max_vals - min_vals
        
        print(f"\n{name} Network Q-values:")
        print(f"  Mean: {mean_vals:.4f}")
        print(f"  Std Dev: {std_vals:.4f}")
        print(f"  Min: {min_vals:.4f}")
        print(f"  Max: {max_vals:.4f}")
        print(f"  Range: {val_range:.4f}")
        
        # Check for reasonable values
        if val_range < 0.1:
            print("  WARNING: Very small range of Q-values (insufficient differentiation)")
        if mean_vals == 0 and std_vals == 0:
            print("  WARNING: All Q-values are zero (network not trained)")
        if min_vals == max_vals:
            print("  WARNING: All Q-values are identical (network not trained)")

def compare_performance(trained_model_path, num_games=50):
    """
    Compare the performance of trained model vs random agent.
    """
    print("\n=== PERFORMANCE COMPARISON ===")
    # Create environment
    env = gym.make("MiniCatanEnv-v0")
    env = env.unwrapped
    
    # Load trained agent
    trained_agent = DQNAgent.DQNAgent(player_index=0)
    trained_agent.load_model(trained_model_path)
    trained_agent.epsilon = 0.05  # Use small epsilon for some exploration
    
    # Create random agent
    random_agent = RandomAgent.RandomAgent(player_index=1)
    
    # First setup: Trained vs Random
    print("Testing Trained Agent (P1) vs Random Agent (P2)...")
    wins_p1 = 0
    rewards_p1 = []
    
    for game in tqdm(range(num_games)):
        obs, _ = env.reset()
        done = False
        trunc = False
        total_reward = 0
        steps = 0
        
        while not done and not trunc and steps < 1000:
            # Determine current player
            if env.board.turn_number == 0 and hasattr(env, 'init_phase'):
                if env.init_phase < 4:
                    current_player_idx = env.init_phase // 2
                else:
                    current_player_idx = 1 - ((env.init_phase - 4) // 2)
            else:
                current_player_idx = env.current_player
            
            # Choose action
            if current_player_idx == 0:
                action = trained_agent.act(obs, env.board, env)
            else:
                action = random_agent.act(obs, env.board, env)
            
            # Take action
            next_obs, reward, done, trunc, info = env.step(action)
            
            # Track rewards for player 1
            if current_player_idx == 0:
                total_reward += reward
            
            obs = next_obs
            steps += 1
        
        # Check for win
        if done and "victory_points" in info:
            if info["victory_points"][0] >= env.max_victory_points:
                wins_p1 += 1
        
        rewards_p1.append(total_reward)
    
    # Second setup: Random vs Random
    print("\nTesting Random Agent (P1) vs Random Agent (P2)...")
    random_agent_p1 = RandomAgent.RandomAgent(player_index=0)
    random_agent_p2 = RandomAgent.RandomAgent(player_index=1)
    
    wins_random = 0
    rewards_random = []
    
    for game in tqdm(range(num_games)):
        obs, _ = env.reset()
        done = False
        trunc = False
        total_reward = 0
        steps = 0
        
        while not done and not trunc and steps < 1000:
            # Determine current player
            if env.board.turn_number == 0 and hasattr(env, 'init_phase'):
                if env.init_phase < 4:
                    current_player_idx = env.init_phase // 2
                else:
                    current_player_idx = 1 - ((env.init_phase - 4) // 2)
            else:
                current_player_idx = env.current_player
            
            # Choose action
            if current_player_idx == 0:
                action = random_agent_p1.act(obs, env.board, env)
            else:
                action = random_agent_p2.act(obs, env.board, env)
            
            # Take action
            next_obs, reward, done, trunc, info = env.step(action)
            
            # Track rewards for player 1
            if current_player_idx == 0:
                total_reward += reward
            
            obs = next_obs
            steps += 1
        
        # Check for win
        if done and "victory_points" in info:
            if info["victory_points"][0] >= env.max_victory_points:
                wins_random += 1
        
        rewards_random.append(total_reward)
    
    # Compare results
    trained_win_rate = wins_p1 / num_games * 100
    random_win_rate = wins_random / num_games * 100
    
    trained_avg_reward = np.mean(rewards_p1)
    random_avg_reward = np.mean(rewards_random)
    
    print(f"\nTrained Agent Win Rate: {trained_win_rate:.2f}%")
    print(f"Random Agent Win Rate: {random_win_rate:.2f}%")
    print(f"Trained Agent Avg Reward: {trained_avg_reward:.2f}")
    print(f"Random Agent Avg Reward: {random_avg_reward:.2f}")
    
    if trained_win_rate > random_win_rate + 5:  # 5% margin to be confident
        print("  ✓ Trained agent performs significantly better than random")
    elif trained_win_rate > random_win_rate:
        print("  ✓ Trained agent performs slightly better than random")
    else:
        print("  WARNING: Trained agent does not outperform random agent")

def examine_learning_curves(metrics_path):
    """
    Analyze learning curves to check for signs of learning.
    """
    print("\n=== LEARNING CURVE ANALYSIS ===")
    if not os.path.exists(metrics_path):
        print(f"Error: Metrics file not found at {metrics_path}")
        return
    
    # Load metrics
    with open(metrics_path, 'rb') as f:
        metrics = pickle.load(f)
    
    # Get key metrics
    episode_rewards = metrics.get('episode_rewards', [])
    win_history = metrics.get('win_history', [])
    epsilon_history = metrics.get('epsilon_history', [])
    
    # Check if rewards are improving
    if len(episode_rewards) > 100:
        first_100_avg = np.mean(episode_rewards[:100])
        last_100_avg = np.mean(episode_rewards[-100:])
        
        print(f"First 100 episodes avg reward: {first_100_avg:.4f}")
        print(f"Last 100 episodes avg reward: {last_100_avg:.4f}")
        
        if last_100_avg > first_100_avg:
            print("  ✓ Rewards improved during training")
        else:
            print("  WARNING: Rewards did not improve during training")
    
    # Check if win rate is improving
    if len(win_history) > 100:
        first_100_win_rate = np.mean(win_history[:100]) * 100
        last_100_win_rate = np.mean(win_history[-100:]) * 100
        
        print(f"First 100 episodes win rate: {first_100_win_rate:.2f}%")
        print(f"Last 100 episodes win rate: {last_100_win_rate:.2f}%")
        
        if last_100_win_rate > first_100_win_rate:
            print("  ✓ Win rate improved during training")
        else:
            print("  WARNING: Win rate did not improve during training")
    
    # Check if epsilon is decreasing
    if len(epsilon_history) > 10:
        first_epsilon = epsilon_history[0]
        last_epsilon = epsilon_history[-1]
        
        print(f"Starting epsilon: {first_epsilon:.4f}")
        print(f"Final epsilon: {last_epsilon:.4f}")
        
        if last_epsilon < first_epsilon:
            print("  ✓ Epsilon decreased properly during training")
        else:
            print("  WARNING: Epsilon did not decrease during training")
    
    # Plot smoothed learning curves
    if len(episode_rewards) > 0:
        plt.figure(figsize=(15, 10))
        
        # Smoothed reward curve
        smoothing_window = min(100, len(episode_rewards) // 10)
        smoothed_rewards = np.convolve(episode_rewards, np.ones(smoothing_window)/smoothing_window, mode='valid')
        
        plt.subplot(2, 2, 1)
        plt.plot(episode_rewards, alpha=0.3, color='blue')
        plt.plot(range(smoothing_window-1, len(episode_rewards)), smoothed_rewards, color='blue')
        plt.title('Episode Rewards (with smoothing)')
        plt.xlabel('Episode')
        plt.ylabel('Reward')
        
        # Smoothed win rate
        if len(win_history) > 0:
            win_rate = []
            window = min(100, len(win_history) // 5)
            for i in range(window, len(win_history)+1):
                win_rate.append(np.mean(win_history[i-window:i]) * 100)
            
            plt.subplot(2, 2, 2)
            plt.plot(range(window, len(win_history)+1), win_rate)
            plt.title('Win Rate (moving average)')
            plt.xlabel('Episode')
            plt.ylabel('Win Rate (%)')
        
        # Epsilon curve
        if len(epsilon_history) > 0:
            plt.subplot(2, 2, 3)
            plt.plot(epsilon_history)
            plt.title('Exploration Rate (Epsilon)')
            plt.xlabel('Episode')
            plt.ylabel('Epsilon')
        
        plt.tight_layout()
        plt.savefig("training_analysis.png")
        plt.close()
        
        print(f"Saved learning curve analysis to training_analysis.png")

# Main function
def verify_training(initial_model="dqn_training/models/dqn_agent_episode_1.pt", 
                    final_model="dqn_training/models/dqn_agent_final.pt",
                    metrics_file="dqn_training/metrics_episode_5000.pkl"):
    """
    Run all verification methods to comprehensively check if model was trained.
    """
    print("=== DQN MODEL TRAINING VERIFICATION ===")
    
    # 1. Compare model weights
    verify_model_weights(initial_model, final_model)
    
    # 2. Analyze Q-values
    analyze_q_values(final_model)
    
    # 3. Compare performance against random agent
    compare_performance(final_model, num_games=20)
    
    # 4. Examine learning curves
    examine_learning_curves(metrics_file)
    
    print("\nVerification complete. Check the results above to determine if model was properly trained.")

if __name__ == "__main__":
    # Provide paths to the models you want to verify
    initial_model = input("Enter path to initial model (default: dqn_training/models/dqn_agent_episode_1.pt): ") or "dqn_training/models/dqn_agent_episode_1.pt"
    final_model = input("Enter path to final model (default: dqn_training/models/dqn_agent_final.pt): ") or "dqn_training/models/dqn_agent_final.pt"
    metrics_file = input("Enter path to metrics file (default: dqn_training/metrics_episode_5000.pkl): ") or "dqn_training/metrics_episode_5000.pkl"
    
    verify_training(initial_model, final_model, metrics_file)