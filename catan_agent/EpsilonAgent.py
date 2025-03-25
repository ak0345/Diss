from catan_agent import GreedyAgent, RandomAgentV2
import numpy as np

class EpsilonAgent:
    def __init__(self, player_index=0, epsilon=0.3):
        self.greedy_brain = GreedyAgent.GreedyAgent(player_index)
        self.random_brain = RandomAgentV2.RandomAgentV2(player_index)
    
    def act(self, obs, board, current_player):
        if np.random.rand() > 0.3:
            return self.greedy_brain.act(obs, board, current_player)
        else:
            return self.random_brain.act(obs, board, current_player)
