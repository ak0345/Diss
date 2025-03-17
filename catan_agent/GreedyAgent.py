import numpy as np
import copy

# Reward variables and functions (assumed defined as before)
S_max = 5
R_max = 10
n_type = 4

alpha = 0.5
U = lambda x: -np.exp(-alpha * x)
R_avg = lambda R_r: np.sum(R_r) / n_type

ROAD_REWARD = lambda S_2R, S_p: 1 + 1.15 * (S_2R / max(S_p, 1))
SETTLEMENT_REWARD = lambda S_p: 2 + 1.5 * (S_p / S_max)
TRADE_BANK_REWARD = lambda d_r: 1 + U(0.25 + np.sum(d_r))
TRADE_PLAYER_REWARD = lambda d_r: 1 + U(0.5 + np.sum(d_r))
REJECTED_TRADE_REWARD = lambda T_R: U(1.5 + np.power(T_R, 2))
COUTNER_OFFER_REJECTED_REWARD = lambda T_R: U(2 + np.power(T_R, 3))
COUTNER_OFFER_ACCEPTED_REWARD = lambda d_r: 1 + U(2.5 + np.sum(d_r))
INVENTORY_BALANCE_REWARD = lambda T_n, R_r: U(T_n + 1 + np.sum(np.abs(R_r - R_avg(R_r))))
LONGEST_ROAD_REWARD = 2
END_TURN_REWARD = 0.05
WIN_REWARD = 10
LOSE_REWARD = -10
INITIATE_ROAD_REWARD = 0.2
INITIATE_SETTLEMENT_REWARD = 0.3
INITIATE_TRADE_BANK_REWARD = 0.1
INITIATE_TRADE_PLAYER_REWARD = 0.15

class GreedyAgent:
    """
    A greedy agent for MiniCatan that follows a two-step decision process:
    
    1. Decide on a main action value (e.g., 0: Build Road, 1: Build Settlement,
       2: Trade with Player, 3: Trade with Bank, 4: End Turn).
    2. For actions that require a candidate index (like choosing a specific road or settlement
       placement), evaluate all available moves and choose the one with the highest estimated reward.
    
    In the initial placement phase (turn 0), the agent uses simulation (not randomness) to
    choose the best candidate.
    
    This implementation assumes that the board object provides:
      - board.all_sides (for road placements)
      - board.all_edges (for settlement placements)
      - board.simulate_place_road(player_index, candidate_side)
      - board.simulate_place_settlement(player_index, candidate_edge)
    """
    def __init__(self, player_index=0):
        self.player_index = player_index
        self.attempts = 0
        self.attempt_threshold = 5  # If too many failed attempts, cancel the move.
        self.cancel_next = False
        # These will hold the decision details:
        self.main_action = None
        self.candidate_index = None

    def _get_inventory(self, obs):
        # Assumes obs[0:4] for player 0 and obs[4:8] for player 1.
        start = self.player_index * 4
        return obs[start:start+4]
    
    def _get_other_inventory(self, obs):
        obs = np.array(obs)
        if self.player_index == 0:
            return obs[4:8]
        else:
            return obs[0:4]
    
    def _generate_valid_trade_action(self, inventory, other_inv=None):
        """
        Generates a valid 2x4 trade action.
        For bank trades, uses only the agent's inventory.
        For player trades, limits the request based on the other player's inventory.
        Returns -1 if a valid trade cannot be generated.
        """
        if not np.any(inventory >= 2):
            return -1  # Cannot afford any trade.
        offer = np.zeros(4, dtype=np.int32)
        # Offer: choose the first resource that has at least 2.
        for i in range(4):
            if inventory[i] >= 2:
                offer[i] = 2  # Offer exactly 2 units.
                break
        
        if other_inv is not None:
            if not np.any(other_inv > 0):
                return -1  # Opponent has nothing to trade.
            request = np.zeros(4, dtype=np.int32)
            for i in range(4):
                if other_inv[i] > 0:
                    request[i] = 1  # Request 1 unit.
                    break
        else:
            request = np.zeros(4, dtype=np.int32)
            request[0] = 1  # Default: request 1 unit of resource 0.
        
        return np.stack([offer, request])
    
    def act(self, obs, real_board):
        # Use a simulation copy of the board.
        board_sim = real_board
        obs = np.array(obs)
        turn_number = int(obs[80])
        
        # Handle trade followup states (same as before)
        b_trade_followup   = int(obs[81])
        p_trade_followup_1 = int(obs[82])
        p_trade_followup_2 = int(obs[83])
        p_trade_followup_3 = int(obs[84])
        reply_to_offer     = int(obs[85])
        
        inventory = self._get_inventory(obs)
        if b_trade_followup:
            action = self._generate_valid_trade_action(inventory, other_inv=None)
            self.main_action = 3  # Trade with Bank
            return action
        
        elif p_trade_followup_1:
            if reply_to_offer:
                self.main_action = 2  # Trade with Player reply
                return np.random.randint(0, 3)
            else:
                other_inventory = self._get_other_inventory(obs)
                action = self._generate_valid_trade_action(inventory, other_inv=other_inventory)
                self.main_action = 2  # Trade with Player
                return action
        
        elif p_trade_followup_2:
            self.main_action = 2  # Counter offer response
            return np.random.randint(0, 3)
        
        elif p_trade_followup_3:
            if reply_to_offer:
                self.main_action = 2  # Counter counter reply
                return np.random.randint(0, 2)
            else:
                other_inventory = self._get_other_inventory(obs)
                action = self._generate_valid_trade_action(inventory, other_inv=other_inventory)
                self.main_action = 2  # Counter counter action
                return action
        
        # For non-trade moves, we now follow a two-step decision:
        # First, decide on a main action and then choose the candidate move if needed.
        if turn_number == 0:
            # *** Initial Placement Phase ***
            # Evaluate all candidate settlement placements:
            best_settlement_reward = -np.inf
            best_settlement_index = None
            for idx, edge in enumerate(board_sim.all_edges):
                try:
                    rwd = board_sim.simulate_place_settlement(self.player_index, edge)
                except Exception:
                    rwd = -np.inf
                if rwd > best_settlement_reward:
                    best_settlement_reward = rwd
                    best_settlement_index = idx
                    
            # Evaluate all candidate road placements:
            best_road_reward = -np.inf
            best_road_index = None
            for idx, side in enumerate(board_sim.all_sides):
                try:
                    rwd = board_sim.simulate_place_road(self.player_index, side)
                except Exception:
                    rwd = -np.inf
                if rwd > best_road_reward:
                    best_road_reward = rwd
                    best_road_index = idx
            
            # Choose the main action based on which candidate is better.
            if best_settlement_reward >= best_road_reward:
                self.main_action = 1  # Build Settlement
                self.candidate_index = best_settlement_index
            else:
                self.main_action = 0  # Build Road
                self.candidate_index = best_road_index
            # Return the candidate move index.
            return self.candidate_index
        
        else:
            # *** Normal Turn (turn > 0) ***
            candidate_rewards = np.zeros(5)
            
            # Action 0: Build Road.
            best_road_reward = -np.inf
            best_road_index = None
            for idx, side in enumerate(board_sim.all_sides):
                try:
                    rwd = board_sim.simulate_place_road(board_sim.players[self.player_index], side)
                except Exception:
                    rwd = -np.inf
                if rwd > best_road_reward:
                    best_road_reward = rwd
                    best_road_index = idx
            candidate_rewards[0] = best_road_reward + LONGEST_ROAD_REWARD
            
            # Action 1: Build Settlement.
            best_settlement_reward = -np.inf
            best_settlement_index = None
            for idx, edge in enumerate(board_sim.all_edges):
                try:
                    rwd = board_sim.simulate_place_settlement(board_sim.players[self.player_index], edge)
                except Exception:
                    rwd = -np.inf
                if rwd > best_settlement_reward:
                    best_settlement_reward = rwd
                    best_settlement_index = idx
            candidate_rewards[1] = best_settlement_reward
            
            # Action 2: Trade with Player.
            candidate_rewards[2] = TRADE_PLAYER_REWARD(np.array([0, 0, 0, 0]))
            
            # Action 3: Trade with Bank.
            candidate_rewards[3] = TRADE_BANK_REWARD(np.array([0, 0, 0, 0]))
            
            # Action 4: End Turn.
            candidate_rewards[4] = END_TURN_REWARD
            
            best_main_action = np.argmax(candidate_rewards)
            self.main_action = best_main_action
            
            if best_main_action == 0:
                self.candidate_index = best_road_index
            elif best_main_action == 1:
                self.candidate_index = best_settlement_index
            else:
                self.candidate_index = None
            
            # For actions that require a candidate move (0 or 1), return that candidate index.
            if best_main_action in [0, 1]:
                return self.candidate_index
            else:
                return best_main_action
