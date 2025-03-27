import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
import os
from collections import deque
from mini_catan.enums import Structure, Biome, HexCompEnum

class DQNetwork(nn.Module):
    """
    Deep Q-Network implementation with simple architecture
    """
    def __init__(self, input_dim, output_dim, hidden_dims=[128, 64]):
        super(DQNetwork, self).__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, output_dim))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)

class DQNAgent:
    """
    DQN Agent with multiple specialized networks for different action types in Mini Catan.
    """
    def __init__(self, player_index=0, obs_space_size=87, 
                 main_action_size=5, 
                 road_action_size=30, 
                 settlement_action_size=24,
                 bank_trade_action_size=16,
                 player_trade_action_size=16,
                 player_trade_response_size=3,
                 memory_size=100000,
                 batch_size=64,
                 gamma=0.99,
                 epsilon_start=1.0,
                 epsilon_end=0.05,
                 epsilon_decay=0.9995,
                 learning_rate=0.0001,
                 hidden_dims=[256, 128, 64]):
        
        self.player_index = player_index
        self.obs_space_size = obs_space_size
        self.main_action_size = main_action_size
        self.road_action_size = road_action_size
        self.settlement_action_size = settlement_action_size
        self.bank_trade_action_size = bank_trade_action_size
        self.player_trade_action_size = player_trade_action_size
        self.player_trade_response_size = player_trade_response_size
        
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.learning_rate = learning_rate
        self.hidden_dims = hidden_dims
        
        # Device configuration
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Create networks for each action type
        self.main_q_network = DQNetwork(obs_space_size, main_action_size, hidden_dims).to(self.device)
        self.road_q_network = DQNetwork(obs_space_size, road_action_size, hidden_dims).to(self.device)
        self.settlement_q_network = DQNetwork(obs_space_size, settlement_action_size, hidden_dims).to(self.device)
        self.bank_trade_q_network = DQNetwork(obs_space_size, bank_trade_action_size, hidden_dims).to(self.device)
        self.player_trade_q_network = DQNetwork(obs_space_size, player_trade_action_size, hidden_dims).to(self.device)
        self.player_trade_response_q_network = DQNetwork(obs_space_size, player_trade_response_size, hidden_dims).to(self.device)
        
        # Target networks for stable learning
        self.main_target_network = DQNetwork(obs_space_size, main_action_size, hidden_dims).to(self.device)
        self.road_target_network = DQNetwork(obs_space_size, road_action_size, hidden_dims).to(self.device)
        self.settlement_target_network = DQNetwork(obs_space_size, settlement_action_size, hidden_dims).to(self.device)
        self.bank_trade_target_network = DQNetwork(obs_space_size, bank_trade_action_size, hidden_dims).to(self.device)
        self.player_trade_target_network = DQNetwork(obs_space_size, player_trade_action_size, hidden_dims).to(self.device)
        self.player_trade_response_target_network = DQNetwork(obs_space_size, player_trade_response_size, hidden_dims).to(self.device)
        
        # Initialize target networks with same weights as Q-networks
        self.update_target_networks(tau=1.0)
        
        # Optimizers
        self.main_optimizer = optim.Adam(self.main_q_network.parameters(), lr=learning_rate)
        self.road_optimizer = optim.Adam(self.road_q_network.parameters(), lr=learning_rate)
        self.settlement_optimizer = optim.Adam(self.settlement_q_network.parameters(), lr=learning_rate)
        self.bank_trade_optimizer = optim.Adam(self.bank_trade_q_network.parameters(), lr=learning_rate)
        self.player_trade_optimizer = optim.Adam(self.player_trade_q_network.parameters(), lr=learning_rate)
        self.player_trade_response_optimizer = optim.Adam(self.player_trade_response_q_network.parameters(), lr=learning_rate)
        
        # Replay memory for each action type
        self.main_memory = deque(maxlen=memory_size)
        self.road_memory = deque(maxlen=memory_size)
        self.settlement_memory = deque(maxlen=memory_size)
        self.bank_trade_memory = deque(maxlen=memory_size)
        self.player_trade_memory = deque(maxlen=memory_size)
        self.player_trade_response_memory = deque(maxlen=memory_size)
        
        # Anti-stalemate mechanisms 
        self.consecutive_end_turns = 0
        self.consecutive_action_failures = {
            "road": 0,
            "settlement": 0,
            "bank_trade": 0,
            "player_trade": 0
        }
        self.max_failures = 5
        self.last_actions = deque(maxlen=10)  # Track recent actions to detect loops
        self.rejected_trades = []
        self.max_rejected_trades = 10
        
        # Current trade offers
        self.current_offer = None
        self.counter_offer_count = 0
        self.max_counter_offers = 2
        
        # Whether we are in training mode
        self.training = True
    
    def update_target_networks(self, tau=0.001):
        """Soft update target networks."""
        self._soft_update(self.main_q_network, self.main_target_network, tau)
        self._soft_update(self.road_q_network, self.road_target_network, tau)
        self._soft_update(self.settlement_q_network, self.settlement_target_network, tau)
        self._soft_update(self.bank_trade_q_network, self.bank_trade_target_network, tau)
        self._soft_update(self.player_trade_q_network, self.player_trade_target_network, tau)
        self._soft_update(self.player_trade_response_q_network, self.player_trade_response_target_network, tau)
    
    def _soft_update(self, local_model, target_model, tau):
        """Soft update model parameters: θ_target = τ*θ_local + (1-τ)*θ_target."""
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)
    
    def convert_s(self, pos):
        """Convert position index to HexCompEnum for sides (roads)."""
        match pos:
            case 0:
                return HexCompEnum.S1
            case 1:
                return HexCompEnum.S2
            case 2:
                return HexCompEnum.S3
            case 3:
                return HexCompEnum.S4
            case 4:
                return HexCompEnum.S5
            case 5:
                return HexCompEnum.S6
            case _:
                return None
        
    def convert_e(self, pos):
        """Convert position index to HexCompEnum for edges (settlements)."""
        match pos:
            case 0:
                return HexCompEnum.E1
            case 1:
                return HexCompEnum.E2
            case 2:
                return HexCompEnum.E3
            case 3:
                return HexCompEnum.E4
            case 4:
                return HexCompEnum.E5
            case 5:
                return HexCompEnum.E6
            case _:
                return None
    
    def select_main_action(self, state, board, env):
        """Select one of the main actions: build road, build settlement, trade with player, trade with bank, end turn."""
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        # With probability epsilon, select a random action
        if not self.training or np.random.random() > self.epsilon:
            with torch.no_grad():
                q_values = self.main_q_network(state_tensor)
                action = torch.argmax(q_values).item()
        else:
            action = np.random.randint(0, self.main_action_size)
        
        # Check if actions are valid before proceeding
        player = board.players[self.player_index]
        
        # Road requires wood and brick
        if action == 0 and not player.cost_check(Structure.ROAD):
            # Can't afford, try another action
            other_actions = [1, 2, 3, 4]
            action = np.random.choice(other_actions)
        
        # Settlement requires wood, brick, sheep, wheat
        if action == 1 and not player.cost_check(Structure.SETTLEMENT):
            # Can't afford, try another action
            other_actions = [0, 2, 3, 4]
            action = np.random.choice(other_actions)
        
        # Anti-stalemate: If we've taken too many consecutive end turns, favor other actions
        if action == 4:  # End Turn
            self.consecutive_end_turns += 1
            if self.consecutive_end_turns > 5 and np.random.random() < 0.5:
                # Try to pick a non-end-turn action
                other_actions = [0, 1, 2, 3]
                action = np.random.choice(other_actions)
        else:
            self.consecutive_end_turns = 0
        
        # Track action for loop detection
        self.last_actions.append(action)
        
        # Check if we're stuck in an action loop
        if len(self.last_actions) == 10 and len(set(self.last_actions)) <= 2:
            # If we're alternating between just 2 actions, try something else
            all_actions = list(range(self.main_action_size))
            recent_actions = list(set(self.last_actions))
            other_actions = [a for a in all_actions if a not in recent_actions]
            if other_actions:
                action = np.random.choice(other_actions)
        
        return action

    def select_road_action(self, state, board, env):
        """Select a position to build a road."""
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        if not self.training or np.random.random() > self.epsilon:
            with torch.no_grad():
                q_values = self.road_q_network(state_tensor)
                action = torch.argmax(q_values).item()
        else:
            action = np.random.randint(0, self.road_action_size)
        
        # Check if this is a valid road position
        candidate_side = board.all_sides[action]
        player = board.players[self.player_index]
        
        # Check if the position is empty and follows placement rules
        try:
            is_valid = (
                candidate_side.parent.pos_is_empty(self.convert_s(candidate_side.n), Structure.ROAD) and
                candidate_side.parent.check_nearby(self.convert_s(candidate_side.n), Structure.ROAD, player, board.turn_number) and
                player.cost_check(Structure.ROAD) and
                player.max_struct_check(Structure.ROAD)
            )
        except:
            is_valid = False
        
        # If invalid action, increment failure counter and try again or cancel
        if not is_valid:
            self.consecutive_action_failures["road"] += 1
            if self.consecutive_action_failures["road"] >= self.max_failures:
                # Reset counter and cancel the action
                self.consecutive_action_failures["road"] = 0
                return -1
            
            # Try to find any valid road position
            valid_positions = []
            for side_idx in range(self.road_action_size):
                side = board.all_sides[side_idx]
                try:
                    if (side.parent.pos_is_empty(self.convert_s(side.n), Structure.ROAD) and
                        side.parent.check_nearby(self.convert_s(side.n), Structure.ROAD, player, board.turn_number) and
                        player.cost_check(Structure.ROAD) and
                        player.max_struct_check(Structure.ROAD)):
                        valid_positions.append(side_idx)
                except:
                    continue
            
            if valid_positions:
                return np.random.choice(valid_positions)
            else:
                # No valid positions, cancel the action
                return -1
        
        # Reset failure counter on valid action
        self.consecutive_action_failures["road"] = 0
        return action
        
    def select_settlement_action(self, state, board, env):
        """Select a position to build a settlement."""
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        if not self.training or np.random.random() > self.epsilon:
            with torch.no_grad():
                q_values = self.settlement_q_network(state_tensor)
                action = torch.argmax(q_values).item()
        else:
            action = np.random.randint(0, self.settlement_action_size)
        
        # Check if this is a valid settlement position
        candidate_edge = board.all_edges[action]
        player = board.players[self.player_index]
        
        # Check if the position is empty and follows placement rules
        try:
            is_valid = (
                candidate_edge.parent.pos_is_empty(self.convert_e(candidate_edge.n), Structure.SETTLEMENT) and
                candidate_edge.parent.check_nearby(self.convert_e(candidate_edge.n), Structure.SETTLEMENT, player, board.turn_number) and
                player.cost_check(Structure.SETTLEMENT) and
                player.max_struct_check(Structure.SETTLEMENT)
            )
        except:
            is_valid = False
        
        # If invalid action, increment failure counter and try again or cancel
        if not is_valid:
            self.consecutive_action_failures["settlement"] += 1
            if self.consecutive_action_failures["settlement"] >= self.max_failures:
                # Reset counter and cancel the action
                self.consecutive_action_failures["settlement"] = 0
                return -1
            
            # Try to find any valid settlement position
            valid_positions = []
            for edge_idx in range(self.settlement_action_size):
                edge = board.all_edges[edge_idx]
                try:
                    if (edge.parent.pos_is_empty(self.convert_e(edge.n), Structure.SETTLEMENT) and
                        edge.parent.check_nearby(self.convert_e(edge.n), Structure.SETTLEMENT, player, board.turn_number) and
                        player.cost_check(Structure.SETTLEMENT) and
                        player.max_struct_check(Structure.SETTLEMENT)):
                        valid_positions.append(edge_idx)
                except:
                    continue
            
            if valid_positions:
                return np.random.choice(valid_positions)
            else:
                # No valid positions, cancel the action
                return -1
        
        # Reset failure counter on valid action
        self.consecutive_action_failures["settlement"] = 0
        return action
        
    def select_bank_trade_action(self, state, board, env):
        """Select a bank trade action (which resources to offer and request)."""
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        if not self.training or np.random.random() > self.epsilon:
            with torch.no_grad():
                q_values = self.bank_trade_q_network(state_tensor)
                action_idx = torch.argmax(q_values).item()
        else:
            action_idx = np.random.randint(0, self.bank_trade_action_size)
        
        # Convert action index to resource combinations (4x4 combinations)
        # For simplicity, we'll use a 2:1 trade ratio and ensure trade is valid
        
        offer_resource = action_idx // 4  # Resource offered to bank (0-3)
        request_resource = action_idx % 4  # Resource requested from bank (0-3)
        
        # We can't trade the same resource
        if offer_resource == request_resource:
            self.consecutive_action_failures["bank_trade"] += 1
            
            # Try to find a valid bank trade
            player = board.players[self.player_index]
            inventory = np.array(player.inventory)
            
            # Find resources we have at least 2 of (for 2:1 trade)
            valid_offers = []
            for i in range(4):
                if inventory[i] >= 2:
                    valid_offers.append(i)
            
            if not valid_offers:
                return -1  # No valid trades possible
            
            # Pick a random offer and request (different resources)
            offer_resource = np.random.choice(valid_offers)
            request_options = [i for i in range(4) if i != offer_resource]
            request_resource = np.random.choice(request_options)
        
        # Create the trade action array
        offered = np.zeros(4, dtype=np.int32)
        requested = np.zeros(4, dtype=np.int32)
        
        # For bank trades: offer 2 of one resource, get 1 of another
        offered[offer_resource] = 2
        requested[request_resource] = 1
        
        # Check if the player can afford this trade
        player = board.players[self.player_index]
        can_afford = player.inventory[offer_resource] >= 2
        
        if not can_afford:
            self.consecutive_action_failures["bank_trade"] += 1
            if self.consecutive_action_failures["bank_trade"] >= self.max_failures:
                self.consecutive_action_failures["bank_trade"] = 0
                return -1
            
            return self.select_bank_trade_action(state, board, env)  # Try again
            
        # Reset failure counter on valid action
        self.consecutive_action_failures["bank_trade"] = 0
        
        return np.stack([offered, requested])
        
    def select_player_trade_action(self, state, board, env):
        """Select a player trade action (which resources to offer and request from another player)."""
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        if not self.training or np.random.random() > self.epsilon:
            with torch.no_grad():
                q_values = self.player_trade_q_network(state_tensor)
                action_idx = torch.argmax(q_values).item()
        else:
            action_idx = np.random.randint(0, self.player_trade_action_size)
        
        # Convert action index to resource combinations
        # Each player has 4 resource types, so we have 16 possible combinations
        offer_resource = action_idx // 4  # Resource offered to other player (0-3)
        request_resource = action_idx % 4  # Resource requested from other player (0-3)
        
        # Create the trade action array
        offered = np.zeros(4, dtype=np.int32)
        requested = np.zeros(4, dtype=np.int32)
        
        # For player trades: offer 1 of one resource, get 1 of another
        # We could make this more complex, but 1:1 trades are a good starting point
        offered[offer_resource] = 1
        requested[request_resource] = 1
        
        # Check if we can afford this trade
        player = board.players[self.player_index]
        my_inventory = np.array(player.inventory)
        
        # Check if opponent has the requested resource
        opponent_idx = (self.player_index + 1) % 2
        opponent = board.players[opponent_idx]
        opponent_inventory = np.array(opponent.inventory)
        
        can_afford = my_inventory[offer_resource] >= 1
        opponent_can_afford = opponent_inventory[request_resource] >= 1
        
        # Don't offer the same resource we're requesting
        if offer_resource == request_resource or not can_afford or not opponent_can_afford:
            self.consecutive_action_failures["player_trade"] += 1
            
            # Try to find a valid player trade
            valid_offers = []
            for i in range(4):
                if my_inventory[i] >= 1:
                    valid_offers.append(i)
            
            valid_requests = []
            for i in range(4):
                if opponent_inventory[i] >= 1 and i not in valid_offers:
                    valid_requests.append(i)
            
            if not valid_offers or not valid_requests:
                self.consecutive_action_failures["player_trade"] = 0
                return -1  # No valid trades possible
            
            # Create a new trade offer
            offer_resource = np.random.choice(valid_offers)
            request_resource = np.random.choice(valid_requests)
            
            offered = np.zeros(4, dtype=np.int32)
            requested = np.zeros(4, dtype=np.int32)
            
            offered[offer_resource] = 1
            requested[request_resource] = 1
        
        # Store the current offer for use in followup actions
        self.current_offer = np.stack([offered, requested])
        
        # Check if we've been trading too much without success
        if self.consecutive_action_failures["player_trade"] >= self.max_failures:
            self.consecutive_action_failures["player_trade"] = 0
            return -1
        
        # Reset failure counter on valid action
        self.consecutive_action_failures["player_trade"] = 0
        
        return self.current_offer
        
    def select_player_trade_response(self, state, board, env):
        """Respond to a trade offer (accept, reject, or counter)."""
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        if not self.training or np.random.random() > self.epsilon:
            with torch.no_grad():
                q_values = self.player_trade_response_q_network(state_tensor)
                action = torch.argmax(q_values).item()
        else:
            action = np.random.randint(0, self.player_trade_response_size)
        
        # If we have a saved offer, evaluate it to make a more informed decision
        if hasattr(self, 'current_offer') and self.current_offer is not None:
            offered = self.current_offer[1]  # What we would receive
            requested = self.current_offer[0]  # What we would give
            
            # Calculate the net gain/loss
            net_change = np.sum(offered) - np.sum(requested)
            
            # If we've reached the maximum number of counter offers, force a decision
            if self.counter_offer_count >= self.max_counter_offers:
                # Reset counter
                self.counter_offer_count = 0
                
                # Only accept if clearly beneficial after many counters
                if net_change > 0:
                    return 0  # Accept
                else:
                    return 1  # Reject
            
            # If the trade is beneficial, accept it
            if net_change > 0:
                self.counter_offer_count = 0  # Reset counter
                return 0  # Accept
            elif net_change == 0:
                # Neutral trade - randomly decide to break cycles
                if np.random.random() < 0.5:
                    return 0  # Accept
                else:
                    return 1  # Reject
            else:
                # Bad trade - counter or reject
                if action == 2:  # Model decided to counter
                    self.counter_offer_count += 1  # Track counters
                    return 2  # Counter
                else:
                    return 1  # Reject
        
        # Default behavior if no current offer is available
        if action == 2 and np.random.random() < 0.5:  # 50% chance to convert counter to reject
            return 1  # Reject instead of counter
            
        return action
    
    def create_counter_offer(self, state, board, env):
        """Create a counter offer in response to a trade proposal."""
        # Get inventories of both players
        player = board.players[self.player_index]
        my_inventory = np.array(player.inventory)
        
        opponent_idx = (self.player_index + 1) % 2
        opponent = board.players[opponent_idx]
        opponent_inventory = np.array(opponent.inventory)
        
        # If we have a previous offer, modify it
        if hasattr(self, 'current_offer') and self.current_offer is not None:
            original_offer = self.current_offer
            they_offered = original_offer[1]  # What we would receive
            they_requested = original_offer[0]  # What we would give
            
            # Try to improve the offer in our favor
            # Strategy: Reduce what we give OR increase what we get
            
            # Option 1: Try to reduce what we give
            we_offer = np.copy(they_requested)
            we_request = np.copy(they_offered)
            
            # Find what we're offering and reduce it if possible
            for i in range(4):
                if we_offer[i] > 0:
                    we_offer[i] = max(0, we_offer[i] - 1)
                    break
            
            # Make sure we're still offering something
            if np.sum(we_offer) == 0:
                for i in range(4):
                    if my_inventory[i] > 0:
                        we_offer[i] = 1
                        break
            
            # Option 2: Try to increase what we get
            for i in range(4):
                if opponent_inventory[i] > we_request[i]:
                    we_request[i] += 1
                    break
            
            # Ensure the trade is possible
            can_afford = True
            for i in range(4):
                if we_offer[i] > my_inventory[i]:
                    can_afford = False
                    break
                
            opponent_can_afford = True
            for i in range(4):
                if we_request[i] > opponent_inventory[i]:
                    opponent_can_afford = False
                    break
            
            if can_afford and opponent_can_afford:
                counter_offer = np.stack([we_offer, we_request]).astype(np.int32)
                return counter_offer
        
        # Fallback: create a brand new trade offer
        return self.select_player_trade_action(state, board, env)
    
    def act(self, state, board, env):
        """Main action selection method called by the environment."""
        # First, check if this is the initialization phase (turn_number == 0)
        if board.turn_number == 0:
            # Determine if we need to build a settlement or road based on init_phase
            # The pattern is: P1 settlement, P1 road, P2 settlement, P2 road, 
            #                P2 settlement2, P2 road2, P1 settlement2, P1 road2
            
            if hasattr(env, 'init_phase'):
                is_settlement = (env.init_phase % 2 == 0)
                
                if is_settlement:
                    """# Choose settlement position in initial phase
                    # For simplicity, we'll use a special initialization strategy
                    
                    # Find best settlement position based on production value
                    best_position = 0
                    best_value = -1
                    
                    for edge_idx in range(24):  # 24 potential settlement locations
                        edge = board.all_edges[edge_idx]
                        
                        # Check if this is a legal placement
                        try:
                            if not edge.parent.pos_is_empty(self.convert_e(edge.n), Structure.SETTLEMENT):
                                continue
                            
                            # For initial settlements, check the distance rule
                            if not edge.parent.check_nearby(self.convert_e(edge.n), Structure.SETTLEMENT, 
                                                         board.players[self.player_index], board.turn_number):
                                continue
                            
                            # Simple valuation: Sum the probability values of adjacent hexes
                            value = 0
                            parent_hex = edge.parent
                            
                            # Get the hex number (probability)
                            hex_num = parent_hex.tile_num
                            if hex_num > 0 and parent_hex.biome != Biome.DESERT:
                                # Higher value for 6 and 8 (common rolls)
                                if hex_num == 6 or hex_num == 8:
                                    value += 5
                                elif hex_num == 5 or hex_num == 9:
                                    value += 4
                                elif hex_num == 4 or hex_num == 10:
                                    value += 3
                                elif hex_num == 3 or hex_num == 11:
                                    value += 2
                                else:
                                    value += 1
                            
                            # Check adjacent hexes if they have links
                            for linked in edge.links:
                                if linked and linked.parent != parent_hex:
                                    linked_hex = linked.parent
                                    hex_num = linked_hex.tile_num
                                    
                                    if hex_num > 0 and linked_hex.biome != Biome.DESERT:
                                        if hex_num == 6 or hex_num == 8:
                                            value += 5
                                        elif hex_num == 5 or hex_num == 9:
                                            value += 4
                                        elif hex_num == 4 or hex_num == 10:
                                            value += 3
                                        elif hex_num == 3 or hex_num == 11:
                                            value += 2
                                        else:
                                            value += 1
                            
                            if value > best_value:
                                best_value = value
                                best_position = edge_idx
                                
                        except:
                            continue
                    
                    return best_position"""
                    return self.select_settlement_action(state, board, env)
                
                else:  # Road placement in initial phase
                    # Choose road adjacent to our most recently placed settlement
                    """player = board.players[self.player_index]
                    
                    if player.settlements:
                        last_settlement = player.settlements[-1]
                        parent_hex = last_settlement.parent
                        edge_idx = last_settlement.n
                        
                        # Find adjacent sides (potential road locations)
                        for side_idx in range(30):
                            side = board.all_sides[side_idx]
                            
                            # Check if side is connected to the last settlement and is empty
                            try:
                                if (side.parent == parent_hex and 
                                    (side.n == edge_idx or (side.n + 1) % 6 == edge_idx or (side.n - 1) % 6 == edge_idx) and
                                    side.parent.pos_is_empty(self.convert_s(side.n), Structure.ROAD) and
                                    side.parent.check_nearby(self.convert_s(side.n), Structure.ROAD, player, board.turn_number)):
                                    return side_idx
                            except:
                                continue"""
                    road = self.select_road_action(state, board, env)
                    if road < 0:
                        # Fallback: choose first available legal road location
                        for side_idx in range(30):
                            side = board.all_sides[side_idx]
                            try:
                                if (side.parent.pos_is_empty(self.convert_s(side.n), Structure.ROAD) and
                                    side.parent.check_nearby(self.convert_s(side.n), Structure.ROAD, 
                                                        board.players[self.player_index], board.turn_number)):
                                    return side_idx
                            except:
                                continue
                    return road

            #return 0  # Default action for initial phase
            
        # Normal game phase - determine what action type is required
        # Check for follow-up actions from previous decisions
        if env.waiting_for_road_build_followup:
            return self.select_road_action(state, board, env)
            
        elif env.waiting_for_settlement_build_followup:
            return self.select_settlement_action(state, board, env)
            
        elif env.waiting_for_b_trade_followup:
            return self.select_bank_trade_action(state, board, env)
            
        elif env.waiting_for_p_trade_followup_1:
            if env.reply_to_offer:
                # Store the offer from the environment if available
                if hasattr(env, 'offer') and env.offer is not None:
                    self.current_offer = env.offer
                
                # Respond to trade offer (accept, reject, counter)
                return self.select_player_trade_response(state, board, env)
            elif env.counter_sent:
                # Create counter offer
                return self.create_counter_offer(state, board, env)
            else:
                # Initiate player trade
                return self.select_player_trade_action(state, board, env)
        
        elif env.waiting_for_p_trade_followup_2:
            # Respond to counter offer
            return self.select_player_trade_response(state, board, env)
            
        elif env.waiting_for_p_trade_followup_3:
            if env.reply_to_offer:
                # Respond to counter-counter offer
                return self.select_player_trade_response(state, board, env)
            else:
                # Create counter-counter offer
                return self.create_counter_offer(state, board, env)
        
        # No follow-up, select main action
        return self.select_main_action(state, board, env)
    
    def store_experience(self, memory, state, action, reward, next_state, done):
        """Store experience in replay memory with automatic state adjustment."""
        try:
            # Convert state and next_state to numpy arrays and flatten them
            state = np.array(state, dtype=np.float32).flatten()
            next_state = np.array(next_state, dtype=np.float32).flatten()
            
            expected_size = self.obs_space_size

            # Pad or trim state if necessary
            if state.size < expected_size:
                padded = np.zeros(expected_size, dtype=np.float32)
                padded[:state.size] = state
                state = padded
                print(f"Padded state from size {state.size} to {expected_size}")
            elif state.size > expected_size:
                state = state[:expected_size]
                print(f"Trimmed state from size {state.size + (state.size - expected_size)} to {expected_size}")
            
            if next_state.size < expected_size:
                padded = np.zeros(expected_size, dtype=np.float32)
                padded[:next_state.size] = next_state
                next_state = padded
                print(f"Padded next_state from size {next_state.size} to {expected_size}")
            elif next_state.size > expected_size:
                next_state = next_state[:expected_size]
                print(f"Trimmed next_state from size {next_state.size + (next_state.size - expected_size)} to {expected_size}")

            # Use deep copies to prevent reference issues
            state_copy = state.copy()
            next_state_copy = next_state.copy()

            # Process action if it's an array
            action_copy = action.copy() if isinstance(action, np.ndarray) else action

            # Store the experience
            memory.append([state_copy, action_copy, reward, next_state_copy, done])
            #print(f"Successfully stored experience. Memory size now: {len(memory)}")
            return True

        except Exception as e:
            print(f"Error storing experience: {str(e)}")
            import traceback
            traceback.print_exc()
            return False


    
    def learn_from_memory(self, memory, q_network, target_network, optimizer, action_idx=None):
        """Train the Q-network using a batch of experiences from memory."""
        if len(memory) < self.batch_size:
            return None
        
        try:
            # Sample random batch of experiences
            batch = random.sample(memory, self.batch_size)
            
            # Unpack experiences
            states, actions, rewards, next_states, dones = zip(*batch)
            
            # Convert to tensors - ensure consistent shapes
            state_shape = states[0].shape
            
            # Filter out experiences with inconsistent shapes
            valid_indices = [i for i, s in enumerate(states) if s.shape == state_shape]
            if len(valid_indices) < self.batch_size // 2:
                print(f"Warning: Not enough valid experiences with shape {state_shape}")
                return None
                
            # Use only valid indices
            states = np.array([states[i] for i in valid_indices])
            rewards = np.array([rewards[i] for i in valid_indices])
            next_states = np.array([next_states[i] for i in valid_indices])
            dones = np.array([dones[i] for i in valid_indices])
            
            # Handle actions
            if action_idx is not None:
                # For pre-processed action indices
                if np.isscalar(action_idx):
                    # If scalar, use same value for all instances
                    actions = torch.LongTensor([action_idx] * len(valid_indices)).to(self.device)
                else:
                    # If list/array, select the indices like before
                    actions = torch.LongTensor([action_idx[i] for i in valid_indices]).to(self.device)
            else:
                # Process actions based on their type
                action_list = []
                for i in valid_indices:
                    a = actions[i]
                    if isinstance(a, (np.ndarray, list)):
                        # For array actions, convert to index 
                        if len(a) > 0:
                            if isinstance(a[0], (np.ndarray, list)):
                                # For complex actions like trades
                                # Just use 0 as fallback
                                idx = 0
                            else:
                                idx = int(a[0])
                        else:
                            idx = 0
                    else:
                        idx = int(a)
                    action_list.append(idx)
                
                actions = torch.LongTensor(action_list).to(self.device)
            
            # Convert to PyTorch tensors
            states = torch.FloatTensor(states).to(self.device)
            next_states = torch.FloatTensor(next_states).to(self.device)
            rewards = torch.FloatTensor(rewards).to(self.device)
            dones = torch.FloatTensor(dones).to(self.device)
            
            # Get current Q values
            curr_q_values = q_network(states)
            
            # Get next Q values from target network
            next_q_values = target_network(next_states).detach()
            
            # Compute target Q values
            max_next_q_values = torch.max(next_q_values, dim=1)[0]
            target_q_values = rewards + (1 - dones) * self.gamma * max_next_q_values
            
            # Get the Q values for the actions taken
            batch_size = len(valid_indices)
            q_range = curr_q_values.shape[1]
            actions = torch.clamp(actions, 0, q_range - 1)  # Ensure actions are within valid range
            pred_q_values = curr_q_values.gather(1, actions.unsqueeze(1)).squeeze(1)
            
            # Compute loss
            loss = nn.MSELoss()(pred_q_values, target_q_values)
            
            # Optimize the model
            optimizer.zero_grad()
            loss.backward()
            # Clip gradients to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(q_network.parameters(), 1)
            optimizer.step()
            
            return loss.item()
        
        except Exception as e:
            print(f"Error in learn_from_memory: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def learn(self, state, action, reward, next_state, done, action_type):
        """Learn from a transition based on the action type with improved error handling."""
        if not self.training:
            return None
            
        try:
            # Validate action_type
            if action_type not in ["main", "road", "settlement", "bank_trade", "player_trade", "player_trade_response"]:
                print(f"Warning: Unknown action_type: {action_type}")
                action_type = "main"  # Default to main
            
            # Get the appropriate memory
            memory_attr = f"{action_type}_memory"
            if not hasattr(self, memory_attr):
                print(f"Error: Agent has no attribute {memory_attr}")
                return None
                
            memory = getattr(self, memory_attr)
            
            # Convert complex actions to indices for certain action types
            action_idx = None
            if action_type in ["bank_trade", "player_trade"] and isinstance(action, np.ndarray) and action.shape == (2, 4):
                offered = action[0]
                requested = action[1]
                
                # Find the indices of the max values
                offer_idx = np.argmax(offered) if np.max(offered) > 0 else 0
                request_idx = np.argmax(requested) if np.max(requested) > 0 else 0
                
                action_idx = offer_idx * 4 + request_idx
            
            # Get the appropriate networks
            if action_type == "main":
                q_network = self.main_q_network
                target_network = self.main_target_network
                optimizer = self.main_optimizer
            elif action_type == "road":
                q_network = self.road_q_network
                target_network = self.road_target_network
                optimizer = self.road_optimizer
            elif action_type == "settlement":
                q_network = self.settlement_q_network
                target_network = self.settlement_target_network
                optimizer = self.settlement_optimizer
            elif action_type == "bank_trade":
                q_network = self.bank_trade_q_network
                target_network = self.bank_trade_target_network
                optimizer = self.bank_trade_optimizer
            elif action_type == "player_trade":
                q_network = self.player_trade_q_network
                target_network = self.player_trade_target_network
                optimizer = self.player_trade_optimizer
            elif action_type == "player_trade_response":
                q_network = self.player_trade_response_q_network
                target_network = self.player_trade_response_target_network
                optimizer = self.player_trade_response_optimizer
            else:
                return None
            
            # Store experience (without learning again, as we'll do that below)
            self.store_experience(memory, state, action if action_idx is None else action_idx, reward, next_state, done)
            
            # Learn from memory
            loss = self.learn_from_memory(memory, q_network, target_network, optimizer, action_idx)
            
            # Update target networks
            self.update_target_networks()
            
            # Decay exploration rate
            self.decay_epsilon()
            
            return loss
            
        except Exception as e:
            print(f"Error in learn method: {str(e)}")
            import traceback
            traceback.print_exc()
            return None
    
    def train(self):
        """Set agent to training mode."""
        self.training = True
        
    def eval(self):
        """Set agent to evaluation mode."""
        self.training = False
        
    def decay_epsilon(self):
        """Decay exploration rate."""
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
    
    def save_model(self, path):
        """Save all Q-networks to disk."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        torch.save({
            'main_q_network': self.main_q_network.state_dict(),
            'road_q_network': self.road_q_network.state_dict(),
            'settlement_q_network': self.settlement_q_network.state_dict(),
            'bank_trade_q_network': self.bank_trade_q_network.state_dict(),
            'player_trade_q_network': self.player_trade_q_network.state_dict(),
            'player_trade_response_q_network': self.player_trade_response_q_network.state_dict(),
            'epsilon': self.epsilon
        }, path)
        
    def load_model(self, path):
        """Load all Q-networks from disk."""
        if not os.path.exists(path):
            print(f"No model found at {path}")
            return
        
        checkpoint = torch.load(path, map_location=self.device)
        
        self.main_q_network.load_state_dict(checkpoint['main_q_network'])
        self.road_q_network.load_state_dict(checkpoint['road_q_network'])
        self.settlement_q_network.load_state_dict(checkpoint['settlement_q_network'])
        self.bank_trade_q_network.load_state_dict(checkpoint['bank_trade_q_network'])
        self.player_trade_q_network.load_state_dict(checkpoint['player_trade_q_network'])
        self.player_trade_response_q_network.load_state_dict(checkpoint['player_trade_response_q_network'])
        
        # Update target networks
        self.update_target_networks(tau=1.0)
        
        # Set epsilon from saved model
        if 'epsilon' in checkpoint:
            self.epsilon = checkpoint['epsilon']