import numpy as np
from mini_catan.enums import Structure, Biome, HexCompEnum

class GreedyAgent:
    def __init__(self, player_index=None):
        """
        Initialize the Greedy Agent.
        
        Args:
            player_index (int, optional): The player ID (0 or 1) of this agent. Defaults to None.
        """
        self.player_index = player_index
        
        # Failure counters to prevent action loops
        self.trade_failures = 0
        self.bank_trade_failures = 0
        self.road_placement_failures = 0
        self.settlement_placement_failures = 0
        self.MAX_FAILURES = 3  # Maximum failures before giving up on an action
        
        # Counter for trade negotiations
        self.counter_offer_count = 0
        self.MAX_COUNTER_OFFERS = 2  # Maximum number of counter offers before accepting or rejecting
        
        # Track rejected trades to avoid repeating them
        self.rejected_trades = []
        self.max_rejected_trades = 10  # Maximum number of rejected trades to remember
        
        # Anti-stalemate measures
        self.consecutive_end_turns = 0
        self.turns_since_last_build = 0
        self.stuck_threshold = 5  # After this many end turns, adjust strategy

    def convert_s(self, pos):
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
        
    def decode_observation(self, obs, board):
        """
        Decode the observation array into a more usable format.
        
        Args:
            obs (numpy.array): The observation from the environment.
            board: The game board object.
            
        Returns:
            dict: A dictionary containing parsed observation data.
        """
        idx = 0
        num_players = 2  # Fixed for Mini Catan
        
        inventories = obs[idx:(idx + num_players * 4)].reshape(num_players, 4)
        idx += num_players * 4
        
        edges = obs[idx:(idx + 24)].reshape(24, )
        idx += 24
        
        sides = obs[idx:(idx + 30)].reshape(30, )
        idx += 30
        
        longest_road_owner = obs[idx]
        idx += 1
        
        victory_points = obs[idx:(idx + num_players)].reshape(num_players, )
        idx += num_players
        
        biomes = obs[idx:(idx + board.board_size)]
        idx += board.board_size
        
        hex_nums = obs[idx:(idx + board.board_size)]
        idx += board.board_size
        
        robber_loc = obs[idx]
        idx += 1
        
        turn_number = obs[idx]
        idx += 1
        
        b_trade_followup = obs[idx]
        idx += 1
        
        p_trade_followup_1 = obs[idx]
        idx += 1
        
        p_trade_followup_2 = obs[idx]
        idx += 1
        
        p_trade_followup_3 = obs[idx]
        idx += 1
        
        reply_to_offer = obs[idx]
        idx += 1
        
        counter_sent = obs[idx]
        
        return {
            "inventories": inventories,
            "edges": edges,
            "sides": sides,
            "longest_road_owner": longest_road_owner,
            "victory_points": victory_points,
            "biomes": biomes,
            "hex_nums": hex_nums,
            "robber_loc": robber_loc,
            "turn_number": turn_number,
            "b_trade_followup": b_trade_followup,
            "p_trade_followup_1": p_trade_followup_1,
            "p_trade_followup_2": p_trade_followup_2,
            "p_trade_followup_3": p_trade_followup_3,
            "reply_to_offer": reply_to_offer,
            "counter_sent": counter_sent
        }
        
    def evaluate_road(self, candidate_side_idx, board, player_idx):
        """
        Evaluate the reward for building a road at the given position.
        
        Args:
            candidate_side_idx (int): Index of the candidate side (road position).
            board: The game board object.
            player_idx (int): The index of the current player.
            
        Returns:
            float: The estimated reward for building a road at this position.
        """
        player = board.players[player_idx]
        
        # Check if the player can afford a road
        if not player.cost_check(Structure.ROAD):
            return -np.inf
            
        # Check if the player has reached their maximum road limit
        if not player.max_struct_check(Structure.ROAD):
            return -np.inf
            
        # Get the side at the given index
        candidate_side = board.all_sides[candidate_side_idx]
        
        # Check if the position is empty
        if not candidate_side.parent.pos_is_empty(self.convert_s(candidate_side.n), Structure.ROAD):
            return -np.inf
            
        # Check if the road can be placed according to the rules
        if not candidate_side.parent.check_nearby(self.convert_s(candidate_side.n), Structure.ROAD, player, board.turn_number):
            return -np.inf
            
        # Calculate the base reward for building a road
        s2r = player.get_player_s2r()  # Settlements to Roads ratio
        s_p = max(len(player.settlements), 1)  # Number of settlements
        base_reward = 1 + 1.15 * (s2r / s_p)
        
        # Simulate the potential impact on longest road
        original_roads = player.roads.copy()
        player.roads.append(candidate_side)
        
        longest = board.longest_road(player)
        additional_reward = 0
        
        # Restore original state
        player.roads = original_roads
        
        # Check if this road could earn the longest road bonus
        if longest >= board.min_longest_road and longest > board.current_longest_road:
            additional_reward = 2  # LONGEST_ROAD_REWARD
            
        # Anti-stalemate: If we've been stuck, make building more attractive
        if self.consecutive_end_turns >= self.stuck_threshold:
            base_reward += 0.2 * self.consecutive_end_turns
            
        return base_reward + additional_reward
        
    def evaluate_settlement(self, candidate_edge_idx, board, player_idx):
        """
        Evaluate the reward for building a settlement at the given position.
        
        Args:
            candidate_edge_idx (int): Index of the candidate edge (settlement position).
            board: The game board object.
            player_idx (int): The index of the current player.
            
        Returns:
            float: The estimated reward for building a settlement at this position.
        """
        player = board.players[player_idx]
        
        # Check if the player can afford a settlement
        if not player.cost_check(Structure.SETTLEMENT):
            return -np.inf
            
        # Check if the player has reached their maximum settlement limit
        if not player.max_struct_check(Structure.SETTLEMENT):
            return -np.inf
            
        # Get the edge at the given index
        candidate_edge = board.all_edges[candidate_edge_idx]
        
        # Check if the position is empty
        if not candidate_edge.parent.pos_is_empty(self.convert_e(candidate_edge.n), Structure.SETTLEMENT):
            return -np.inf
            
        # Check if the settlement can be placed according to the rules
        if not candidate_edge.parent.check_nearby(self.convert_e(candidate_edge.n), Structure.SETTLEMENT, player, board.turn_number):
            return -np.inf
            
        # Calculate the base reward for building a settlement
        s_p = len(player.settlements)
        s_max = 5  # Maximum number of settlements
        base_reward = 2 + 1.5 * (s_p / s_max)
        
        # Add extra weight to settlements based on expected resource production
        production_value = self.evaluate_settlement_production(candidate_edge, board)
        reward = base_reward + production_value
        
        # Settlements are generally more valuable early in the game
        # Add extra incentive to prioritize settlements over trades
        reward += 0.5
        
        # Anti-stalemate: If we've been stuck, make building settlements more attractive
        if self.consecutive_end_turns >= self.stuck_threshold:
            reward += 0.3 * self.consecutive_end_turns
            
        return reward
        
    def evaluate_settlement_production(self, edge, board):
        """
        Evaluate the production value of a settlement at the given edge.
        
        Args:
            edge: The edge where the settlement would be placed.
            board: The game board object.
            
        Returns:
            float: An estimate of the settlement's production value.
        """
        production_value = 0
        
        # Check each adjacent hex
        hex_block = edge.parent
        adjacent_hexes = [hex_block]
        
        # Add other hexes that share this edge
        for linked in edge.links:
            if linked and linked.parent not in adjacent_hexes:
                adjacent_hexes.append(linked.parent)
        
        for hex_block in adjacent_hexes:
            # Skip desert or hexes with robber
            if hex_block.biome == Biome.DESERT or board.robber_loc == board.map_hexblocks.index(hex_block):
                continue
                
            # Calculate probability factor (higher for numbers closer to 7)
            num = hex_block.tile_num
            if num <= 0:
                continue
                
            if num == 6:  # Desert
                probability = 0
            else:
                probability = 5 - abs(3 - num)
            
            # Add value based on biome and probability
            if hex_block.biome == Biome.FOREST:  # Wood
                production_value += 0.5 * probability
            elif hex_block.biome == Biome.HILLS:  # Brick
                production_value += 0.6 * probability  # Slightly more valuable
            elif hex_block.biome == Biome.PASTURE:  # Sheep
                production_value += 0.4 * probability
            elif hex_block.biome == Biome.FIELDS:  # Wheat
                production_value += 0.5 * probability
                
        return production_value
        
    def evaluate_bank_trade(self, obs_dict, player_idx):
        """
        Evaluate the best bank trade possible.
        
        Args:
            obs_dict (dict): The decoded observation.
            player_idx (int): The index of the current player.
            
        Returns:
            tuple: (reward, trade_action) where trade_action is a numpy array of shape (2, 4)
                  representing [offered_resources, requested_resources]
        """
        my_inventory = obs_dict["inventories"][player_idx]
        
        # No trades possible if we don't have at least 2 of any resource (bank trades are 2:1)
        if max(my_inventory) < 2:
            return -np.inf, None
            
        best_reward = -np.inf
        best_trade = None
        
        # Try all possible trades: offer 2 of something, get 1 of something else
        for offer_idx, offer_amount in enumerate(my_inventory):
            # Check if we have enough of this resource type for a bank trade
            if offer_amount < 2:  # Need at least 2 items for a bank trade
                continue
                
            # Calculate how many items we could get with a 2:1 ratio
            max_request_amount = min(offer_amount // 2, 1)  # Start with minimal 2:1 trade
            
            for request_idx in range(4):
                if request_idx == offer_idx:  # Can't trade for the same resource
                    continue
                    
                # Create the trade action - strictly follow 2:1 ratio
                offered = np.zeros(4, dtype=np.int32)
                requested = np.zeros(4, dtype=np.int32)
                
                # Offer exactly one type of resource
                offered[offer_idx] = 2 * max_request_amount  # This will be 2, 4, 6, etc.
                
                # Request exactly one type of resource
                requested[request_idx] = max_request_amount  # This will be 1, 2, 3, etc.
                
                # Verify if we can afford this trade
                can_afford = True
                for i in range(4):
                    if offered[i] > my_inventory[i]:
                        can_afford = False
                        break
                
                if not can_afford:
                    continue  # Skip trades we can't afford
                
                # Check if this trade is similar to previously rejected trades
                trade_signature = f"bank_o{offer_idx}_r{request_idx}"
                if trade_signature in self.rejected_trades and self.consecutive_end_turns < self.stuck_threshold:
                    continue  # Skip previously rejected trade patterns unless we're stuck
                
                # Calculate the reward with diminishing returns for bank trades
                d_r = requested - (offered / 2)  # net resource change with 2:1 ratio
                reward = 0.5 + (-np.exp(-0.5 * (0.25 + np.sum(d_r))))  # TRADE_BANK_REWARD
                
                # Adjust reward based on failure counts - make trading less attractive
                # after repeated failures
                reward -= 0.3 * self.bank_trade_failures
                
                # Anti-stalemate: Make bank trades more attractive when stuck
                if self.consecutive_end_turns >= self.stuck_threshold:
                    # Check if this trade would lead to completing a structure
                    if self.would_enable_building(my_inventory.copy(), offered, requested, player_idx):
                        reward += 1.0  # Significant boost for strategic trades
                    else:
                        reward += 0.2 * self.consecutive_end_turns  # Modest boost just to move resources
                
                if reward > best_reward:
                    best_reward = reward
                    best_trade = np.stack([offered, requested])
                    
        return best_reward, best_trade
    
    def would_enable_building(self, inventory, offered, requested, player_idx):
        """
        Check if this trade would enable building a structure.
        
        Args:
            inventory: Current inventory array
            offered: Resources to be offered
            requested: Resources to be requested
            player_idx: Player index
            
        Returns:
            bool: True if the trade enables building, False otherwise
        """
        # Update inventory with the trade
        updated_inv = inventory.copy()
        for i in range(4):
            updated_inv[i] -= offered[i]
            updated_inv[i] += requested[i]
        
        # Check if we can build a road after the trade
        # Road costs: 1 wood, 1 brick (indices 0, 1)
        road_possible = (updated_inv[0] >= 1 and updated_inv[1] >= 1)
        
        # Check if we can build a settlement after the trade
        # Settlement costs: 1 wood, 1 brick, 1 sheep, 1 wheat (indices 0, 1, 2, 3)
        settlement_possible = all(updated_inv >= 1)
        
        return road_possible or settlement_possible
        
    def evaluate_player_trade(self, obs_dict, player_idx):
        """
        Evaluate the best player trade possible.
        
        Args:
            obs_dict (dict): The decoded observation.
            player_idx (int): The index of the current player.
            
        Returns:
            tuple: (reward, trade_action) where trade_action is a numpy array of shape (2, 4)
                  representing [offered_resources, requested_resources]
        """
        my_inventory = obs_dict["inventories"][player_idx]
        opponent_idx = (player_idx + 1) % 2
        opponent_inventory = obs_dict["inventories"][opponent_idx]
        
        best_reward = -np.inf
        best_trade = None
        
        # If we're stuck, be willing to try more complex trades
        max_offer = 1 if self.consecutive_end_turns < self.stuck_threshold else 2
        
        # Try different trade combinations
        for offer_idx, offer_amount in enumerate(my_inventory):
            # Skip resource types we don't have
            if offer_amount < 1:
                continue
                
            for request_idx in range(4):
                # Skip same resource type
                if request_idx == offer_idx:
                    continue
                
                # Skip if opponent doesn't have this resource
                if opponent_inventory[request_idx] < 1:
                    continue
                
                # Try with different amounts when stuck
                max_offer_amount = min(offer_amount, max_offer)
                max_request_amount = min(opponent_inventory[request_idx], max_offer)
                
                for offer_qty in range(1, max_offer_amount + 1):
                    for request_qty in range(1, max_request_amount + 1):
                        # Create the trade action
                        offered = np.zeros(4)
                        requested = np.zeros(4)
                        
                        offered[offer_idx] = offer_qty
                        requested[request_idx] = request_qty
                        
                        # Verify if both players can afford this trade
                        i_can_afford = True
                        for i in range(4):
                            if offered[i] > my_inventory[i]:
                                i_can_afford = False
                                break
                                
                        opponent_can_afford = True
                        for i in range(4):
                            if requested[i] > opponent_inventory[i]:
                                opponent_can_afford = False
                                break
                        
                        # Skip this trade if either player can't afford it
                        if not (i_can_afford and opponent_can_afford):
                            continue
                        
                        # Check if this trade is similar to previously rejected trades
                        trade_signature = f"player_o{offer_idx}_r{request_idx}"
                        if trade_signature in self.rejected_trades and self.consecutive_end_turns < self.stuck_threshold:
                            continue  # Skip previously rejected trades unless we're stuck
                        
                        # Calculate the reward
                        d_r = requested - offered  # net resource change
                        reward = 0.8 + (-np.exp(-0.5 * (0.5 + np.sum(d_r))))  # TRADE_PLAYER_REWARD
                        
                        # Adjust reward based on failure counts
                        reward -= 0.3 * self.trade_failures
                        
                        # Anti-stalemate: Make trades more attractive when stuck
                        if self.consecutive_end_turns >= self.stuck_threshold:
                            # Check if this trade would complete a building
                            if self.would_enable_building(my_inventory.copy(), offered, requested, player_idx):
                                reward += 1.0  # Significant boost
                            else:
                                # Even for neutral or slightly negative trades, boost when stuck
                                reward += 0.1 * self.consecutive_end_turns
                        
                        if reward > best_reward:
                            best_reward = reward
                            best_trade = np.stack([offered, requested]).astype(np.int32)
                    
        return best_reward, best_trade
        
    def respond_to_trade_offer(self, obs_dict, board, player_idx):
        """
        Decide how to respond to a trade offer.
        
        Args:
            obs_dict (dict): The decoded observation.
            board: The game board object.
            player_idx (int): The index of the current player.
            
        Returns:
            int: 0 (accept), 1 (reject), or 2 (counter)
        """
        # If we have a saved offer, evaluate it
        if hasattr(self, 'current_offer') and self.current_offer is not None:
            offered = self.current_offer[1]  # What we would receive
            requested = self.current_offer[0]  # What we would give
            
            # Calculate the net gain
            d_r = offered - requested
            
            # If we've reached the maximum number of counter offers, force a decision
            if self.counter_offer_count >= self.MAX_COUNTER_OFFERS:
                # Reset counter
                self.counter_offer_count = 0
                
                # Make a more definitive decision - only accept if clearly beneficial
                if np.sum(d_r) > 0.5:  # Higher threshold for acceptance after many counters
                    return 0  # Accept
                else:
                    # Remember the trade type being rejected
                    offer_resource_type = None
                    request_resource_type = None
                    for i in range(4):
                        if requested[i] > 0:
                            offer_resource_type = i
                        if offered[i] > 0:
                            request_resource_type = i
                            
                    if offer_resource_type is not None and request_resource_type is not None:
                        trade_signature = f"player_o{offer_resource_type}_r{request_resource_type}"
                        if trade_signature not in self.rejected_trades:
                            self.rejected_trades.append(trade_signature)
                    
                    return 1  # Reject
            
            # Anti-stalemate: If we're stuck, evaluate if the trade enables building
            my_inventory = obs_dict["inventories"][player_idx]
            if self.consecutive_end_turns >= self.stuck_threshold:
                if self.would_enable_building(my_inventory, requested, offered, player_idx):
                    return 0  # Accept trades that enable building when stuck
                elif np.sum(d_r) >= -0.5:  # Accept even slightly unfavorable trades
                    return 0  # Accept
            
            # Normal evaluation if we haven't reached max counters
            if np.sum(d_r) > 0:
                self.counter_offer_count = 0  # Reset counter on acceptance
                return 0  # Accept
            elif np.sum(d_r) == 0:
                # If neutral, break the cycle with increasing probability
                # The more counter offers, the more likely to make a decision
                decision_probability = 0.5 + (0.25 * self.counter_offer_count)
                if np.random.random() < decision_probability:
                    self.counter_offer_count = 0  # Reset counter
                    return np.random.choice([0, 1])  # Randomly accept or reject
                else:
                    self.counter_offer_count += 1  # Increment counter
                    return 2  # Counter offer
            else:
                # If not beneficial, counter or reject
                if obs_dict["counter_sent"]:
                    self.counter_offer_count = 0  # Reset counter
                    return 1  # Reject if we've already sent a counter
                else:
                    self.counter_offer_count += 1  # Increment counter
                    
                    # With increasing probability, just reject instead of countering
                    reject_probability = 0.2 * self.counter_offer_count
                    if np.random.random() < reject_probability:
                        self.counter_offer_count = 0  # Reset counter
                        return 1  # Reject
                    else:
                        return 2  # Send counter offer
        
        # Default: reject
        return 1
        
    import numpy as np

    def create_counter_offer(self, obs_dict, board, player_idx):
        """
        Create a strategic counter offer for trading.
        
        Args:
            obs_dict (dict): The decoded observation.
            board: The game board object.
            player_idx (int): The index of the current player.
        
        Returns:
            numpy.array: A trade action of shape (2, 4) [offered_resources, requested_resources]
        """
        my_inventory = obs_dict["inventories"][player_idx]
        opponent_idx = (player_idx + 1) % 2
        opponent_inventory = obs_dict["inventories"][opponent_idx]
        
        # First, try to use our trade evaluation method
        _, trade = self.evaluate_player_trade(obs_dict, player_idx)
        
        # If we have a previous offer, we'll modify our strategy
        if hasattr(self, 'current_offer') and self.current_offer is not None:
            original_offer = self.current_offer
            they_offered = original_offer[1]  # What we would receive
            they_requested = original_offer[0]  # What we would give
            
            # Blend the evaluated trade with the counter-offer strategy
            if trade is not None:
                # Combine the evaluated trade with the counter-offer logic
                we_offer = np.copy(trade[0])
                we_request = np.copy(trade[1])
            else:
                we_offer = np.copy(they_requested)
                we_request = np.copy(they_offered)
            
            # Anti-stalemate mechanism
            if self.consecutive_end_turns >= self.stuck_threshold:
                # When stuck, be more flexible
                for i in range(4):
                    # Slightly adjust offer to make it more attractive
                    if my_inventory[i] > we_offer[i]:
                        we_offer[i] = min(we_offer[i] + 1, my_inventory[i])
                    if opponent_inventory[i] > we_request[i]:
                        we_request[i] = min(we_request[i] + 1, opponent_inventory[i])
            else:
                # Normal negotiation strategy
                # Try to reduce what we offer or increase what we request
                for i in range(4):
                    if we_offer[i] > 0 and my_inventory[i] >= we_offer[i] - 1:
                        we_offer[i] = max(1, we_offer[i] - 1)
                        break
                    
                    if opponent_inventory[i] > we_request[i]:
                        we_request[i] += 1
                        break
            
            # Final affordability check
            i_can_afford = my_inventory >= we_offer
            opponent_can_afford = opponent_inventory >= we_request
            
            # Ensure the trade is possible
            if np.all(i_can_afford) and np.all(opponent_can_afford):
                counter_offer = np.stack([we_offer, we_request]).astype(np.int32)
                return counter_offer
        
        # Fallback to the evaluated trade if available
        if trade is not None:
            return trade
        
        # If no viable trade is found
        return -1
    
    def act(self, obs, board, game):
        """
        Determine the best action based on the current observation.
        
        Args:
            obs (numpy.array): The observation from the environment.
            board: The game board object.
            game: The game environment object.
            
        Returns:
            The selected action.
        """
        # Determine which player we are if not already set
        if self.player_index is None:
            self.player_index = game.current_player
            
        player_idx = game.current_player
        obs_dict = self.decode_observation(obs, board)
        
        # Anti-stalemate: Periodically forget some rejected trades
        if board.turn_number % 10 == 0 and len(self.rejected_trades) > 0:
            # Forget half of the rejected trades every 10 turns
            self.rejected_trades = self.rejected_trades[:len(self.rejected_trades)//2]
        
        # Initial placement phase (special case)
        if board.turn_number == 0:
            if ((game.init_phase % 2) == 0):
                # Choose best settlement location for initial placement
                best_reward = -np.inf
                best_action = 0
                
                for edge_idx in range(24):  # 24 potential settlement locations
                    edge = board.all_edges[edge_idx]
                    
                    # Check if this is a legal placement
                    if not edge.parent.pos_is_empty(self.convert_e(edge.n), Structure.SETTLEMENT):
                        continue
                    
                    # For initial settlements, we need to check the distance rule
                    if not edge.parent.check_nearby(self.convert_e(edge.n), Structure.SETTLEMENT, board.players[player_idx], board.turn_number):
                        continue
                    
                    # For initial placements, evaluate production value
                    reward = self.evaluate_settlement_production(edge, board)
                    
                    if reward > best_reward:
                        best_reward = reward
                        best_action = edge_idx
                
                return best_action
            
            else:
                # Choose road adjacent to our most recently placed settlement
                player = board.players[player_idx]
                if player.settlements:
                    last_settlement = player.settlements[-1]
                    parent_hex = last_settlement.parent
                    edge_idx = last_settlement.n
                    
                    # Find adjacent sides (potential road locations)
                    valid_sides = []
                    for side_idx, side in enumerate(board.all_sides):
                        # Check if side is connected to the last settlement and is empty
                        if (side.parent == parent_hex and 
                            (side.n == edge_idx or side.n == (edge_idx + 5) % 6) and
                            side.parent.pos_is_empty(self.convert_s(side.n), Structure.ROAD) and
                            side.parent.check_nearby(self.convert_s(side.n), Structure.ROAD, player, board.turn_number)):
                            valid_sides.append(side_idx)
                    
                    if valid_sides:
                        # Choose the first valid side (they should all be equivalent in initial placement)
                        return valid_sides[0]
                
                # Fallback: choose first available legal road location
                for side_idx in range(30):
                    side = board.all_sides[side_idx]
                    if (side.value is None and 
                        side.parent.check_nearby(self.convert_s(side.n), Structure.ROAD, player, board.turn_number)):
                        return side_idx
        
        # Handle ongoing trade sequences
        if obs_dict["b_trade_followup"]:
            # Bank trade followup - create the most beneficial bank trade
            _, trade = self.evaluate_bank_trade(obs_dict, player_idx)
            if trade is not None:
                # Verify the bank trade is valid according to game rules
                my_inventory = obs_dict["inventories"][player_idx]
                my_items = trade[0]
                b_items = trade[1]
                
                # 1. Ensure exactly one resource type is being offered
                offered_types = [i for i, x in enumerate(my_items) if x > 0]
                if len(offered_types) != 1:
                    self.bank_trade_failures += 1
                    return -1
                
                # 2. Ensure exactly one resource type is being requested
                requested_types = [i for i, x in enumerate(b_items) if x > 0]
                if len(requested_types) != 1:
                    self.bank_trade_failures += 1
                    return -1
                
                # 3. Ensure offered and requested resource types are different
                if offered_types[0] == requested_types[0]:
                    self.bank_trade_failures += 1
                    return -1
                
                # 4. Enforce the 2:1 ratio
                offered_amount = sum(my_items)
                requested_amount = sum(b_items)
                if offered_amount != 2 * requested_amount:
                    self.bank_trade_failures += 1
                    return -1
                
                # 5. Check if we have enough resources
                can_afford = True
                for i in range(4):
                    if my_items[i] > my_inventory[i]:
                        can_afford = False
                        break
                        
                if can_afford:
                    # Reset failure counter on successful trade attempt
                    self.bank_trade_failures = 0
                    self.consecutive_end_turns = 0  # Reset end turn counter on successful trade
                    return trade
            
            # Increment failure counter and check if we should give up
            self.bank_trade_failures += 1
            if self.bank_trade_failures >= self.MAX_FAILURES:
                print(f"Agent {player_idx} giving up on bank trade after {self.bank_trade_failures} failed attempts")
                self.bank_trade_failures = 0  # Reset counter
            
            # If no valid bank trade is possible or too many failures, cancel
            return -1
        
        if obs_dict["p_trade_followup_1"]:
            if obs_dict["reply_to_offer"]:
                # We need to respond to a trade offer
                response = self.respond_to_trade_offer(obs_dict, board, player_idx)
                if response == 1:  # If rejecting
                    # Remember the rejected trade type
                    offer_resource_type = None
                    request_resource_type = None
                    
                    if hasattr(self, 'current_offer') and self.current_offer is not None:
                        # Find the resource types in the offer
                        for i in range(4):
                            if self.current_offer[0][i] > 0:
                                offer_resource_type = i
                            if self.current_offer[1][i] > 0:
                                request_resource_type = i
                                
                        if offer_resource_type is not None and request_resource_type is not None:
                            trade_signature = f"player_o{offer_resource_type}_r{request_resource_type}"
                            if trade_signature not in self.rejected_trades:
                                self.rejected_trades.append(trade_signature)
                                # Keep the list at a reasonable size
                                if len(self.rejected_trades) > self.max_rejected_trades:
                                    self.rejected_trades.pop(0)
                
                return response
            elif obs_dict["counter_sent"]:
                # We need to make a counter offer
                counter_offer = self.create_counter_offer(obs_dict, board, player_idx)
                
                # Verify the counter offer is valid (we can afford it)
                if type(counter_offer) != int:
                    if counter_offer is not None and np.sum(counter_offer[0]) > 0:
                        my_inventory = obs_dict["inventories"][player_idx]
                        can_afford = True
                        for i in range(4):
                            if counter_offer[0][i] > my_inventory[i]:
                                can_afford = False
                                break
                        
                        if can_afford:
                            # Reset failure counter on successful counter offer
                            self.trade_failures = 0
                            return counter_offer
                
                # Increment failure counter
                self.trade_failures += 1
                if self.trade_failures >= self.MAX_FAILURES:
                    print(f"Agent {player_idx} giving up on counter offer after {self.trade_failures} failed attempts")
                    self.trade_failures = 0  # Reset counter
                
                # If we can't make a valid counter offer, cancel the trade
                return -1
            else:
                # We need to initiate a trade offer
                _, trade = self.evaluate_player_trade(obs_dict, player_idx)
                if trade is not None and np.sum(trade[0]) > 0 and np.sum(trade[1]) > 0:
                    # Check if we can afford the trade
                    my_inventory = obs_dict["inventories"][player_idx]
                    can_afford = True
                    for i in range(4):
                        if trade[0][i] > my_inventory[i]:
                            can_afford = False
                            break
                    
                    if can_afford:
                        # Reset failure counter on successful trade attempt
                        self.trade_failures = 0
                        self.consecutive_end_turns = 0  # Reset end turn counter on successful trade
                        return trade
                
                # Increment failure counter
                self.trade_failures += 1
                if self.trade_failures >= self.MAX_FAILURES:
                    print(f"Agent {player_idx} giving up on player trade after {self.trade_failures} failed attempts")
                    self.trade_failures = 0  # Reset counter
                
                # If no valid trade is possible, cancel
                return -1
        
        if obs_dict["p_trade_followup_2"]:
            # We need to respond to a counter offer
            if hasattr(self, 'current_offer') and self.current_offer is not None:
                offered = self.current_offer[1]  # What we would receive
                requested = self.current_offer[0]  # What we would give
                
                # Anti-stalemate: When stuck, be more willing to accept trades
                if self.consecutive_end_turns >= self.stuck_threshold:
                    my_inventory = obs_dict["inventories"][player_idx]
                    if self.would_enable_building(my_inventory, requested, offered, player_idx):
                        self.consecutive_end_turns = 0  # Reset end turn counter on trade
                        return 0  # Accept if it helps build
                    elif np.sum(offered) >= np.sum(requested) - 0.5:
                        self.consecutive_end_turns = 0
                        return 0  # Accept even slightly unfavorable trades
                
                # Normal behavior
                if np.sum(offered) >= np.sum(requested):
                    self.consecutive_end_turns = 0  # Reset end turn counter on accept
                    return 0  # Accept
                else:
                    # Remember rejected trade pattern
                    offer_resource_type = None
                    request_resource_type = None
                    for i in range(4):
                        if requested[i] > 0:
                            offer_resource_type = i
                        if offered[i] > 0:
                            request_resource_type = i
                            
                    if offer_resource_type is not None and request_resource_type is not None:
                        trade_signature = f"player_o{offer_resource_type}_r{request_resource_type}"
                        if trade_signature not in self.rejected_trades:
                            self.rejected_trades.append(trade_signature)
                            # Keep the list at a reasonable size
                            if len(self.rejected_trades) > self.max_rejected_trades:
                                self.rejected_trades.pop(0)
                    
                    return 1  # Reject
                    
            return 1  # Default reject
        
        if obs_dict["p_trade_followup_3"]:
            if obs_dict["reply_to_offer"]:
                # We need to respond to a counter-counter offer
                # Increment our counter to track negotiation length
                self.counter_offer_count += 1
                
                # For simplicity, just accept if it's not terrible
                if hasattr(self, 'current_offer') and self.current_offer is not None:
                    offered = self.current_offer[1]
                    requested = self.current_offer[0]
                    
                    # If we've been counter-offering too many times, force a decision
                    if self.counter_offer_count >= self.MAX_COUNTER_OFFERS:
                        self.counter_offer_count = 0  # Reset counter
                        
                        # Only accept if clearly beneficial after long negotiation
                        if np.sum(offered) > np.sum(requested) + 0.5:
                            self.consecutive_end_turns = 0  # Reset end turn counter
                            return 0  # Accept
                        else:
                            # Remember rejected trade pattern
                            offer_resource_type = None
                            request_resource_type = None
                            for i in range(4):
                                if requested[i] > 0:
                                    offer_resource_type = i
                                if offered[i] > 0:
                                    request_resource_type = i
                                    
                            if offer_resource_type is not None and request_resource_type is not None:
                                trade_signature = f"player_o{offer_resource_type}_r{request_resource_type}"
                                if trade_signature not in self.rejected_trades:
                                    self.rejected_trades.append(trade_signature)
                                    # Keep the list at a reasonable size
                                    if len(self.rejected_trades) > self.max_rejected_trades:
                                        self.rejected_trades.pop(0)
                            
                            self.trade_failures += 1
                            return 1  # Reject
                    
                    # Anti-stalemate: When stuck, be more willing to accept trades
                    if self.consecutive_end_turns >= self.stuck_threshold:
                        my_inventory = obs_dict["inventories"][player_idx]
                        if self.would_enable_building(my_inventory, requested, offered, player_idx):
                            self.consecutive_end_turns = 0
                            return 0  # Accept if it helps build
                        elif np.sum(offered) >= np.sum(requested) - 0.5:
                            self.consecutive_end_turns = 0
                            return 0  # Accept even slightly unfavorable trades
                    
                    # Normal evaluation
                    if np.sum(offered) >= np.sum(requested):
                        # Reset counters on successful trade
                        self.counter_offer_count = 0
                        self.trade_failures = 0
                        self.consecutive_end_turns = 0
                        return 0  # Accept
                    else:
                        # Consider the length of negotiations in decision
                        # The more back-and-forth, the more likely to just end it
                        reject_probability = 0.3 + (0.2 * self.counter_offer_count)
                        if np.random.random() < reject_probability:
                            # Remember rejected trade pattern
                            offer_resource_type = None
                            request_resource_type = None
                            for i in range(4):
                                if requested[i] > 0:
                                    offer_resource_type = i
                                if offered[i] > 0:
                                    request_resource_type = i
                                    
                            if offer_resource_type is not None and request_resource_type is not None:
                                trade_signature = f"player_o{offer_resource_type}_r{request_resource_type}"
                                if trade_signature not in self.rejected_trades:
                                    self.rejected_trades.append(trade_signature)
                            
                            # Increment failure counter on rejection
                            self.trade_failures += 1
                            if self.trade_failures >= self.MAX_FAILURES:
                                print(f"Agent {player_idx} giving up on trade negotiations after {self.trade_failures} failed attempts")
                                self.trade_failures = 0
                                self.counter_offer_count = 0
                            return 1  # Reject
                        else:
                            return 1  # Still reject but don't remember it
                
                # Increment failure counter on default reject
                self.trade_failures += 1
                self.counter_offer_count = 0  # Reset counter
                if self.trade_failures >= self.MAX_FAILURES:
                    print(f"Agent {player_idx} giving up on trade negotiations after {self.trade_failures} failed attempts")
                    self.trade_failures = 0
                return 1  # Default reject
            else:
                # We need to make a counter-counter offer
                counter_offer = self.create_counter_offer(obs_dict, board, player_idx)
                
                # Increment our counter
                self.counter_offer_count += 1
                
                # If we've been counter-offering too many times, just cancel instead of continuing
                if self.counter_offer_count >= self.MAX_COUNTER_OFFERS:
                    self.counter_offer_count = 0  # Reset counter
                    self.trade_failures += 1
                    
                    if self.trade_failures >= self.MAX_FAILURES:
                        print(f"Agent {player_idx} giving up on counter-offers after max attempts")
                        self.trade_failures = 0
                    
                    return -1  # Cancel at this point
                
                # Verify the counter offer is valid (we can afford it)
                if counter_offer is not None and np.sum(counter_offer[0]) > 0:
                    my_inventory = obs_dict["inventories"][player_idx]
                    can_afford = True
                    for i in range(4):
                        if counter_offer[0][i] > my_inventory[i]:
                            can_afford = False
                            break
                    
                    if can_afford:
                        return counter_offer
                
                # Increment failure counter
                self.trade_failures += 1
                if self.trade_failures >= self.MAX_FAILURES:
                    print(f"Agent {player_idx} giving up on counter-counter offer after {self.trade_failures} failed attempts")
                    self.trade_failures = 0  # Reset counter
                    self.counter_offer_count = 0  # Reset counter
                
                # If we can't make a valid counter offer, cancel the trade
                return -1
        
        # If no ongoing trades, evaluate all possible actions
        actions = {
            0: -np.inf,  # Build Road
            1: -np.inf,  # Build Settlement
            2: -np.inf,  # Trade with Player
            3: -np.inf,  # Trade with Bank
            4: END_TURN_REWARD - (0.01 * self.consecutive_end_turns)  # Diminishing reward for ending turn
        }
        
        # Evaluate building roads
        for side_idx in range(30):
            reward = self.evaluate_road(side_idx, board, player_idx)
            actions[0] = max(actions[0], reward)
        
        # Evaluate building settlements
        for edge_idx in range(24):
            reward = self.evaluate_settlement(edge_idx, board, player_idx)
            actions[1] = max(actions[1], reward)
        
        # Evaluate trading with player
        trade_reward, trade_action = self.evaluate_player_trade(obs_dict, player_idx)
        
        # Only consider player trade as viable if players have resources to trade
        # and it returns a valid trade action
        if trade_action is not None and np.sum(trade_action[0]) > 0 and np.sum(trade_action[1]) > 0:
            my_inventory = obs_dict["inventories"][player_idx]
            opponent_inventory = obs_dict["inventories"][1 - player_idx]
            
            # Double-check that we can afford the trade (should already be verified in evaluate_player_trade)
            can_afford = True
            for i in range(4):
                if trade_action[0][i] > my_inventory[i]:
                    can_afford = False
                    break
                    
            # Double-check that opponent can afford the trade
            opponent_can_afford = True
            for i in range(4):
                if trade_action[1][i] > opponent_inventory[i]:
                    opponent_can_afford = False
                    break
                    
            # Only consider trading if both players can afford it
            if can_afford and opponent_can_afford:
                # Penalty for repeated trading attempts
                if self.trade_failures > 0:
                    trade_reward -= 0.3 * self.trade_failures
                
                actions[2] = trade_reward
            else:
                actions[2] = -np.inf
        else:
            actions[2] = -np.inf
        
        # Evaluate trading with bank
        bank_reward, bank_trade = self.evaluate_bank_trade(obs_dict, player_idx)
        
        # Only consider bank trade as viable if it satisfies all requirements
        if bank_trade is not None:
            my_inventory = obs_dict["inventories"][player_idx]
            my_items = bank_trade[0]
            b_items = bank_trade[1]
            
            # 1. Check if exactly one resource type is being offered
            offered_types = [i for i, x in enumerate(my_items) if x > 0]
            valid_offer = len(offered_types) == 1
            
            # 2. Check if exactly one resource type is being requested
            requested_types = [i for i, x in enumerate(b_items) if x > 0]
            valid_request = len(requested_types) == 1
            
            # 3. Check if offered and requested types are different
            different_types = valid_offer and valid_request and offered_types[0] != requested_types[0]
            
            # 4. Check if the 2:1 ratio is maintained
            offered_amount = sum(my_items)
            requested_amount = sum(b_items)
            valid_ratio = offered_amount == 2 * requested_amount
            
            # 5. Double-check that we can afford the trade
            can_afford = True
            for i in range(4):
                if my_items[i] > my_inventory[i]:
                    can_afford = False
                    break
                    
            # Only consider the trade if all conditions are met
            if valid_offer and valid_request and different_types and valid_ratio and can_afford:
                # Penalty for repeated bank trade attempts
                if self.bank_trade_failures > 0:
                    bank_reward -= 0.3 * self.bank_trade_failures
                    
                actions[3] = bank_reward
            else:
                actions[3] = -np.inf
        else:
            actions[3] = -np.inf
        
        # Check for follow-up actions from previous decisions
        if game.waiting_for_road_build_followup:
            # Use our existing evaluation method to find the best legal road position
            best_reward = -np.inf
            best_side_idx = None
            
            for side_idx in range(30):
                reward = self.evaluate_road(side_idx, board, player_idx)
                # evaluate_road already checks legality and returns -np.inf for illegal placements
                if reward > best_reward:
                    best_reward = reward
                    best_side_idx = side_idx
            
            # If we found a legal position, return it
            if best_side_idx is not None and best_reward > -np.inf:
                # Reset failure counter on successful placement
                self.road_placement_failures = 0
                self.consecutive_end_turns = 0  # Reset end turn counter on build
                return best_side_idx
            else:
                # Increment failure counter
                self.road_placement_failures += 1
                if self.road_placement_failures >= self.MAX_FAILURES:
                    print(f"Agent {player_idx} giving up on road placement after {self.road_placement_failures} failed attempts")
                    self.road_placement_failures = 0  # Reset counter
                
                # No legal positions found, cancel the action
                return -1
        
        if game.waiting_for_settlement_build_followup:
            # Use our existing evaluation method to find the best legal settlement position
            best_reward = -np.inf
            best_edge_idx = None
            
            for edge_idx in range(24):
                reward = self.evaluate_settlement(edge_idx, board, player_idx)
                # evaluate_settlement already checks legality and returns -np.inf for illegal placements
                if reward > best_reward:
                    best_reward = reward
                    best_edge_idx = edge_idx
            
            # If we found a legal position, return it
            if best_edge_idx is not None and best_reward > -np.inf:
                # Reset failure counter on successful placement
                self.settlement_placement_failures = 0
                self.consecutive_end_turns = 0  # Reset end turn counter on build
                return best_edge_idx
            else:
                # Increment failure counter
                self.settlement_placement_failures += 1
                if self.settlement_placement_failures >= self.MAX_FAILURES:
                    print(f"Agent {player_idx} giving up on settlement placement after {self.settlement_placement_failures} failed attempts")
                    self.settlement_placement_failures = 0  # Reset counter
                
                # No legal positions found, cancel the action
                return -1
        
        # Choose the action with the highest expected reward
        best_action = max(actions, key=actions.get)
        
        # If selected to build road or settlement, we need a followup to choose where
        if best_action == 0:  # Build Road
            # Choose the best road location
            best_reward = -np.inf
            best_side_idx = 0
            
            for side_idx in range(30):
                reward = self.evaluate_road(side_idx, board, player_idx)
                if reward > best_reward:
                    best_reward = reward
                    best_side_idx = side_idx
            
            # Save the selected location for the followup step
            self.selected_road_idx = best_side_idx
            self.consecutive_end_turns = 0  # Reset end turn counter on build attempt
            
        elif best_action == 1:  # Build Settlement
            # Choose the best settlement location
            best_reward = -np.inf
            best_edge_idx = 0
            
            for edge_idx in range(24):
                reward = self.evaluate_settlement(edge_idx, board, player_idx)
                if reward > best_reward:
                    best_reward = reward
                    best_edge_idx = edge_idx
            
            # Save the selected location for the followup step
            self.selected_settlement_idx = best_edge_idx
            self.consecutive_end_turns = 0  # Reset end turn counter on build attempt
            
        elif best_action == 2:  # Trade with Player
            _, trade = self.evaluate_player_trade(obs_dict, player_idx)
            # Save the offer for later reference
            self.current_offer = trade
            self.consecutive_end_turns = 0  # Reset end turn counter on trade attempt
            
        elif best_action == 3:  # Trade with Bank
            _, trade = self.evaluate_bank_trade(obs_dict, player_idx)
            # Save the offer for later reference
            self.current_bank_offer = trade
            self.consecutive_end_turns = 0  # Reset end turn counter on trade attempt
        
        elif best_action == 4:  # End Turn
            # Increment the consecutive end turns counter
            self.consecutive_end_turns += 1
            
            # Debug info
            if self.consecutive_end_turns % 5 == 0:
                print(f"Agent {player_idx} has ended turn {self.consecutive_end_turns} times in a row")
                
            # Anti-stalemate: If we've been ending turns for too long, forget all rejected trades
            if self.consecutive_end_turns >= self.stuck_threshold * 2:
                self.rejected_trades = []
                print(f"Agent {player_idx} forgetting all rejected trades after {self.consecutive_end_turns} end turns")
        
        return best_action

# Constants needed for the agent
END_TURN_REWARD = 0.05