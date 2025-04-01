import gymnasium as gym
from gymnasium import spaces
import numpy as np
from mini_catan.Board import Board
import random
from mini_catan.enums import Biome, Structure, HexCompEnum
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.collections import PatchCollection

import logging
logging.basicConfig(level=logging.INFO, filename="games.log",filemode="a", format="[%(levelname)s | %(asctime)s | %(lineno)d] %(message)s")
def print(*args, **kwargs):
    logging.info(*args)

"""# Reward variables
S_max = 5
R_max = 10
n_type = 4

alpha = 0.5
U = lambda x: -np.exp(-alpha * x)
R_avg = lambda R_r: np.sum(R_r) / n_type

# Reward functions
ROAD_REWARD = lambda S_2R, S_p: 1 + 1.15 * ( S_2R / max(S_p, 1))
SETTLEMENT_REWARD = lambda S_p: 2 + 1.5 * (S_p / S_max)
TRADE_BANK_REWARD = lambda d_r: 1 + U(0.25 + np.sum(d_r))
TRADE_PLAYER_REWARD = lambda d_r: 1 + U(0.5 + np.sum(d_r))
REJECTED_TRADE_REWARD = lambda T_R: U(1.5 + np.power(T_R,2))
COUTNER_OFFER_REJECTED_REWARD = lambda T_R: U(2 + np.power(T_R,3))
COUTNER_OFFER_ACCEPTED_REWARD = lambda d_r: 1 + U(2.5 + np.sum(d_r))
INVENTORY_BALANCE_REWARD = lambda T_n, R_r: U(T_n + 1 + np.sum(np.abs(R_r - R_avg(R_r))))
TURN_PUNISHMENT = lambda t: 0.2/(1 + np.exp(-0.04*(t-200)))
LONGEST_ROAD_REWARD = 4
END_TURN_REWARD = 0.05
WIN_REWARD = 20
LOSE_REWARD = -20
STALEMATE_REWARD = -8
INITIATE_ROAD_REWARD = 0.8
INITIATE_SETTLEMENT_REWARD = 1
INITIATE_TRADE_BANK_REWARD = 0.3
INITIATE_TRADE_PLAYER_REWARD = 0.5
"""

# Parameters for board-position reward
centrality_weight = 0.6  # Weight for centrality bonus
resource_diversity_weight = 0.8  # Weight for resource diversity bonus
probability_weight = 0.7  # Weight for probability bonus

# Settlement placement reward based on board position
def SETTLEMENT_POSITION_REWARD(edge_id, biomes, hex_nums):
    """
    Calculate reward for settlement placement based on position on the board
    for a smaller Catan map with 24 vertices and 7 hexes.
    
    Args:
        edge_id: The edge ID where settlement is placed (0-23)
        biomes: Array of biome types for each hex
        hex_nums: Array of dice numbers for each hex
    
    Returns:
        Reward value based on position factors
    """
    # Define mapping from edge to adjacent hexes based on the smaller map
    edges_to_hexes = {
        0: [0], 1: [0, 2], 2: [0, 2, 3], 3: [0, 1, 3], 
        4: [0, 1], 5: [0], 6: [1, 3, 4], 7: [1, 4],
        8: [1], 9: [1], 10: [2], 11: [2], 
        12: [2, 5], 13: [2, 3, 5], 14: [3, 5, 6], 15: [3, 4, 6],
        16: [4, 6], 17: [4], 18: [4], 19: [5],
        20: [5], 21: [5, 6], 22: [6], 23: [6]
    }
    
    # Centrality rating - vertices closer to center get higher values
    # For a smaller map, the most central vertices are those around the middle hex
    edge_centrality = {
        # Most central vertices (around middle hex - H4)
        2: 1.0, 3: 1.0, 6: 1.0, 15: 1.0, 14: 1.0, 13: 1.0,
        # Secondary central vertices
        4: 0.8, 1: 0.8, 12: 0.8, 21: 0.8, 16: 0.8, 7: 0.8,
        # Outer vertices
        0: 0.5, 5: 0.5, 10: 0.5, 11: 0.5, 19: 0.5, 20: 0.5, 22: 0.5, 23: 0.5, 17: 0.5,
        18: 0.5, 8: 0.5, 9: 0.5
    }
    centrality = edge_centrality.get(edge_id, 0.5)
    
    # Resource diversity - vertices touching different resource types get bonus
    adjacent_hexes = edges_to_hexes.get(edge_id, [])
    biome_types = set()
    
    for hex_id in adjacent_hexes:
        if hex_id < len(biomes):
            biome_type = biomes[hex_id]
            # Skip desert (biome type 0)
            if biome_type != 0:
                biome_types.add(biome_type)
    
    # More unique resources = higher reward (max 3 different resources)
    diversity = len(biome_types) / 3  # Normalize to 0-1 range
    
    # Probability value - sum of probability values of adjacent hexes
    # With dice 1-5 (and 6 for desert/robber), probabilities are different
    probability_sum = 0
    for hex_id in adjacent_hexes:
        if hex_id < len(hex_nums):
            dice_num = hex_nums[hex_id]
            # Skip desert (marked as 6) or invalid numbers
            if dice_num == 6:
                continue
            # With 1-5 dice, central numbers have highest probability
            elif dice_num == 3:  # Highest probability
                probability_sum += 1.0
            elif dice_num in [2, 4]:  # Medium-high probability
                probability_sum += 0.8
            elif dice_num in [1, 5]:  # Lower probability
                probability_sum += 0.5
    
    # Normalize probability (max would be 3 adjacent hexes with highest probability)
    max_probability = 3.0
    probability_value = min(probability_sum / max_probability, 1.0)
    
    # Add robber penalty - reduce settlement value if adjacent to a desert hex
    # (which is where the robber starts)
    robber_penalty = 0
    for hex_id in adjacent_hexes:
        if hex_id < len(hex_nums) and hex_nums[hex_id] == 6:
            robber_penalty = 0.2  # Penalty for being adjacent to initial robber location
            break
    
    # Calculate combined reward with robber penalty
    position_reward = (
        centrality_weight * centrality +
        resource_diversity_weight * diversity +
        probability_weight * probability_value
    ) - robber_penalty
    
    # Scale to make it comparable to other rewards
    return 2.5 * position_reward

S_max = 5           # Maximum settlements
R_max = 10          # Maximum roads
n_type = 4          # Number of resource types
alpha = 0.5         # Risk aversion parameter
beta = 0.7          # New expansion incentive parameter
gamma = 0.6
R_avg = lambda R_r: np.sum(R_r) / n_type  # Average resources

# Core gameplay rewards
ROAD_REWARD = lambda S_2R, S_p: 0.5 * (1 + np.tanh(S_2R / (S_p + 1)))
SETTLEMENT_REWARD = lambda S_p: 3 * (1 - np.exp(-0.4 * S_p))
TRADE_BANK_REWARD = lambda d_r: 0.8 * (1 + np.tanh(np.sum(d_r) - 0.5))
TRADE_PLAYER_REWARD = lambda d_r: 1.2 * (1 + np.tanh(np.sum(d_r) - 0.3))

# Trade negotiation rewards
REJECTED_TRADE_REWARD = lambda T_R: -0.5 * np.power(T_R, 1.5)
COUTNER_OFFER_REJECTED_REWARD = lambda T_R: -0.3 * np.power(T_R, 1.2)
COUTNER_OFFER_ACCEPTED_REWARD = lambda d_r: 2 * np.tanh(np.sum(d_r))

# Strategic rewards
RESOURCE_DIVERSITY_REWARD = lambda R_r: gamma * (1 - np.std(R_r) / (R_avg(R_r) + 0.1))
EXPANSION_POTENTIAL_REWARD = lambda S_p, R_p, S_a: beta * (S_p/S_max + R_p/R_max) * S_a
INVENTORY_BALANCE_REWARD = lambda T_n, R_r: 0.8 * np.exp(-0.4 * np.sum(np.abs(R_r - R_avg(R_r))))

# Time-based rewards
TURN_PUNISHMENT = lambda t: 0.2/(1 + np.exp(-0.04*(t-200)))
END_TURN_REWARD = lambda t: 0.1 * np.exp(-0.01 * t)  # Decreases over time to encourage faster play

# Achievement rewards
def safe_longest_road_reward(L_all):
    """Calculate longest road reward safely without producing complex numbers."""
    if len(L_all) == 0:
        return 3.0  # Default value for empty array
    
    max_val = np.max(L_all)
    # Extract values that aren't equal to the max
    non_max_vals = L_all[L_all != max_val]
    
    # If there are no non-max values (all values are the same)
    # or if there's only one value in the array
    if len(non_max_vals) == 0:
        return 3.0  # Just return the base reward
    
    # Otherwise calculate using mean of non-max values
    mean_non_max = np.mean(non_max_vals)
    return 3.0 + 0.5 * (max_val - mean_non_max)

# Replace the lambda with this safe function
LONGEST_ROAD_REWARD = lambda L_all: safe_longest_road_reward(L_all)
# L_all is longest road history
#LONGEST_ROAD_REWARD = lambda L_all: 3 + 0.5 * (np.max(L_all) - np.mean(L_all[L_all != np.max(L_all)]))

WIN_REWARD = 200 #30
LOSE_REWARD = -100 #-20
STALEMATE_REWARD = -40 #-6

# Action initiation rewards
INITIATE_ROAD_REWARD = lambda R_p: 0.7 * (1 - R_p/R_max)**0.5
INITIATE_SETTLEMENT_REWARD = lambda S_p: 1.1 * (1 - S_p/S_max)**0.6
INITIATE_TRADE_BANK_REWARD = 0.4
INITIATE_TRADE_PLAYER_REWARD = 0.6


# Pixel positions for render function
settlement_positions = [
    (436, 188), (476, 257), (436, 327), (356, 327),
    (316, 257), (356, 188), (316, 396), (236, 396),
    (196, 327), (236, 257), (556, 257), (596, 327),
    (556, 396), (476, 396), (436, 465), (356, 465),
    (316, 535), (236, 535), (196, 465), (596, 465),
    (556, 535), (476, 535), (436, 604), (356, 604)
]

road_positions = [
    (460, 227), (460, 296), (400, 331), (340, 296),
    (340, 227), (400, 192), (340, 365), (280, 400),
    (220, 365), (220, 296), (280, 261), (580, 296),
    (580, 365), (520, 400), (460, 365), (520, 261),
    (460, 435), (400, 469), (340, 435), (340, 504),
    (280, 539), (220, 504), (220, 435), (580, 435),
    (580, 504), (520, 539), (460, 504), (460, 573),
    (400, 608), (340, 573)
]

tile_polygons = [
    [settlement_positions[i] for i in [0,1,2,3,4,5]], 
    [settlement_positions[i] for i in [4,3,6,7,8,9]], 
    [settlement_positions[i] for i in [10,11,12,13,2,1]], 
    [settlement_positions[i] for i in [2,13,14,15,6,3]], 
    [settlement_positions[i] for i in [6,15,16,17,18,7]], 
    [settlement_positions[i] for i in [12,19,20,21,14,13]], 
    [settlement_positions[i] for i in [14,21,22,23,16,15]]
]

# Dummy color maps for demonstration
PLAYER_COLOR_MAP = {
    0: "gray",    # unowned
    1: "cyan",    # player 0
    2: "fuchsia"      # player 1
}

# --- Define color maps for biomes and players ---
BIOME_COLOR_MAP = {
    Biome.DESERT:  "gold",
    Biome.FOREST:  "darkgreen",
    Biome.HILLS:   "red",
    Biome.FIELDS:  "saddlebrown",
    Biome.PASTURE: "limegreen",
}

def biome_num(b):
    match b:
        case Biome.DESERT:
            return 0
        case Biome.FOREST:
            return 1
        case Biome.HILLS:
            return 2
        case Biome.FIELDS:
            return 3
        case Biome.PASTURE:
            return 4
        case _:
            return None

class MiniCatanEnv(gym.Env):
    """Custom Gym Environment for Catan."""
    
    #metadata = {"render_modes": ["human"], "render_fps": 30}
    
    def __init__(self, render_mode=None):
        super().__init__()

        self.render_mode = render_mode # human, bot
        self.num_players = 2  # Modify based on your game setup
        self.max_victory_points = 5

        self.main_player = random.randint(0,1)

        self.board = Board(["P1", "P2"])

        self.prev_longest_road_owner = self.board.get_longest_road_owner()

        self.reset_followup_variables()
        self.trade_initiator = None
        
        # Define action space (e.g., 5 actions: Build Road, Build Settlement, Trade with Player, Trade with Bank, End Turn)
        self.action_space = spaces.Discrete(5)
        
        self.build_settlement_action_space = spaces.Discrete(24)

        self.build_road_action_space = spaces.Discrete(30)

        self.player_trade_action_space = spaces.Box(low=0, high=10, shape=(2, 4), dtype=np.int32)
        self.player_trade_offer_request_action_space =  spaces.Discrete(3) # 0: Yes, 1: No, 2: Counter
        self.counter_offer_action_space = spaces.Box(low=0, high=10, shape=(2, 4), dtype=np.int32)

        self.counter_offer_response_action_space = spaces.Discrete(3) # 0: Yes, 1: No, 2: Counter the Counter

        self.counter_counter_offer_action_space = spaces.Box(low=0, high=10, shape=(2, 4), dtype=np.int32)
        self.counter_counter_offer_reply_action_space =  spaces.Discrete(2) # 0: Yes, 1: No

        self.bank_trade_action_space = spaces.Box(low=0, high=10, shape=(2, 4), dtype=np.int32)
        
        # Define observation space (simplified example: resources, settlements, roads)
        """
        "inventories": 0
        "edges": 1
        "sides": 2
        "longest_road_owner": 3
        "victory_points": 4
        "biomes": 5
        "hex_nums": 6 
        "robber_loc": 7
        "turn_number": 8
        "p_trade_followup_1": 9
        "p_trade_followup_2": 10
        "p_trade_followup_3": 11
        "reply_to_offer": 12
        "counter_sent": 13
        """
        self.obs_space_size = (
            (self.num_players * 4) +      # inventories
            (24) +                        # edges
            (30) +                        # sides
            (1) +                         # longest_road_owner
            (self.num_players) +          # victory_points
            (self.board.board_size) +     # biomes
            (self.board.board_size) +     # hex_nums
            (1) +                         # robber_loc
            (1) +                         # turn_number
            (1) +                         # b_trade_followup
            (1) +                         # p_trade_followup_1
            (1) +                         # p_trade_followup_2
            (1) +                         # p_trade_followup_3
            (1) +                         # reply_to_offer
            (1)                           # counter_sent
        )
        self.observation_space = spaces.Box(low = 0, high = 20, shape=(self.obs_space_size, ), dtype=np.int32)


        # Initial state
        self.state = self.get_initial_state()
        self.current_player = 0
        self.init_phase = 0  # 0=P1 settlement, 1=P1 road, 2=P2 settlement, 3=P2 road, 
                         # 4=P2 settlement2, 5=P2 road2, 6=P1 settlement2, 7=P1 road2

    def resouce_collection_round(self):
        self.dice_val = self.board.roll_dice()
        for p in self.board.players:
            self.board.give_resources(p, self.dice_val)
    
    def get_initial_state(self):
        """Define the initial state of the game."""
        self.board.make_board()
                
        state = {
            "inventories": np.zeros((self.num_players, 4), dtype=np.int32),
            "edges": np.zeros((24,), dtype=np.int32),
            "sides": np.zeros((30,), dtype=np.int32),
            "longest_road_owner": 0,
            "victory_points": np.zeros((self.num_players, ), dtype=np.int32),
            "biomes" : np.array([biome_num(b) for b in self.board.get_hex_biomes()]),
            "hex_nums" : np.array(self.board.get_hex_nums()),
            "robber_loc" : self.board.robber_loc,
            "turn_number" : self.board.turn_number,
            "b_trade_followup" : self.waiting_for_b_trade_followup,
            "p_trade_followup_1": self.waiting_for_p_trade_followup_1,
            "p_trade_followup_2": self.waiting_for_p_trade_followup_2,
            "p_trade_followup_3": self.waiting_for_p_trade_followup_3,
            "reply_to_offer": self.reply_to_offer,
            "counter_sent": self.counter_sent
        }
        return self.encode_observation(state)
    
    def encode_observation(self, state):
        obs = np.concatenate((
            state["inventories"].flatten(),  # Flatten inventories
            state["edges"].flatten(),  # Flatten edges
            state["sides"].flatten(),  # Flatten sides
            np.array([state["longest_road_owner"]]),  # Longest road owner
            state["victory_points"].flatten(),  # Victory points
            state["biomes"].flatten(),  # Biomes
            state["hex_nums"].flatten(),  # Hex numbers
            np.array([state["robber_loc"]]),  # Robber location
            np.array([state["turn_number"]]),  # Turn Number
            np.array([state["b_trade_followup"]]), # Bank trade
            np.array([state["p_trade_followup_1"]]),
            np.array([state["p_trade_followup_2"]]),
            np.array([state["p_trade_followup_3"]]),
            np.array([state["reply_to_offer"]]),
            np.array([state["counter_sent"]])
        ))
        return obs
    
    def reset_followup_variables(self):

        self.init_settlement_build = True
        self.init_road_build = False
        self.init_build = 0
        self.waiting_for_settlement_build_followup = False
        self.waiting_for_road_build_followup = False
        self.waiting_for_b_trade_followup = False
        self.waiting_for_p_trade_followup_1 = False
        self.offer = None
        self.reply_to_offer = False
        self.counter_sent = False
        self.waiting_for_p_trade_followup_2 = False
        self.waiting_for_p_trade_followup_3 = False
        self.dice_val = 0
    
    def reset(self, seed=None, options=None):
        """Reset the environment to the initial state."""
        super().reset(seed=seed)
        self.board = Board(["P1", "P2"])
        self.state = self.get_initial_state()
        self.current_player = 0

        self.reset_followup_variables()

        return self.state, {}
    
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
            
    def decode_observation(self, obs):
        idx = 0
        inventories = obs[idx:(idx + self.num_players * 4)].reshape(self.num_players, 4)
        idx += self.num_players * 4
        
        edges = obs[idx:(idx + 24)].reshape(24, )
        idx += 24
        
        sides = obs[idx:(idx + 30)].reshape(30, )
        idx += 30
        
        longest_road_owner = obs[idx]
        idx += 1
        
        victory_points = obs[idx:(idx + self.num_players)].reshape(self.num_players, )
        idx += self.num_players
        
        biomes = obs[idx:(idx + self.board.board_size)]
        idx += self.board.board_size
        
        hex_nums = obs[idx:(idx + self.board.board_size)]
        idx += self.board.board_size
        
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

    def step(self, action):
        """Perform an action in the environment."""

        # Placeholder logic for handling actions
        reward = 0
        done = False
        trunc = False
        info = {}
        curr_player = self.board.players[self.current_player]

        if self.board.turn_number > 0:
            if type(action) == int and action < 0:
                print("Cancelling action and returning to action selection")
                self.reset_followup_variables()
                # If a trade dialogue was in progress, revert to the trade initiator.
                if self.trade_initiator is not None:
                    self.current_player = self.trade_initiator
                    self.trade_initiator = None
                reward -= END_TURN_REWARD(self.board.turn_number) / 2
                obs = self.decode_observation(self.state)
                obs["edges"] = np.array(self.board.get_edges())
                obs["sides"] = np.array(self.board.get_sides())
                obs["victory_points"] = np.array(self.board.get_vp())
                obs["inventories"] = np.array(self.board.get_all_invs())
                obs["longest_road_owner"] = self.board.get_longest_road_owner()
                obs["robber_loc"] = self.board.robber_loc
                obs["turn_number"] = self.board.turn_number
                obs["b_trade_followup"] = self.waiting_for_b_trade_followup
                obs["p_trade_followup_1"] = self.waiting_for_p_trade_followup_1
                obs["p_trade_followup_2"] = self.waiting_for_p_trade_followup_2
                obs["p_trade_followup_3"] = self.waiting_for_p_trade_followup_3
                obs["reply_to_offer"] = self.reply_to_offer
                obs["counter_sent"] = self.counter_sent
                return self.encode_observation(obs), reward, done, trunc, info

            elif self.waiting_for_road_build_followup:
                assert self.build_road_action_space.contains(action), "Invalid Road Position"

                for i,s in enumerate(self.board.all_sides):
                    if i == action:
                        placement = self.board.place_struct(curr_player, s.parent, self.convert_s(s.n), Structure.ROAD)
                        if placement == -1: raise AssertionError("Cannot Place Structure Here")
                        elif placement == -2: raise AssertionError("Cannot Afford Structure")
                        elif placement == -3: raise AssertionError("Reached Max Structure Limit")
                        
                        print(f"Player {self.current_player + 1} built a road at position {action}")
                        reward += ROAD_REWARD(curr_player.get_player_s2r(), len(curr_player.settlements))
                        self.waiting_for_road_build_followup = False

            elif self.waiting_for_settlement_build_followup:
                assert self.build_settlement_action_space.contains(action), "Invalid Settlement Position"

                for i,e in enumerate(self.board.all_edges):
                    if i == action:
                        placement = self.board.place_struct(curr_player, e.parent, self.convert_e(e.n), Structure.SETTLEMENT)
                        if placement == -1: raise AssertionError("Cannot Place Structure Here")
                        elif placement == -2: raise AssertionError("Cannot Afford Structure")
                        elif placement == -3: raise AssertionError("Reached Max Structure Limit")
                        
                        print(f"Player {self.current_player + 1} built a settlement at position {action}")
                        reward += SETTLEMENT_REWARD(len(curr_player.settlements)) + SETTLEMENT_POSITION_REWARD(action, [biome_num(b) for b in self.board.get_hex_biomes()], self.board.get_hex_nums())
                        self.waiting_for_settlement_build_followup = False

            elif self.waiting_for_b_trade_followup:
                assert self.bank_trade_action_space.contains(action), "Invalid Bank Trade Action"
                assert any(a>0 for a in action[0]), "Cannot Offer Nothing"

                trade = curr_player.trade_I_with_b(action[0], action[1])
                assert trade, "Cannot Afford Trade"
                print(f"Player {self.current_player + 1} traded with bank: offered {action[0]} to get {action[1]}")
                reward += TRADE_BANK_REWARD(action[1] - (action[0] * 2)) #Requested - Offering
                self.waiting_for_b_trade_followup = False

            elif self.waiting_for_p_trade_followup_1:

                if self.reply_to_offer:
                    assert self.player_trade_offer_request_action_space.contains(action), "Invalid Reply to Offer"
                    #offer is saved in self.offer
                    if action == 0:
                        self.current_player = (self.current_player + 1) % self.num_players
                        self.waiting_for_p_trade_followup_1 = False
                        trade = self.board.players[self.current_player].trade_I_with_p(curr_player, self.offer[0], self.offer[1])
                        assert trade, "Cannot Afford Trade"
                        print(f"Player {self.current_player + 1} accepted trade offer from Player {(self.current_player + 1) % self.num_players + 1}")
                        
                    elif action == 1:
                        self.current_player = (self.current_player + 1) % self.num_players
                        self.waiting_for_p_trade_followup_1 = False
                        print(f"Player {(self.current_player % self.num_players) + 1} rejected trade offer from Player {self.current_player + 1}")
                        reward += REJECTED_TRADE_REWARD(self.board.players[self.current_player].trades_rejected) # trade rejected
                        
                    elif action == 2:
                        self.counter_sent = True
                        print(f"Player {(self.current_player % self.num_players) + 1} is making a counter offer to Player {self.current_player + 1}'s trade")
                    
                    self.reply_to_offer = False
                    
                elif self.counter_sent:
                    assert self.counter_offer_action_space.contains(action), "Invalid Reply to Offer"
                    self.offer = action
                    trade = curr_player.trade_cost_check(self.board.players[(self.current_player + 1) % self.num_players], action[0], action[1])
                    assert trade, "Cannot Afford Trade"

                    print(f"Player {self.current_player + 1} made counter offer: offering {action[0]} to get {action[1]}")
                    self.counter_sent = False
                    self.current_player = (self.current_player + 1) % self.num_players
                    self.waiting_for_p_trade_followup_2 = True
                    self.waiting_for_p_trade_followup_1 = False
                else:
                    #send offer
                    assert self.player_trade_action_space.contains(action), "Invalid Player Trade Action"
                    self.offer = action
                    trade = curr_player.trade_cost_check(self.board.players[(self.current_player + 1) % self.num_players], action[0], action[1])
                    assert trade, "Cannot Afford Trade"

                    print(f"Player {self.current_player + 1} offered a trade to Player {(self.current_player + 1) % self.num_players + 1}: offering {action[0]} to get {action[1]}")
                    reward += TRADE_PLAYER_REWARD(action[1] - action[0])
                    self.reply_to_offer = True
                    self.current_player = (self.current_player + 1) % self.num_players

            elif self.waiting_for_p_trade_followup_2:
                assert self.counter_offer_response_action_space.contains(action), "Invalid Counter Offer Response Action"

                if action == 0:
                    trade = self.board.players[(self.current_player + 1) % self.num_players].trade_I_with_p(curr_player, self.offer[0], self.offer[1])
                    assert trade, "Cannot Afford Trade"
                    print(f"Player {self.current_player + 1} accepted the counter offer from Player {(self.current_player + 1) % self.num_players + 1}")
                elif action == 1:
                    #end trade
                    self.board.players[self.current_player].trades_rejected += 1
                    print(f"Player {self.current_player + 1} rejected the counter offer from Player {(self.current_player + 1) % self.num_players + 1}")
                    reward += COUTNER_OFFER_REJECTED_REWARD(self.board.players[self.current_player].trades_rejected)
                elif action == 2:
                    self.waiting_for_p_trade_followup_3 = True
                    print(f"Player {self.current_player + 1} is making a counter-counter offer")
                    reward += TRADE_PLAYER_REWARD(self.offer[1] - self.offer[0]) # countering a bad trade
                self.waiting_for_p_trade_followup_2 = False


            elif self.waiting_for_p_trade_followup_3:
                if self.reply_to_offer:
                    assert self.counter_counter_offer_reply_action_space.contains(action), "Invalid Counter Counter Offer Reply Action"

                    if action == 0:
                        self.current_player = (self.current_player + 1) % self.num_players
                        trade = self.board.players[self.current_player].trade_I_with_p(curr_player, self.offer[0], self.offer[1])
                        assert trade, "Cannot Afford Trade"
                        print(f"Player {(self.current_player % self.num_players) + 1} accepted the counter-counter offer")
                        reward += COUTNER_OFFER_ACCEPTED_REWARD(self.offer[1] - self.offer[0])#trade accepted and ressources gained and negotiated well
                        
                    elif action == 1:
                        self.current_player = (self.current_player + 1) % self.num_players
                        self.board.players[self.current_player].trades_rejected += 1
                        print(f"Player {(self.current_player % self.num_players) + 1} rejected the counter-counter offer")
                        reward += COUTNER_OFFER_REJECTED_REWARD(self.board.players[self.current_player].trades_rejected) # trade rejected twice

                    self.reply_to_offer = False
                    self.waiting_for_p_trade_followup_3 = False

                else:
                    assert self.counter_counter_offer_action_space.contains(action), "Invalid Counter Counter Offer Action"
                
                    self.offer = action
                    trade = curr_player.trade_cost_check(self.board.players[(self.current_player + 1) % self.num_players], action[0], action[1])
                    assert trade, "Cannot Afford Trade"

                    print(f"Player {self.current_player + 1} made a counter-counter offer: offering {action[0]} to get {action[1]}")
                    self.current_player = (self.current_player + 1) % self.num_players
                    self.reply_to_offer = True
                    reward += TRADE_PLAYER_REWARD(action[1] - action[0])
                
            
            else:
                assert self.action_space.contains(action), "Invalid Action"
                
                if action == 0:  # Build Road
                    print("Player attemtps to build a road.")
                    self.waiting_for_road_build_followup = True
                    reward += INITIATE_ROAD_REWARD(len(curr_player.roads))

                elif action == 1:  # Build Settlement
                    print("Player attemtps to build a settlement.")
                    self.waiting_for_settlement_build_followup = True
                    reward += INITIATE_SETTLEMENT_REWARD(len(curr_player.settlements))

                elif action == 2:  # Trade with player
                    print("Player attemtps to initiate a trade with Player.")
                    self.waiting_for_p_trade_followup_1 = True
                    self.trade_initiator = self.current_player
                    reward += INITIATE_TRADE_PLAYER_REWARD

                elif action == 3:  # Trade with Bank
                    print("Player attemtps to initiate a trade with Bank.")
                    self.waiting_for_b_trade_followup = True
                    reward += INITIATE_TRADE_BANK_REWARD

                elif action == 4:  # End Turn
                    S_a = len(self.board.all_edges) - (len(self.board.players[self.current_player].settlements) + len(curr_player.settlements))
                    reward += EXPANSION_POTENTIAL_REWARD(len(curr_player.settlements), len(curr_player.roads), S_a)

                    print("Player Ends Turn.")
                    self.current_player = (self.current_player + 1) % self.num_players
                    reward += END_TURN_REWARD(self.board.turn_number)
                    self.board.turn_number += 1

                    self.resouce_collection_round()
                    print(f"Dice roll: {self.dice_val}. Resources collected.")

                    reward += INVENTORY_BALANCE_REWARD(curr_player.total_trades, np.array(curr_player.inventory)) #inventory balance penalty at end of turn
                    reward += RESOURCE_DIVERSITY_REWARD(np.array(curr_player.inventory))

        else:  # Initial placement phase (turn_number == 0)
            # Implementation of proper snake draft (1-2-2-1)
            if self.init_phase < 8:
                # Determine current action type (settlement or road)
                is_settlement = (self.init_phase % 2 == 0)
                
                # Determine current player based on snake draft pattern
                if self.init_phase < 4:
                    # First round: P1-P2 order
                    current_player_idx = self.init_phase // 2
                else:
                    # Second round: P2-P1 order (reversed)
                    current_player_idx = 1 - ((self.init_phase - 4) // 2)
                    
                self.current_player = current_player_idx % 2
                curr_player = self.board.players[current_player_idx]
                
                if is_settlement:
                    # Settlement building logic
                    assert self.build_settlement_action_space.contains(action), "Invalid Settlement Position"
                    
                    for i,e in enumerate(self.board.all_edges):
                        if i == action:
                            placement = self.board.place_struct(curr_player, e.parent, self.convert_e(e.n), Structure.SETTLEMENT)
                            if placement == -1: raise AssertionError("Cannot Place Structure Here")
                            elif placement == -2: raise AssertionError("Cannot Afford Structure")
                            elif placement == -3: raise AssertionError("Reached Max Structure Limit")
                            
                            print(f"Initial Phase: Player {current_player_idx + 1} built a settlement at position {action} (Phase {self.init_phase + 1}/8)")
                            reward += SETTLEMENT_REWARD(len(curr_player.settlements))
                            
                            # Track which settlement is first/second for resource distribution
                            if self.init_phase < 4:  # First round
                                curr_player.first_settlement = (e.parent, self.convert_e(e.n))
                            else:  # Second round
                                curr_player.second_settlement = (e.parent, self.convert_e(e.n))
                else:
                    # Road building logic
                    assert self.build_road_action_space.contains(action), "Invalid Road Position"
                    
                    for i,s in enumerate(self.board.all_sides):
                        if i == action:
                            placement = self.board.place_struct(curr_player, s.parent, self.convert_s(s.n), Structure.ROAD)
                            if placement == -1: raise AssertionError("Cannot Place Structure Here")
                            elif placement == -2: raise AssertionError("Cannot Afford Structure")
                            elif placement == -3: raise AssertionError("Reached Max Structure Limit")
                            
                            print(f"Initial Phase: Player {current_player_idx + 1} built a road at position {action} (Phase {self.init_phase + 1}/8)")
                            reward += ROAD_REWARD(curr_player.get_player_s2r(), len(curr_player.settlements))
                
                # Move to next step in initial placement
                self.init_phase += 1
                
                # Check if initial placement is complete
                if self.init_phase == 8:
                    # Give resources from SECOND settlement only (standard Catan rules)
                    for p in self.board.players:
                        if hasattr(p, 'second_settlement'):
                            self.board.give_resources(p, 0, p.second_settlement)
                    
                    print("Initial placement phase complete. Starting regular gameplay.")
                    self.board.turn_number += 1
                    self.resouce_collection_round()
                    print(f"First dice roll: {self.dice_val}. Resources collected.")

        obs = self.decode_observation(self.state)

        obs["edges"] = np.array(self.board.get_edges())
        obs["sides"] = np.array(self.board.get_sides())
        obs["victory_points"] = np.array(self.board.get_vp())
        obs["inventories"] = np.array(self.board.get_all_invs())
        obs["longest_road_owner"] = self.board.get_longest_road_owner()
        obs["robber_loc"] = self.board.robber_loc
        obs["turn_number"] = self.board.turn_number
        obs["b_trade_followup"] = self.waiting_for_b_trade_followup
        obs["p_trade_followup_1"] = self.waiting_for_p_trade_followup_1
        obs["p_trade_followup_2"] = self.waiting_for_p_trade_followup_2
        obs["p_trade_followup_3"] = self.waiting_for_p_trade_followup_3
        obs["reply_to_offer"] = self.reply_to_offer
        obs["counter_sent"] = self.counter_sent

        if self.prev_longest_road_owner != self.board.get_longest_road_owner():
            new_owner = self.board.get_longest_road_owner()
            if new_owner > 0:  # 0 means no owner
                print(f"Player {new_owner} now has the longest road!")
            
            if len(curr_player.longest_road_history) > 1:
                reward += LONGEST_ROAD_REWARD(np.array(curr_player.longest_road_history))
            else:
                reward += 3
                
        self.prev_longest_road_owner = self.board.get_longest_road_owner()

        # Check for game end condition
        if any(player >= self.max_victory_points for player in obs["victory_points"]):
            done = True
            if curr_player.vp >= self.max_victory_points:
                print(f"Player {self.current_player + 1} wins the game!")
                reward = WIN_REWARD
            else: 
                print(f"Player {(self.current_player + 1) % self.num_players + 1} wins the game!")
                reward = LOSE_REWARD
            print("============================================ Game Concluded ============================================")

        info = obs

        reward -= TURN_PUNISHMENT(self.board.turn_number)

        # Return the observation, reward, done, and info
        return self.encode_observation(obs), reward, done, trunc, info

    def render(self, t):
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Draw Hexes (tiles)
        hex_biomes = self.board.get_hex_biomes()
        hex_nums = self.board.get_hex_nums()
        robber_loc = self.board.robber_loc
        hex_patches = []
        
        # Build hex patches with a low zorder so they appear in the background.
        for i, poly in enumerate(tile_polygons):
            biome = hex_biomes[i]
            color = BIOME_COLOR_MAP.get(biome, "white")
            poly_patch = patches.Polygon(poly, closed=True, facecolor=color, edgecolor="black", lw=2, zorder=1)
            hex_patches.append(poly_patch)
        
        # Add the hex patches collection to the axis.
        ax.add_collection(PatchCollection(hex_patches, match_original=True, zorder=1))
        
        # Now, add tile numbers and robber marker on top (with higher zorder).
        for i, poly in enumerate(tile_polygons):
            xs, ys = zip(*poly)
            center = (sum(xs)/len(xs), sum(ys)/len(ys))
            ax.text(center[0], center[1], str(hex_nums[i]),
                    ha="center", va="center", fontsize=14, color="white", weight="bold", zorder=3)
            if i == robber_loc:
                robber_patch = patches.RegularPolygon(center, numVertices=4, radius=20,
                                                    orientation=0.785, color="black", zorder=2)
                ax.add_patch(robber_patch)
        
        # Draw Settlements
        edges = self.board.get_edges()
        for idx, pos in enumerate(settlement_positions):
            owner = edges[idx]
            color = PLAYER_COLOR_MAP.get(owner, "gray")
            size = 14
            rect = patches.Rectangle((pos[0]-size/2, pos[1]-size/2), size, size,
                                    facecolor=color, edgecolor="black", zorder=3)
            ax.add_patch(rect)
        
        # Draw Roads
        sides = self.board.get_sides()
        for idx, pos in enumerate(road_positions):
            owner = sides[idx]
            color = PLAYER_COLOR_MAP.get(owner, "gray")
            circle = patches.Circle(pos, radius=8, facecolor=color, edgecolor="black", zorder=3)
            ax.add_patch(circle)
        
        # Draw an Inventory Box on the side
        inventories = self.board.get_all_invs()
        inv_box = patches.Rectangle((650, 200), 250, 300, facecolor="lightgray", edgecolor="black", zorder=3)
        ax.add_patch(inv_box)
        
        # Write each player's inventory inside the box
        y_text = 320
        inv_str = f"Longest Road Owner: {self.board.get_longest_road_owner()-1} \n"
        for player, inv in enumerate(inventories):
            inv_lines = [f"{k}: {v}" for k, v in zip(["Wood", "Brick", "Sheep", "Wheat"], inv)]
            inv_str = inv_str + f"P{player}:\n" + "\n".join(inv_lines)
            ax.text(660, y_text, inv_str, fontsize=10, verticalalignment="top", zorder=4)
            y_text -= 100  # Adjust spacing for each player's info
            inv_str = ""
        
        # Set display limits and hide axes for a cleaner look
        ax.set_xlim(200, 900)
        ax.set_ylim(0, 800)
        ax.set_aspect("equal")
        ax.axis("off")
        ax.invert_yaxis()
        
        plt.show()

        import time
        time.sleep(t)

        plt.close()


    def close(self):
        """Close any resources used by the environment."""
        pass