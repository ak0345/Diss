import gymnasium as gym
from gymnasium import spaces
import numpy as np
from mini_catan.Board import Board
import random
from mini_catan.enums import Biome, Structure, HexCompEnum
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.collections import PatchCollection

# Reward variables
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
LONGEST_ROAD_REWARD = 2
END_TURN_REWARD = 0.05
WIN_REWARD = 10
LOSE_REWARD = -10
INITIATE_ROAD_REWARD = 0.2
INITIATE_SETTLEMENT_REWARD = 0.3
INITIATE_TRADE_BANK_REWARD = 0.1
INITIATE_TRADE_PLAYER_REWARD = 0.15

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

        self._reset_followup_variables()
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
        self.state = self._get_initial_state()
        self.current_player = 0

    def assign_main_player(self, player):
        self.main_player = player

    def _resouce_collection_round(self):
        self.dice_val = self.board.roll_dice()
        for p in self.board.players:
            self.board.give_resources(p, self.dice_val)
    
    def _get_initial_state(self):
        """Define the initial state of the game."""
        #Get and process Hex Biomes
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
        return self._encode_observation(state)
    
    def _encode_observation(self, state):
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
    
    def _reset_followup_variables(self):

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
        self.state = self._get_initial_state()
        self.current_player = 0

        self._reset_followup_variables()

        return self.state, {}
    
    def _convert_s(self, pos):
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
        
    def _convert_e(self, pos):
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
            
    def _decode_observation(self, obs):
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
                self._reset_followup_variables()
                # If a trade dialogue was in progress, revert to the trade initiator.
                if self.trade_initiator is not None:
                    self.current_player = self.trade_initiator
                    self.trade_initiator = None
                reward -= END_TURN_REWARD / 2
                obs = self._decode_observation(self.state)
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
                return self._encode_observation(obs), reward, done, trunc, info

            elif self.waiting_for_road_build_followup:
                assert self.build_road_action_space.contains(action), "Invalid Road Position"

                for i,s in enumerate(self.board.all_sides):
                    if i == action:
                        placement = self.board.place_struct(curr_player, s.parent, self._convert_s(s.n), Structure.ROAD)
                        if placement == -1: raise AssertionError("Cannot Place Structure Here")
                        elif placement == -2: raise AssertionError("Cannot Afford Structure")
                        elif placement == -3: raise AssertionError("Reached Max Structure Limit")
                        
                        reward += ROAD_REWARD(curr_player.get_player_s2r(), len(curr_player.settlements))
                        self.waiting_for_road_build_followup = False

            elif self.waiting_for_settlement_build_followup:
                assert self.build_settlement_action_space.contains(action), "Invalid Settlement Position"

                for i,e in enumerate(self.board.all_edges):
                    if i == action:
                        placement = self.board.place_struct(curr_player, e.parent, self._convert_e(e.n), Structure.SETTLEMENT)
                        if placement == -1: raise AssertionError("Cannot Place Structure Here")
                        elif placement == -2: raise AssertionError("Cannot Afford Structure")
                        elif placement == -3: raise AssertionError("Reached Max Structure Limit")
                        
                        reward += SETTLEMENT_REWARD(len(curr_player.settlements))
                        self.waiting_for_settlement_build_followup = False

            elif self.waiting_for_b_trade_followup:
                assert self.bank_trade_action_space.contains(action), "Invalid Bank Trade Action"
                assert any(a>0 for a in action[0]), "Cannot Offer Nothing"

                trade = curr_player.trade_I_with_p(self.board.bank, action[0] * 2, action[1])
                assert trade, "Cannot Afford Trade"
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
                        
                    elif action == 1:
                        self.current_player = (self.current_player + 1) % self.num_players
                        self.waiting_for_p_trade_followup_1 = False
                        reward += REJECTED_TRADE_REWARD(self.board.players[self.current_player].trades_rejected) # trade rejected
                        
                    elif action == 2:
                        self.counter_sent = True
                    
                    self.reply_to_offer = False
                    
                elif self.counter_sent:
                    assert self.counter_offer_action_space.contains(action), "Invalid Reply to Offer"
                    self.offer = action
                    trade = curr_player.trade_cost_check(self.board.players[(self.current_player + 1) % self.num_players], action[0], action[1])
                    assert trade, "Cannot Afford Trade"

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

                    reward += TRADE_PLAYER_REWARD(action[1] - action[0])
                    self.reply_to_offer = True
                    self.current_player = (self.current_player + 1) % self.num_players

            elif self.waiting_for_p_trade_followup_2:
                assert self.counter_offer_response_action_space.contains(action), "Invalid Counter Offer Response Action"

                if action == 0:
                    trade = self.board.players[(self.current_player + 1) % self.num_players].trade_I_with_p(curr_player, self.offer[0], self.offer[1])
                    assert trade, "Cannot Afford Trade"
                elif action == 1:
                    #end trade
                    self.board.players[self.current_player].trades_rejected += 1
                    reward += COUTNER_OFFER_REJECTED_REWARD(self.board.players[self.current_player].trades_rejected)
                elif action == 2:
                    self.waiting_for_p_trade_followup_3 = True
                    reward += TRADE_PLAYER_REWARD(self.offer[1] - self.offer[0]) # countering a bad trade
                self.waiting_for_p_trade_followup_2 = False


            elif self.waiting_for_p_trade_followup_3:
                if self.reply_to_offer:
                    assert self.counter_counter_offer_reply_action_space.contains(action), "Invalid Counter Counter Offer Reply Action"

                    if action == 0:
                        self.current_player = (self.current_player + 1) % self.num_players
                        trade = self.board.players[self.current_player].trade_I_with_p(curr_player, self.offer[0], self.offer[1])
                        assert trade, "Cannot Afford Trade"
                        reward += COUTNER_OFFER_ACCEPTED_REWARD(self.offer[1] - self.offer[0])#trade accepted and ressources gained and negotiated well
                        
                    elif action == 1:
                        self.current_player = (self.current_player + 1) % self.num_players
                        self.board.players[self.current_player].trades_rejected += 1
                        reward += COUTNER_OFFER_REJECTED_REWARD(self.board.players[self.current_player].trades_rejected) # trade rejected twice

                    self.reply_to_offer = False
                    self.waiting_for_p_trade_followup_3 = False

                else:
                    assert self.counter_counter_offer_action_space.contains(action), "Invalid Counter Counter Offer Action"
                
                    self.offer = action
                    trade = curr_player.trade_cost_check(self.board.players[(self.current_player + 1) % self.num_players], action[0], action[1])
                    assert trade, "Cannot Afford Trade"

                    self.current_player = (self.current_player + 1) % self.num_players
                    self.reply_to_offer = True
                    reward += TRADE_PLAYER_REWARD(action[1] - action[0])
                
            
            else:
                assert self.action_space.contains(action), "Invalid Action"
                
                if action == 0:  # Build Road
                    print("Player attemtps to build a road.")
                    self.waiting_for_road_build_followup = True
                    reward += INITIATE_ROAD_REWARD

                elif action == 1:  # Build Settlement
                    print("Player attemtps to build a settlement.")
                    self.waiting_for_settlement_build_followup = True
                    reward += INITIATE_SETTLEMENT_REWARD

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
                    print("Player Ends Turn.")
                    self.current_player = (self.current_player + 1) % self.num_players
                    self.board.turn_number += 1
                    reward += END_TURN_REWARD

                    self._resouce_collection_round()

                    reward += INVENTORY_BALANCE_REWARD(curr_player.total_trades, np.array(curr_player.inventory)) #inventory balance penalty at end of turn

        else: 
            # Build set 1 and road 1 and then set 2 and road 2

            if self.init_settlement_build and self.init_build < 4:
                assert self.build_settlement_action_space.contains(action), "Invalid Settlement Position"

                for i,e in enumerate(self.board.all_edges):
                    if i == action:
                        placement = self.board.place_struct(curr_player, e.parent, self._convert_e(e.n), Structure.SETTLEMENT)
                        if placement == -1: raise AssertionError("Cannot Place Structure Here")
                        elif placement == -2: raise AssertionError("Cannot Afford Structure")
                        elif placement == -3: raise AssertionError("Reached Max Structure Limit")
                        
                        reward += SETTLEMENT_REWARD(len(curr_player.settlements))
                        
                        if self.init_build < 2:
                            curr_player.first_settlement = (e.parent, self._convert_e(e.n))
                        self.init_road_build = True
                        self.init_settlement_build = False

            elif self.init_road_build and self.init_build < 4:
                assert self.build_road_action_space.contains(action), "Invalid Road Position"

                for i,s in enumerate(self.board.all_sides):
                    if i == action:
                        placement = self.board.place_struct(curr_player, s.parent, self._convert_s(s.n), Structure.ROAD)
                        if placement == -1: raise AssertionError("Cannot Place Structure Here")
                        elif placement == -2: raise AssertionError("Cannot Afford Structure")
                        elif placement == -3: raise AssertionError("Reached Max Structure Limit")

                        reward += ROAD_REWARD(curr_player.get_player_s2r(), len(curr_player.settlements))

                        self.init_road_build = False
                        self.init_settlement_build = True
                        self.current_player = (self.current_player + 1) % self.num_players
                        self.init_build += 1
            
            if self.init_build == 4:
                for p in self.board.players:
                    self.board.give_resources(p, 0, p.first_settlement)
                self.board.turn_number += 1

                self._resouce_collection_round()

        obs = self._decode_observation(self.state)

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
            reward += LONGEST_ROAD_REWARD
        self.prev_longest_road_owner = self.board.get_longest_road_owner()

        # Check for game end condition
        if any(player >= self.max_victory_points for player in obs["victory_points"]):
            done = True
            if curr_player.vp >= self.max_victory_points:
                reward = WIN_REWARD
            else: 
                reward = LOSE_REWARD

        info = obs

        # Return the observation, reward, done, and info
        return self._encode_observation(obs), reward, done, trunc, info

    def render(self):
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


    def close(self):
        """Close any resources used by the environment."""
        pass