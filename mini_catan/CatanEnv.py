import gymnasium as gym
from gymnasium import spaces
import numpy as np
from mini_catan.Board import Board
import random
from mini_catan.enums import Biome, Resource, Structure, HexCompEnum
from mini_catan.Hex import HexBlock
from mini_catan.Player import Player
from mini_catan.Die import Die

# Reward variables
S_max = 5
n_type = 4

alpha = 0.5
U = lambda x: -np.exp(-alpha * x)
R_avg = lambda R_r: np.sum(R_r) / n_type

# Reward functions
ROAD_REWARD = lambda S_2R, S_p: 1 + 1.15 * ( S_2R / max(S_p, 1))
SETTLEMENT_REWARD = lambda S_p: 2 + 1.15 * (S_p / S_max)
TRADE_BANK_REWARD = lambda d_r: 1 + U(0.25 + np.sum(d_r))
TRADE_PLAYER_REWARD = lambda d_r: 1 + U(0.5 + np.sum(d_r))
REJECTED_TRADE_REWARD = lambda T_R: U(1.5 + np.power(T_R,2))
COUTNER_OFFER_REJECTED_REWARD = lambda T_R: U(2 + np.power(T_R,3))
COUTNER_OFFER_ACCEPTED_REWARD = lambda d_r: 1 + U(2.5 + np.sum(d_r))
INVENTORY_BALANCE_REWARD = lambda T_n, d_r, R_r: U(T_n + 1 + np.sum(np.abs(d_r - R_avg(R_r))))
LONGEST_ROAD_REWARD = 2
END_TURN_REWARD = 0.15
WIN_REWARD = 10
LOSE_REWARD = -10

class MiniCatanEnv(gym.Env):
    """Custom Gym Environment for Catan."""
    
    #metadata = {"render_modes": ["human"], "render_fps": 30}
    
    def __init__(self, render_mode=None):
        super().__init__()
        
        self.render_mode = render_mode # human, bot
        self.num_players = 2  # Modify based on your game setup
        self.max_victory_points = 10

        self.board = Board(["P1", "P2"])

        self._reset_followup_variables()
        
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
        """
        self.obs_space_size = (
            (self.num_players * 4) +      # inventories
            (24) +                        # edges
            (30) +                        # sides
            (1) +                         # longest_road_owner
            (self.num_players) +          # victory_points
            (self.board.board_size) +     # biomes
            (self.board.board_size) +     # hex_nums
            (1)                           # robber_loc
        )
        self.observation_space = spaces.Box(low = 0, high = 20, shape=(self.obs_space_size, ), dtype=np.int32)


        # Initial state
        self.state = self._get_initial_state()
        self.current_player = 0

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
            np.array([state["robber_loc"]])  # Robber location
        ))
        return obs
    
    def _reset_followup_variables(self):

        self.init_settlement_build = True
        self.init_road_build = False
        self.init_build = 0
        self.waiting_for_followup = False
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
        
        return {
            "inventories": inventories,
            "edges": edges,
            "sides": sides,
            "longest_road_owner": longest_road_owner,
            "victory_points": victory_points,
            "biomes": biomes,
            "hex_nums": hex_nums,
            "robber_loc": robber_loc
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
            if self.waiting_for_road_build_followup:
                assert self.build_road_action_space.contains(action), "Invalid Road Position"

                for i,s in enumerate(self.board.all_sides):
                    if i == action:
                        placement = self.board.place_struct(curr_player, s.parent, self._convert_s(s.n), Structure.ROAD)
                        if placement == -1: raise AssertionError("Cannot Place Structure Here")
                        elif placement == -2: raise AssertionError("Cannot Afford Structure")
                        elif placement == -3: raise AssertionError("Reached Max Structure Limit")

                        self.waiting_for_road_build_followup = False

            elif self.waiting_for_settlement_build_followup:
                assert self.build_settlement_action_space.contains(action), "Invalid Settlement Position"

                for i,e in enumerate(self.board.all_edges):
                    if i == action:
                        placement = self.board.place_struct(curr_player, e.parent, self._convert_e(e.n), Structure.SETTLEMENT)
                        if placement == -1: raise AssertionError("Cannot Place Structure Here")
                        elif placement == -2: raise AssertionError("Cannot Afford Structure")
                        elif placement == -3: raise AssertionError("Reached Max Structure Limit")

                        self.waiting_for_settlement_build_followup = False

            elif self.waiting_for_b_trade_followup:
                assert self.bank_trade_action_space.contains(action), "Invalid Bank Trade Action"
                assert any(a>0 for a in action[0]), "Cannot Offer Nothing"

                trade = curr_player.trade_I_with_p(self.board.bank, action[0] * 2, action[1])
                assert trade, "Cannot Afford Trade"
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
                        reward = 2 #trade accepted and ressources gained
                        
                    elif action == 1:
                        self.current_player = (self.current_player + 1) % self.num_players
                        self.waiting_for_p_trade_followup_1 = False
                        reward = -1.5 # trade rejected
                        
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
                    reward = -1 # trade rejected

                else:
                    #send offer
                    assert self.player_trade_action_space.contains(action), "Invalid Player Trade Action"
                    self.offer = action
                    trade = curr_player.trade_cost_check(self.board.players[(self.current_player + 1) % self.num_players], action[0], action[1])
                    assert trade, "Cannot Afford Trade"

                    self.reply_to_offer = True
                    self.current_player = (self.current_player + 1) % self.num_players

            elif self.waiting_for_p_trade_followup_2:
                assert self.counter_offer_response_action_space.contains(action), "Invalid Counter Offer Response Action"

                if action == 0:
                    trade = self.board.players[(self.current_player + 1) % self.num_players].trade_I_with_p(curr_player, self.offer[0], self.offer[1])
                    assert trade, "Cannot Afford Trade"
                    reward = 0.25 # counter accepted
                elif action == 1:
                    #end trade
                    reward = 0.5 # rejecting a bad trade
                elif action == 2:
                    self.waiting_for_p_trade_followup_3 = True
                    reward = 0.5 # countering a bad trade
                self.waiting_for_p_trade_followup_2 = False


            elif self.waiting_for_p_trade_followup_3:
                if self.reply_to_offer:
                    assert self.counter_counter_offer_reply_action_space.contains(action), "Invalid Counter Counter Offer Reply Action"

                    if action == 0:
                        self.current_player = (self.current_player + 1) % self.num_players
                        trade = self.board.players[self.current_player].trade_I_with_p(curr_player, self.offer[0], self.offer[1])
                        assert trade, "Cannot Afford Trade"
                        reward = 2.5 #trade accepted and ressources gained and negotiated well
                        
                    elif action == 1:
                        self.current_player = (self.current_player + 1) % self.num_players
                        reward = -2 # trade rejected twice

                    self.reply_to_offer = False
                    self.waiting_for_p_trade_followup_3 = False

                else:
                    assert self.counter_counter_offer_action_space.contains(action), "Invalid Counter Counter Offer Action"
                
                    self.offer = action
                    trade = curr_player.trade_cost_check(self.board.players[(self.current_player + 1) % self.num_players], action[0], action[1])
                    assert trade, "Cannot Afford Trade"

                    self.current_player = (self.current_player + 1) % self.num_players
                    self.reply_to_offer = True
                
            
            else:
                assert self.action_space.contains(action), "Invalid Action"
                
                if action == 0:  # Build Road
                    print("Player builds a road.")
                    reward = ROAD_REWARD  # Modify as per game logic
                    self.waiting_for_road_build_followup = True

                elif action == 1:  # Build Settlement
                    print("Player builds a settlement.")
                    reward = 2  # Modify as per game logic
                    self.waiting_for_settlement_build_followup = True

                elif action == 2:  # Trade with player
                    print("Player initiates a trade with Player.")
                    reward = 0.5  # Modify as per game logic
                    self.waiting_for_p_trade_followup_1 = True

                elif action == 3:  # Trade with Bank
                    print("Player initiates a trade with Bank.")
                    reward = 0.35  # Modify as per game logic
                    self.waiting_for_b_trade_followup = True

                elif action == 4:  # End Turn
                    print("Player Ends Turn.")
                    self.current_player = (self.current_player + 1) % self.num_players
                    self.board.turn_number += 1
                    reward = 0.15

                    self._resouce_collection_round()

        else: 
            #build set 1 and road 1 and then set 2 and road 2

            if self.init_settlement_build and self.init_build < 4:
                assert self.build_settlement_action_space.contains(action), "Invalid Settlement Position"

                for i,e in enumerate(self.board.all_edges):
                    if i == action:
                        placement = self.board.place_struct(curr_player, e.parent, self._convert_e(e.n), Structure.SETTLEMENT)
                        if placement == -1: raise AssertionError("Cannot Place Structure Here")
                        elif placement == -2: raise AssertionError("Cannot Afford Structure")
                        elif placement == -3: raise AssertionError("Reached Max Structure Limit")

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

        # Check for game end condition
        if any(player >= self.max_victory_points for player in obs["victory_points"]):
            done = True
            reward = 10

        info = obs

        # Return the observation, reward, done, and info
        return self._encode_observation(obs), reward, done, trunc, info

    def render(self):
        """Render the game state."""
        if self.render_mode == "human":
            print(f"Current Player: {self.current_player}")
            print("inventories:", self.state["inventories"])
            print("Map Edges:", self.state["edges"])
            print("Map Sides:", self.state["sides"])
            print("Victory Points:", self.state["victory_points"])
            print(self.board.get_board_array())

    def close(self):
        """Close any resources used by the environment."""
        pass


    