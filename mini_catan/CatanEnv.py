import gymnasium as gym
from gymnasium import spaces
import numpy as np
from Board import Board
import random
from enums import Biome, Resource, Structure, HexCompEnum
from Hex import HexBlock
from Player import Player
from Die import Die

class MiniCatanEnv(gym.Env):
    """Custom Gym Environment for Catan."""
    
    #metadata = {"render_modes": ["human"], "render_fps": 30}
    
    def __init__(self, render_mode=None):
        super().__init__()
        
        self.render_mode = render_mode # human, bot
        self.num_players = 2  # Modify based on your game setup
        self.max_victory_points = 10

        self.board = Board(["P1", "P2"])
        
        # Define action space (e.g., 5 actions: Build Road, Build Settlement, Trade with Player, Trade with Bank, End Turn)
        self.action_space = spaces.Discrete(5)
        self.waiting_for_followup = False
        
        self.waiting_for_settlement_build_followup = False
        self.build_settlement_action_space = spaces.Discrete(24)

        self.waiting_for_road_build_followup = False
        self.build_road_action_space = spaces.Discrete(30)

        self.waiting_for_p_trade_followup_1 = False
        self.player_trade_action_space = spaces.Dict({
            "offer": spaces.Box(low=0, high=10, shape=(4,), dtype=np.int32),  # Resources offered
            "request": spaces.Box(low=0, high=10, shape=(4,), dtype=np.int32),  # Resources requested
        })
        self.offer = None
        self.reply_to_offer = False
        self.trade_accepted = False
        self.trade_rejected = False
        self.player_trade_offer_request_action_space =  spaces.Discrete(3) # 0: Yes, 1: No, 2: Counter
        self.counter_offer_action_space = spaces.Dict({
            "offer": spaces.Box(low=0, high=10, shape=(4,), dtype=np.int32),
            "request": spaces.Box(low=0, high=10, shape=(4,), dtype=np.int32),
        })

        self.waiting_for_p_trade_followup_2 = False
        self.counter_offer_response_action_space = spaces.Discrete(3) # 0: Yes, 1: No, 2: Counter the Counter

        self.waiting_for_p_trade_followup_3 = False
        self.counter_counter_offer_action_space = spaces.Dict({
            "offer": spaces.Box(low=0, high=10, shape=(4,), dtype=np.int32),  # Resources offered
            "request": spaces.Box(low=0, high=10, shape=(4,), dtype=np.int32),  # Resources requested
        })
        self.counter_counter_offer_reply_action_space =  spaces.Discrete(2) # 0: Yes, 1: No


        self.waiting_for_b_trade_followup = False
        self.bank_trade_action_space = spaces.Dict({
            "offer": spaces.Box(low=0, high=10, shape=(4,), dtype=np.int32),  # Resources offered
            "request": spaces.Box(low=0, high=10, shape=(4,), dtype=np.int32),  # Resources requested
        })
        
        # Define observation space (simplified example: resources, settlements, roads)
        self.observation_space = spaces.Dict({
            "inventories": spaces.Box(low=0, high=20, shape=(self.num_players, 4), dtype=np.int32), # 4 types of resources
            "edges": spaces.Box(low=0, high=1, shape=(self.num_players, 24), dtype=np.int32), # Edges indexed 0 - 23
            "sides": spaces.Box(low=0, high=1, shape=(self.num_players, 30), dtype=np.int32), # Sides indexed 0 - 29
            "longest_road_owner": spaces.Discrete(self.num_players + 1), # 0: No One, 1: Player 1, 2: Player 2
            "victory_points": spaces.Box(low=0, high=10, shape=(self.num_players, 1), dtype=np.int32), # Victory points
            "biomes": spaces.Box(low=0, high=4, shape=(self.board.board_size, 1), dtype=np.int32),  # Biomes for each hex
            "hex_nums": spaces.Box(low=1, high=6, shape=(self.board.board_size, 1), dtype=np.int32),  # Hex numbers for each hex
            "robber_loc": spaces.Discrete(self.board.board_size)  # Robber location (index of hex block)
        })

        # Initial state
        self.board.make_board()
        self.state = self._get_initial_state()
        self.current_player = 0
    
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
        return {
            "inventories": np.zeros((self.num_players, 4), dtype=np.int32), #np.array([[4, 4, 2, 2] for _ in range(self.num_players)], dtype=np.int32),
            "edges": np.zeros((self.num_players, 24), dtype=np.int32),
            "sides": np.zeros((self.num_players, 30), dtype=np.int32),
            "longest_road_owner": 0,
            "victory_points": np.zeros((self.num_players, 1), dtype=np.int32),
            "biomes" : np.array([biome_num(b) for b in self.board.get_hex_biomes()]),
            "hex_nums" : np.array(self.board.get_hex_nums()),
            "robber_loc" : self.board.robber_loc,
        }
    
    def reset(self, seed=None, options=None):
        """Reset the environment to the initial state."""
        super().reset(seed=seed)
        self.state = self._get_initial_state()
        self.current_player = 0
        return self.state, {}

    def step(self, action): #, player):
        """Perform an action in the environment."""

        # Placeholder logic for handling actions
        reward = 0
        done = False
        info = {}
        curr_player = self.board.players[self.current_player]

        if self.waiting_for_road_build_followup:
            assert self.build_road_action_space.contains(action), "Invalid Road Position"

            for i,s in enumerate(self.board.all_sides()):
                if i == action:
                    self.board.place_struct(curr_player, s.parent, s, Structure.ROAD)

        elif self.waiting_for_settlement_build_followup:
            assert self.build_settlement_action_space.contains(action), "Invalid Settlement Position"

            for i,e in enumerate(self.board.all_edges()):
                if i == action:
                    self.board.place_struct(curr_player, e.parent, e, Structure.SETTLEMENT)

        elif self.waiting_for_b_trade_followup:
            assert self.bank_trade_action_space.contains(action), "Invalid Bank Trade Action"

            action["offer"] = [o*2 for o in action["offer"]]
            curr_player.trade_I_with_p(self.board.bank, action["offer"], action["request"])

        elif self.waiting_for_p_trade_followup_1:

            if self.reply_to_offer:
                assert self.player_trade_offer_request_action_space.contains(action), "Invalid Reply to Offer"
                #offer is saved in self.offer
                if action == 0:
                    self.trade_accepted = True
                    self.current_player = (self.current_player + 1) % self.num_players
                    self.waiting_for_p_trade_followup_1 = False
                    self.board.players[self.current_player].trade_I_with_p(curr_player, self.offer["offer"], self.offer["request"])
                    reward = 2 #trade accepted and ressources gained
                    
                elif action == 1:
                    self.trade_rejected = True
                    self.current_player = (self.current_player + 1) % self.num_players
                    self.waiting_for_p_trade_followup_1 = False
                    reward = -1.5 # trade rejected
                    
                elif action == 2:
                    self.counter_sent = True
                
                self.reply_to_offer = False

            elif self.counter_sent:
                assert self.counter_offer_action_space.contains(action), "Invalid Reply to Offer"

                self.counter_sent = False
                self.offer = action
                self.current_player = (self.current_player + 1) % self.num_players
                self.waiting_for_p_trade_followup_2 = True
                self.waiting_for_p_trade_followup_1 = False
                reward = -1 # trade rejected

            else:
                #send offer
                assert self.player_trade_action_space.contains(action), "Invalid Player Trade Action"

                self.reply_to_offer = True
                self.offer = action
                self.current_player = (self.current_player + 1) % self.num_players

        elif self.waiting_for_p_trade_followup_2:
            assert self.counter_offer_response_action_space.contains(action), "Invalid Counter Offer Response Action"

            if action == 0:
                self.board.players[(self.current_player + 1) % self.num_players].trade_I_with_p(curr_player, self.offer["offer"], self.offer["request"])
            elif action == 1:
                #end trade
                self.waiting_for_p_trade_followup_2 = False
                reward = 0.5 # rejecting a bad trade
            elif action == 2:
                self.waiting_for_p_trade_followup_3 = True
                reward = 0.5 # countering a bad trade


        elif self.waiting_for_p_trade_followup_3:
            if self.reply_to_offer:
                assert self.counter_counter_offer_reply_action_space.contains(action), "Invalid Counter Counter Offer Reply Action"

                if action == 0:
                    self.current_player = (self.current_player + 1) % self.num_players
                    self.board.players[self.current_player].trade_I_with_p(curr_player, self.offer["offer"], self.offer["request"])
                    reward = 2.5 #trade accepted and ressources gained and negotiated well
                    
                elif action == 1:
                    self.current_player = (self.current_player + 1) % self.num_players
                    reward = -2 # trade rejected twice

                self.reply_to_offer = False
                self.waiting_for_p_trade_followup_3 = False

            else:
                assert self.counter_counter_offer_action_space.contains(action), "Invalid Counter Counter Offer Action"
            
                self.offer = action
                self.current_player = (self.current_player + 1) % self.num_players
                self.reply_to_offer = True
            
        
        else:
            assert self.action_space.contains(action), "Invalid Action"
            
            if action == 0:  # Build Road
                print("Player builds a road.")
                reward = 1  # Modify as per game logic
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
                self.current_player = (self.current_player + 1) % self.num_players

        self.state["edges"] = self.board.get_edges()
        self.state["sides"] = self.board.get_sides()
        self.state["victory_points"] = self.board.get_vp()
        self.state["inventories"] = self.board.get_all_invs()
        self.state["longest_road_owner"] = self.board.get_longest_road_owner()
        self.state["robber_loc"] = self.board.robber_loc

        # Check for game end condition
        if any(player["victory_points"] >= self.max_victory_points for player in self.state["victory_points"]):
            done = True
            reward = 10

        # Return the observation, reward, done, and info
        return self.state, reward, done, info

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