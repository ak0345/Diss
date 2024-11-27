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
    
    metadata = {"render_modes": ["human"], "render_fps": 30}
    
    def __init__(self, render_mode=None):
        super().__init__()
        
        self.render_mode = render_mode # human, bot
        self.num_players = 2  # Modify based on your game setup
        self.max_victory_points = 10

        self.board = Board()
        
        # Define action space (e.g., 4 actions: Build Road, Build Settlement, Trade, End Turn)
        self.action_space = spaces.Discrete(4)
        self.build_settlement_action_space = spaces.Discrete(24)
        self.build_road_action_space = spaces.Discrete(30)
        self.trade_action_space = spaces.Dict({
            "offer": spaces.Box(low=0, high=10, shape=(4,), dtype=np.int32),  # Resources offered
            "request": spaces.Box(low=0, high=10, shape=(4,), dtype=np.int32),  # Resources requested
            "other_player_response": spaces.Discrete(3),  # 0: reject, 1: accept, 2: counter-offer
            "response_to_counter": spaces.Discrete(2)  # 0: reject, 1: accept
        })
        
        # Define observation space (simplified example: resources, settlements, roads)
        self.observation_space = spaces.Dict({
            "inventories": spaces.Box(low=0, high=20, shape=(self.num_players, 4), dtype=np.int32), #4 types of resources
            "edges": spaces.Box(low=0, high=self.num_players, shape=(self.num_players, 24), dtype=np.int32), #edges/settlements
            "sides": spaces.Box(low=0, high=self.num_players, shape=(self.num_players, 30), dtype=np.int32), #sides/roads
            "longest_road": spaces.Box(low=0, high=20, shape=(self.num_players, 1), dtype=np.int32),
            "victory_points": spaces.Box(low=0, high=10, shape=(self.num_players, 1), dtype=np.int32), #victory points
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
        return {
            "inventories": np.zeros((self.num_players, 4), dtype=np.int32), #np.array([[4, 4, 2, 2] for _ in range(self.num_players)], dtype=np.int32),
            "edges": np.zeros((self.num_players, 24), dtype=np.int32),
            "sides": np.zeros((self.num_players, 30), dtype=np.int32),
            "longest_road": np.zeros((self.num_players, 1), dtype=np.int32),
            "victory_points": np.zeros((self.num_players, 1), dtype=np.int32),
        }
    
    def reset(self, seed=None, options=None):
        """Reset the environment to the initial state."""
        super().reset(seed=seed)
        self.state = self._get_initial_state()
        self.current_player = 0
        return self.state, {}

    def step(self, action):#, player):
        """Perform an action in the environment."""
        assert self.action_space.contains(action), "Invalid Action"

        # Placeholder logic for handling actions
        reward = 0
        done = False
        info = {}

        if action == 0:  # Build Road
            print("Player builds a road.")
            reward = 1  # Modify as per game logic

        elif action == 1:  # Build Settlement
            print("Player builds a settlement.")
            reward = 2  # Modify as per game logic

        elif action == 2:  # Trade
            print("Player initiates a trade.")
            reward = 0.5  # Modify as per game logic
            #if trade is successful:
                #reward = 3
            #else:
                #reward = -1

        elif action == 3:  # End Turn
            self.current_player = (self.current_player + 1) % self.num_players

        #self.state["edges"] = self.board.get_edges()
        #self.state["sides"] = self.board.get_sides()
        #self.state["victory_points"] = self.board.get_vp()
        #self.state["inventories"] = self.board.get_all_invs()
        #self.state["longest_road"] = self.board.get_longest_road()
        #self.state["robber_loc"] = self.board.get_robber_loc()

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
            print("Victory Points:", self.state["victory_ponts"])
            print(Board)

    def close(self):
        """Close any resources used by the environment."""
        pass