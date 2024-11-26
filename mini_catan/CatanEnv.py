import gymnasium as gym
from gymnasium import spaces
import numpy as np
from Board import Board
from enums import Structure

class MiniCatanEnv(gym.Env):
    def __init__(self, player_names=["Player1", "Player2"], win_vp=5):
        super(MiniCatanEnv, self).__init__()
        
        # Initialize the board
        self.board = Board(player_names, win_vp)
        self.board.make_board()
        self.win_vp = win_vp
        
        # Define observation space (e.g., board state, inventories)
        self.observation_space = spaces.Dict({
            #structure_mapping = {
            #    None: 0,
            #    Structure.ROAD: 1,
            #    Structure.SETTLEMENT: 2
            #}
            #player_mapping = {
            #    None: 0,
            #    "Player1": 1,
            #    "Player2": 2
            #}
            "board_state": spaces.Box(low=0, high=2, shape=(7, 2, 6, 2), dtype=np.float32),  # Example: abstract board
            "player_inventories": spaces.Box(low=0, high=20, shape=(len(player_names), 4), dtype=np.int32),  # Resources
        })
        
        # Define action space (e.g., place settlement, build road)
        self.action_space = spaces.Dict({
            "action_type": spaces.Discrete(3),  # 0: Place Settlement, 1: Build Road, 2: Trade
            "hex_index": spaces.Discrete(7),  # Hex position {h1, h2, ..., h7}
            "location": spaces.Discrete(6),   # Location (side or edge) {S1, S2, ..., S6} or {E1, E2, ..., E6}
        })

        self.current_player = 0
        self.done = False

    def reset(self):
        """Resets the game to the initial state."""
        self.board = Board(["Player1", "Player2"], self.win_vp)
        self.board.make_board()
        self.current_player = 0
        self.done = False
        return self._get_obs()
    
    def step(self, action):
        """
        Executes a given action in the environment.
        action: dict with keys "action_type", "hex_index", "location"
        """
        if self.done:
            raise RuntimeError("Game is over. Please reset the environment.")

        # Extract action details
        action_type = action["action_type"]
        hex_index = action["hex_index"]
        location = action["location"]
        player = self.board.players[self.current_player]

        reward = 0
        if action_type == 0:  # Place Settlement
            pos_obj = self.board.get_hex_component(hex_index, location, Structure.SETTLEMENT)
            if self.board.place_struct(player, self.board.map_hexblocks[hex_index], pos_obj, Structure.SETTLEMENT):
                reward = 1  # Example reward
        elif action_type == 1:  # Build Road
            pos_obj = self.board.get_hex_component(hex_index, location, Structure.ROAD)
            if self.board.place_struct(player, self.board.map_hexblocks[hex_index], pos_obj, Structure.ROAD):
                reward = 0.5  # Example reward
        elif action_type == 2:  # Trade
            # Define simple trade logic for now
            reward = 0.1  # Small reward for trading
            
        # Check for win condition
        if player.vp >= self.win_vp:
            self.done = True
            reward = 10  # Winning reward
        
        # Update current player
        self.current_player = (self.current_player + 1) % len(self.board.players)

        return self._get_obs(), reward, self.done, {}

    def _get_obs(self):
        """Returns the current state of the environment."""
        board_state = self.board.get_board_state()  # You need to implement this
        player_inventories = [p.inventory for p in self.board.players]
        return {
            "board_state": board_state,
            "player_inventories": np.array(player_inventories),
        }

    def render(self, mode="human"):
        """Optional rendering for visualization."""
        self.board.print_board()  # You can define this method in your `Board` class

