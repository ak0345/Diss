__all__ = ["Bank", "Board", "enums", "Die", "Hex", "Player"]

from . import Bank
from . import enums
from . import Die
from . import Hex
from . import Player
from . import Board

from gymnasium.envs.registration import register

# Automatically register the MiniCatanEnv environment
register(
    id="MiniCatanEnv-v0",
    entry_point="mini_catan.CatanEnv:MiniCatanEnv",
    max_episode_steps=1000,
)

# Allow importing MiniCatanEnv directly from the package
from .CatanEnv import MiniCatanEnv