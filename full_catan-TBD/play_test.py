from Board import Board
import random
from enums import Biome, Resource, Structure, HexCompEnum
#from Bank import Bank
from Hex import HexBlock
from Player import Player
from Die import Die


board = Board(["ali", "jak"], 1)

board.make_board()

board.players[0].add_2_inv([2,2,2,2])
board.players[1].add_2_inv([2,2,2,2])


board.place_struct(board.players[0], board.h3, HexCompEnum.E6, Structure.SETTLEMENT)
board.place_struct(board.players[0], board.h3, HexCompEnum.S1, Structure.ROAD)

board.place_struct(board.players[1], board.h2, HexCompEnum.E6, Structure.SETTLEMENT)

board.place_struct(board.players[1], board.h2, HexCompEnum.S6, Structure.ROAD)

board.give_resources(board.players[1])


print(board)

print(board.hex_nums())

print(board.robber_loc)

print(board.hex_biomes())