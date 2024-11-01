from Board import Board
import random
from enums import Biome, Resource, Structure, HexEnum
from Bank import Bank
from Hex import HexBlock
from Player import Player
from Die import Die

game = True

while game:

    board = Board(["ali", "jak"], 1)

    board.make_board()

    board.players[0].add_2_inv([2,2,2,2])
    board.players[1].add_2_inv([2,2,2,2])


    board.place_struct(board.players[0], board.h3, HexEnum.E6, Structure.SETTLEMENT)
    board.place_struct(board.players[0], board.h3, HexEnum.S1, Structure.ROAD)

    board.place_struct(board.players[1], board.h2, HexEnum.E6, Structure.SETTLEMENT)

    board.place_struct(board.players[1], board.h2, HexEnum.S6, Structure.ROAD)


    print(board)

    print(board.hex_nums())

    print(board.robber_loc)

    print(board.hex_biomes())

    game = False