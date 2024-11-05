

import random
from enums import Biome, Resource, Structure, HexCompEnum
from Bank import Bank
from Hex import HexBlock
from Player import Player
from Die import Die

class Board:
    def __init__(self, player_names, num_dice):
        self.players = [Player(name) for name in player_names]
        self.bank = Bank(20)
        self.dice = [Die() for _ in range(num_dice)]
        self.robber_loc = -1
        self.turn_number = 0
        #setting board numbers
        self.board_size = 7

        #defining Hex blocks
        self.h1 = HexBlock(0,1)
        self.h2 = HexBlock(-1,0)
        self.h3 = HexBlock(1,1)
        self.h4 = HexBlock(0,0) #Center
        self.h5 = HexBlock(-1,-1)
        self.h6 = HexBlock(1,0)  
        self.h7 = HexBlock(0,-1)
        self.h1.set_sides_edges(None, None, self.h3, self.h4, self.h2, None)
        self.h2.set_sides_edges(None, self.h1, self.h4, self.h5, None, None)
        self.h3.set_sides_edges(None, None, None, self.h6, self.h4, self.h1)
        self.h4.set_sides_edges(self.h1, self.h3, self.h6, self.h7, self.h5, self.h2)
        self.h5.set_sides_edges(self.h2, self.h4, self.h7, None, None, None)
        self.h6.set_sides_edges(self.h3, None, None, None, self.h7, self.h4)
        self.h7.set_sides_edges(self.h4, self.h6, None, None, None, self.h5)
        
        self.map_hexblocks = [self.h1, self.h2, self.h3, self.h4, self.h5, self.h6, self.h7]
        
    def set_biomes(self):
        biome_distribution = [Biome.FOREST, Biome.HILLS, Biome.FIELDS, Biome.PASTURE, Biome.DESERT]
        while len(biome_distribution) < self.board_size:
            biome_distribution.append(random.choice([Biome.FOREST, Biome.HILLS, Biome.FIELDS, Biome.PASTURE]))
        
        random.shuffle(biome_distribution)
        for hex, biome in zip(self.map_hexblocks, biome_distribution):
            hex.set_biome(biome)

    def set_hex_nums(self):
        num_pool = [i for i in range(1,(6*len(self.dice)))] # because final number for desert

        while len(num_pool) < self.board_size-1: #all non desert hexes
            num_pool.append(random.choice(num_pool))
        
        num_pool.append(6*len(self.dice))
        random.shuffle(num_pool)

        for i, (hex, num) in enumerate(zip(self.map_hexblocks, num_pool)):
            hex.set_tile_num(num)
            if num == 6:
                self.move_robber(i)

    def make_board(self):
        self.set_biomes()
        self.set_hex_nums()

    def move_robber(self, i):
        self.robber_loc = i
    
    def place_struct(self, p, hn, pos, struct):    
        if p.cost_check(struct):
            if hn.pos_is_empty(pos, struct) and hn.check_nearby(pos, struct, p, self.turn_number):
                    hn.place_struct_in_pos(pos, struct, p)
                    p.build_struct(struct)
                    if struct == Structure.SETTLEMENT:
                        p.inc_vp()
                    return True
            else:
                print("cannot place structure here")
        else:
            print("cannot afford structure")

    def give_resources(self, p, d_i=0, ignore_struct=None):
        #[Wood, Brick, Sheep, Wheat]
        p_inv = [0, 0, 0, 0]
        for i, hex in enumerate(self.map_hexblocks):
            if d_i == 0 or hex.tile_num == d_i:
                for edge in hex.edges:
                    if ignore_struct:
                        if ignore_struct[0] != hex and ignore_struct[1] != edge:
                            if edge.value == p.tag:
                                if hex.biome.value and self.robber_loc != i:
                                    p_inv[hex.biome.value.value] += 1
                            
        p.add_2_inv(p_inv)
        print(f"Given Player {p.name}: {p_inv[0]} Wood, {p_inv[1]} Brick, {p_inv[2]} Sheep, {p_inv[3]} Wheat")

    def roll_dice(self):
        for die in self.dice:
            die.roll()
        return sum([x.value for x in self.dice])
    
    def __str__(self):
        #show board
        h1 = self.h1.values()
        h2 = self.h2.values()
        h3 = self.h3.values()
        h4 = self.h4.values()
        h5 = self.h5.values()
        h6 = self.h6.values()
        h7 = self.h7.values()

        return str(h1) + str(h2) + str(h3) + str(h4) + str(h5) + str(h6) + str(h7)
    
    def hex_nums(self):
        h1 = self.h1.tile_num
        h2 = self.h2.tile_num
        h3 = self.h3.tile_num
        h4 = self.h4.tile_num
        h5 = self.h5.tile_num
        h6 = self.h6.tile_num
        h7 = self.h7.tile_num

        return str(h1) +" "+ str(h2) +" "+ str(h3) +" "+ str(h4) +" "+ str(h5) +" "+ str(h6) +" "+ str(h7) 
    
    def hex_biomes(self):
        h1 = self.h1.biome
        h2 = self.h2.biome
        h3 = self.h3.biome
        h4 = self.h4.biome
        h5 = self.h5.biome
        h6 = self.h6.biome
        h7 = self.h7.biome

        return str(h1) +" "+ str(h2) +" "+ str(h3) +" "+ str(h4) +" "+ str(h5) +" "+ str(h6) +" "+ str(h7)