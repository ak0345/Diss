

import random
from collections import deque
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
        self.min_longest_road = 3
        self.current_longest_road = 0
        self.longest_road_owner = None

        #defining Hex blocks
        self.h1 = HexBlock(0,1)
        self.h2 = HexBlock(-1,0)
        self.h3 = HexBlock(1,1)
        self.h4 = HexBlock(0,0) #Center
        self.h5 = HexBlock(-1,-1)
        self.h6 = HexBlock(1,0)  
        self.h7 = HexBlock(0,-1)
        self.h1.set_sides_edges(None, self.h3, self.h4, self.h2, None, None)
        self.h2.set_sides_edges(self.h1, self.h4, self.h5, None, None, None)
        self.h3.set_sides_edges(None, None, self.h6, self.h4, self.h1, None)
        self.h4.set_sides_edges(self.h3, self.h6, self.h7, self.h5, self.h2, self.h1)
        self.h5.set_sides_edges(self.h4, self.h7, None, None, None, self.h2)
        self.h6.set_sides_edges(None, None, None, self.h7, self.h4, self.h3)
        self.h7.set_sides_edges(self.h6, None, None, None, self.h5, self.h4)
        
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

    def longest_road_queue(self, p):
        def hn_name(hn):
            match hn:
                case (0,1):
                    return "h1"
                case (-1,0):
                    return "h2"
                case (1,1):
                    return "h3"
                case (0,0):
                    return "h4"
                case (-1,-1):
                    return "h5"
                case (1,0):
                    return "h6"
                case (0,-1):
                    return "h7"
                case _:
                    return "woowoo"
                
        other_player_tags = [p_.tag for p_ in self.players if p_.tag != p.tag]

        visited = deque()
        visited_set = set()
        for hex in self.map_hexblocks:
            for i, edge in enumerate(hex.edges):
                side = hex.sides[i]
                # Check if the side has already been visited
                if side not in visited_set and side.value == p.tag:
                    visited.appendleft(side)
                    visited_set.add(side)
                    if side.links:
                        visited_set.add(side.links)
                elif side not in visited_set and side.value in other_player_tags:
                    if hex.check_nearby(HexCompEnum(side.n), Structure.ROAD, self.players[1], self.turn_number):
                        visited.appendleft(side)
                        visited_set.add(side)
                        if side.links:
                            visited_set.add(side.links)
                
                if edge.links:
                    for link in edge.links:
                        if link:
                            ns = [link.n]#, (link.n - 1)  % len(link.parent.sides), (link.n + 1) % len(link.parent.sides)]
                            for n in ns:
                                side_2_add_2 = link.parent.sides[n]
                                if side_2_add_2 not in visited_set and side_2_add_2.value == p.tag:
                                    visited.appendleft(side_2_add_2)
                                    visited_set.add(side_2_add_2)
                                    if side_2_add_2.links:
                                        visited_set.add(side_2_add_2.links)
                                elif side_2_add_2 not in visited_set and side_2_add_2.value in other_player_tags:
                                    if link.parent.check_nearby(HexCompEnum(side_2_add_2.n), Structure.ROAD, self.players[1], self.turn_number):
                                        visited.appendleft(side_2_add_2)
                                        visited_set.add(side_2_add_2)
                                        if side_2_add_2.links:
                                            visited_set.add(side_2_add_2.links)

        for v in visited:
            print(hn_name(v.parent.coords), f"S{v.n + 1}", "--- NO" if v.value == self.players[1].tag else "" )

        return visited
    
    def longest_road(self, p):
        def hn_name(hn):
            match hn:
                case (0,1):
                    return "h1"
                case (-1,0):
                    return "h2"
                case (1,1):
                    return "h3"
                case (0,0):
                    return "h4"
                case (-1,-1):
                    return "h5"
                case (1,0):
                    return "h6"
                case (0,-1):
                    return "h7"
                case _:
                    return "woowoo"
                
        def is_connected(side1, side2):
            """
            Determines if two road segments (hex1, side1) and (hex2, side2) are connected.
            Modify this function based on the game board logic.
            """
            #if side1.parent == side2.parent:
                #if side1.n + 1 == side2.n or side1.n - 1 == side2.n:
                    #return True
            
            i = side1.n
            (self.sides[(i+1) % len(self.sides)] == side2 or 
            self.sides[(i-1) % len(self.sides)] == side2)
            
            # Example logic: If sides are direct neighbors or linked via the same road
            #return True  # Replace with actual logic


        def is_negated(segment, path):
            """
            Determines if a segment invalidates a path (e.g., another player's road blocks it).
            Modify this logic based on game rules.
            """
            # Example: If a segment belongs to another player and links with this path
            return False  # Replace with actual logic

        queue = self.longest_road_queue(p)
        paths = []  # List to store distinct paths (each path is a list of road segments)

        for segment in queue:
            side = segment

            # Flag to check if the segment was added to an existing path
            added_to_path = False

            for path in paths:
                # Check if this segment connects to the last segment in the path
                last_side = path[-1]
                if is_connected(last_side, side):  # You define is_connected logic
                    path.append(side)  # Add to this path
                    added_to_path = True
                    break

            # If not added to any path, start a new path
            if not added_to_path:
                paths.append([side])

        # Post-processing: Remove negated paths if necessary
        valid_paths = []
        for path in paths:
            if not any(is_negated(segment, path) for segment in path):  # Define is_negated logic
                valid_paths.append(path)

        for i, path in enumerate(valid_paths):
            print(f"Path {i + 1}: {[(hn_name(p.parent.coords), f"S{p.n + 1}") for p in path]}")

        return 0
    
    def place_struct(self, p, hn, pos, struct):
        if p.cost_check(struct):
            if hn.pos_is_empty(pos, struct) and hn.check_nearby(pos, struct, p, self.turn_number):
                    hn.place_struct_in_pos(pos, struct, p)
                    p.build_struct(struct)
                    if struct == Structure.SETTLEMENT:
                        p.inc_vp()
                    elif struct == Structure.ROAD:
                        longest = self.longest_road(p)
                        print(longest)
                        p.longest_road = longest
                        if longest >= self.min_longest_road:  # Minimum 5 roads required for "Longest Road"
                            if longest > self.current_longest_road:
                                    if self.longest_road_owner is not None:
                                        self.longest_road_owner.dec_vp()
                                        print(f"{self.longest_road_owner.name} lost a vp")
                                    self.longest_road_owner = p
                                    self.longest_road_owner.inc_vp()
                                    print(f"{self.longest_road_owner.name} gained a vp")
                                    self.current_longest_road = longest
                                    print(f"Player {p.name} now has the Longest Road: {longest}")
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
    
    def get_board_array(self):
        #show board
        h1 = self.h1.values()
        h2 = self.h2.values()
        h3 = self.h3.values()
        h4 = self.h4.values()
        h5 = self.h5.values()
        h6 = self.h6.values()
        h7 = self.h7.values()

        """for h in [self.h1, self.h2, self.h3, self.h4, self.h5, self.h6, self.h7]:
            print(f"Hex ({h.x}, {h.y}):")
            for i, side in enumerate(h.sides):
                linked_hex = side.links.parent if side.links else None
                linked_side = side.links.n if side.links else None
                print(f"  Side {i} links to Hex ({linked_hex.x if linked_hex else "None"}, {linked_hex.y if linked_hex else "None"}), Side {linked_side}")"""

        return[h1, h2, h3, h4, h5, h6, h7]
    
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