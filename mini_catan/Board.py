

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
                            n = link.n
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

        #for v in visited:
            #print(hn_name(v.parent.coords), f"S{v.n + 1}", "--- NOT MY ROAD" if v.value == self.players[1].tag else "" )

        return visited
    
    def longest_road(self, p):
        other_player_tags = [p_.tag for p_ in self.players if p_.tag != p.tag]
        visited = set()

        def is_negated(path, p, other_player_tags):
            def check_adjacent_sides(side):
                edge_i = side.n
                adjacent_sides = [
                    side.parent.sides[edge_i],
                    side.parent.sides[(edge_i - 1) % len(side.parent.sides)],
                    side.parent.sides[(edge_i + 1) % len(side.parent.sides)],
                ]
                count_p_tag = sum(1 for s in adjacent_sides if s.value == p.tag)
                count_other_tag = sum(1 for s in adjacent_sides if s.value in other_player_tags)
                return count_p_tag == 2 and count_other_tag == 1

            for side in path:
                if check_adjacent_sides(side):
                    return True
                for link in side.parent.edges[side.n].links:
                    if link and check_adjacent_sides(link):
                        return True
            return False

        def is_connected(side1, side2):
            """Check if two sides are directly connected."""
            if side1.parent == side2.parent:
                # Direct neighbors within the same hex
                if abs(side1.n - side2.n) in {1, len(side1.parent.sides) - 1}:
                    return True

            # Check connections via linked edges
            for link in side1.parent.edges[side1.n].links:
                if link and (link == side2 or link.parent.sides[(link.n - 1) % len(link.parent.sides)] == side2):
                    return True

            return False

        def find_path(start_side):
            """Find all connected sides forming a single road."""
            path = []
            stack = [start_side]

            while stack:
                side = stack.pop()
                if side in visited or side.value != p.tag:
                    continue

                visited.add(side)
                if len(path) > 0:
                    if path[-1].links != side:
                        path.append(side)
                else:
                    path.append(side)

                # Add all connected sides to the stack
                for neighbor in side.parent.sides:
                    if neighbor not in visited and is_connected(side, neighbor):
                        stack.append(neighbor)
                
                if side.links:
                    for neighbor in side.links.parent.sides:
                        if neighbor not in visited and is_connected(side, neighbor):
                            stack.append(neighbor)

            return path

        # Identify distinct paths
        paths = []
        for side in self.longest_road_queue(p):
            if side not in visited and side.value == p.tag:
                paths.append(find_path(side))

        # Validate paths
        valid_paths = []
        for path in paths:
            if not is_negated(path, p, other_player_tags):
                valid_paths.append(path)

        # Output results
        #for i, path in enumerate(valid_paths):
            #print(f"Path {i + 1}: {[(hn_name(p.parent.coords), f'S{p.n + 1}') for p in path]}")

        return max([len(path) for path in valid_paths])

    
    def place_struct(self, p, hn, pos, struct):
        if p.cost_check(struct):
            if hn.pos_is_empty(pos, struct) and hn.check_nearby(pos, struct, p, self.turn_number):
                    hn.place_struct_in_pos(pos, struct, p)
                    p.build_struct(struct)
                    if struct == Structure.SETTLEMENT:
                        p.inc_vp()
                    elif struct == Structure.ROAD:
                        longest = self.longest_road(p)
                        p.longest_road = longest
                        if longest >= self.min_longest_road:  # Minimum 5 roads required for "Longest Road"
                            if longest > self.current_longest_road:
                                    if self.longest_road_owner is not None:
                                        self.longest_road_owner.dec_vp()
                                        #print(f"{self.longest_road_owner.name} lost a vp")
                                    self.longest_road_owner = p
                                    self.longest_road_owner.inc_vp()
                                    #print(f"{self.longest_road_owner.name} gained a vp")
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