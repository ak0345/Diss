

import random
from mini_catan.enums import Biome, Structure
from mini_catan.Bank import Bank
from mini_catan.Hex import HexBlock
from mini_catan.Player import Player
from mini_catan.Die import Die

class Board:
    def __init__(self, player_names):
        """
        Initialize the game board.
        
        Args:
            player_names (list[str]): List of player names.
        """
        self.players = [Player(name) for name in player_names]
        self.bank = Bank(20)
        self.num_dice = 1
        self.die_sides = 6
        self.robber_num = self.num_dice * self.die_sides
        self.dice = [Die(self.die_sides) for _ in range(self.num_dice)]
        self.robber_loc = -1
        self.turn_number = 0

        # Board settings
        self.board_size = 7
        self.min_longest_road = 4
        self.current_longest_road = 0
        self.longest_road_owner = None
        self.desert_num = 6

        # Define Hex blocks
        self.h1 = HexBlock(0, 1)
        self.h2 = HexBlock(-1, 0)
        self.h3 = HexBlock(1, 1)
        self.h4 = HexBlock(0, 0)  # Center
        self.h5 = HexBlock(-1, -1)
        self.h6 = HexBlock(1, 0)
        self.h7 = HexBlock(0, -1)
        
        #Set Hex BLock Connections to form board
        self.h1.set_sides_edges(None, self.h3, self.h4, self.h2, None, None)
        self.h2.set_sides_edges(self.h1, self.h4, self.h5, None, None, None)
        self.h3.set_sides_edges(None, None, self.h6, self.h4, self.h1, None)
        self.h4.set_sides_edges(self.h3, self.h6, self.h7, self.h5, self.h2, self.h1)
        self.h5.set_sides_edges(self.h4, self.h7, None, None, None, self.h2)
        self.h6.set_sides_edges(None, None, None, self.h7, self.h4, self.h3)
        self.h7.set_sides_edges(self.h6, None, None, None, self.h5, self.h4)
        
        self.map_hexblocks = [self.h1, self.h2, self.h3, self.h4, self.h5, self.h6, self.h7]

        self.all_edges = []
        self.init_all_edges_list()

        self.all_sides = []
        self.init_all_sides_list()

    def init_all_edges_list(self):
        """
        Populate the `all_edges` list with the object instances of all unique edges from the hexagonal blocks.

        This method iterates through all the hexagonal blocks (`map_hexblocks`) and their edges. 
        It adds an edge to `all_edges` only if none of its links are already present in the list.
        """
        for hn in self.map_hexblocks:
            for e in hn.edges:
                # Check if any link in e.links is already in self.all_edges
                if not any(link in self.all_edges for link in e.links):
                    # Add edge to self.all_edges if it is not already present
                    self.all_edges.append(e)
    
    def init_all_sides_list(self):
        """
        Populate the `all_sides` list with the object instances of all unique sides from the hexagonal blocks.

        This method iterates through all the hexagonal blocks (`map_hexblocks`) and their sides. 
        It adds a side to `all_sides` only if its link is not already present in the list.
        """
        for hn in self.map_hexblocks:
            for s in hn.sides:
                # Check if s.links is already in self.all_sides
                if s.links not in self.all_sides:
                    # Add side to self.all_sides if not already present
                    self.all_sides.append(s)

    def hn_name(self, hn_coords):
        """
        Helper function to get the name of a hex based on its coordinates.
        
        Args:
            hn_coords (tuple[int, int]): Coordinates of the hex.
        
        Returns:
            str: Name of the hex or an empty string if not found.
        """
        match hn_coords:
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
                return ""
        
    def set_biomes(self):
        """
        Randomly assign biomes to the hex blocks.
        """
        biome_distribution = [Biome.FOREST, Biome.HILLS, Biome.FIELDS, Biome.PASTURE, Biome.DESERT]
        while len(biome_distribution) < self.board_size:
            biome_distribution.append(random.choice([Biome.FOREST, Biome.HILLS, Biome.FIELDS, Biome.PASTURE]))
        
        random.shuffle(biome_distribution)
        for hex, biome in zip(self.map_hexblocks, biome_distribution):
            hex.set_biome(biome)

    def set_hex_nums(self):
        """
        Randomly assign numbers to hex blocks, ensuring the robber starts on a tile with number 6.
        """
        num_pool = [i for i in range(1,self.desert_num)] # because final number for desert

        while len(num_pool) < self.board_size: #all non desert hexes
            num_pool.append(random.choice(num_pool))
        
        #num_pool.append(self.desert_num)
        random.shuffle(num_pool)

        for i, (hex, num) in enumerate(zip(self.map_hexblocks, num_pool)):
            if hex.biome == Biome.DESERT:
                hex.set_tile_num(self.desert_num)
                self.move_robber(i)
            else:
                hex.set_tile_num(num)
                

    def make_board(self):
        """
        Set up the game board by assigning biomes and numbers to hex blocks.
        """
        self.set_biomes()
        self.set_hex_nums()

    def move_robber(self, i):
        """
        Move the robber to a specified hex block.
        
        Args:
            index (int): Index of the hex block to move the robber to.
        """
        self.robber_loc = i
    
    def roll_dice(self):
        """
        Roll all dice and return the total value.
        
        Returns:
            int: The total value rolled.
        """
        for die in self.dice:
            die.roll()
        val = sum([x.value for x in self.dice])
        if val == self.robber_num:
            i = random.randint(0, self.board_size-1)
            self.move_robber(i)
            print(f"moved robber to {self.hn_name(self.map_hexblocks[i].coords)}")
            print(f"halving all resource cards....")
            for p in self.players:
                p.half_inv()
        return val
    
    def longest_road(self, p):
        """
        Calculate the longest road for a given player.
        
        Args:
            player (Player): The player whose longest road is being calculated.
        
        Returns:
            int: The length of the longest valid road.
        """
        def is_negated(path, p, other_player_tags):
            """
            Check if a path is negated by nearby structures of other players.
            
            Args:
                path (list): The path to check.
                player (Player): The player owning the road.
                other_player_tags (list): Tags of other players.
            
            Returns:
                bool: True if the path is negated, False otherwise.
            """
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
            """
            Check if two sides are directly connected.
            
            Args:
                side1: The first side.
                side2: The second side.
            
            Returns:
                bool: True if connected, False otherwise.
            """
            if side1.parent == side2.parent:
                # Direct neighbors within the same hex
                if abs(side1.n - side2.n) in {1, len(side1.parent.sides) - 1}:
                    return True

            # Check connections via linked edges
            for i in [side1.n, (side1.n + 1) % len(side1.parent.edges)]:#, (side1.n - 1) % len(side1.parent.edges)]:
                for link in side1.parent.edges[i].links:
                    if link and (link.parent.sides[(link.n) % len(link.parent.sides)] == side2 or link.parent.sides[(link.n - 1) % len(link.parent.sides)] == side2):
                        return True

            return False

        def find_path(start_side):
            """
            Find all connected sides forming a single road.
            
            Args:
                start_side: The starting side of the path.
            
            Returns:
                list: The sides forming the path.
            """
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

            i = 1
            final_path = [path[0]]
            while i < len(path) - 1:
                if is_connected(path[i], final_path[-1]):
                    final_path.append(path[i])
                i += 1
            i2 = len(final_path) - 1
            final_final_path = [final_path[-1]]
            while i2 > -1:
                if is_connected(final_path[i2], final_final_path[-1]):
                    final_final_path.append(final_path[i2])
                i2 -= 1

            return final_final_path

        # Identify distinct paths
        paths = []
        other_player_tags = [p_.tag for p_ in self.players if p_.tag != p.tag]
        for side in p.roads:
            visited = set()
            if side not in visited and side.value == p.tag:
                paths.append(find_path(side))
                

        # Validate paths
        valid_paths = []
        for path in paths:
            for s in path:
                if s.links and s.links in path:
                    path.remove(s.links)
            if not is_negated(path, p, other_player_tags):
                valid_paths.append(path)

        """

        # Output results
        for i, path in enumerate(valid_paths):
            print(f"Path {i + 1}: {[(hn_name(p.parent.coords), f'S{p.n + 1}') for p in path]}, length: {len(path)}")"""

        return max([len(path) for path in valid_paths]) if len(valid_paths) > 0 else 0

    
    def place_struct(self, p, hn, pos, struct):
        """
        Place a structure (e.g., road, settlement) on the board for a player.
        
        Args:
            player (Player): The player placing the structure.
            hex_block (HexBlock): The hex block where the structure will be placed.
            position (int): The position within the hex block.
            structure (Structure): The type of structure being placed.
        """
        if p.max_struct_check(struct):
            if p.cost_check(struct):
                if hn.pos_is_empty(pos, struct) and hn.check_nearby(pos, struct, p, self.turn_number):
                    hn.place_struct_in_pos(pos, struct, p)
                    p.build_struct(struct)
                    if struct == Structure.SETTLEMENT:
                        p.inc_vp()
                    elif struct == Structure.ROAD:
                        longest = self.longest_road(p)
                        p.longest_road = longest
                        if longest >= self.min_longest_road:  # Minimum 4 roads required for "Longest Road"
                            if longest > self.current_longest_road:
                                if self.longest_road_owner is not None:
                                    self.longest_road_owner.dec_vp()
                                    #print(f"{self.longest_road_owner.name} lost a vp")
                                self.longest_road_owner = p
                                self.longest_road_owner.inc_vp()
                                #print(f"{self.longest_road_owner.name} gained a vp")
                                self.current_longest_road = longest
                                #print(f"Player {p.name} now has the Longest Road: {longest}")
                    return 0
                else:
                    #print("cannot place structure here")
                    return -1
            else:
                #print("cannot afford structure")
                return -2
        else:
            #print("Player has reached max limit of building this structure")
            return -3

    def give_resources(self, p, d_i=0, ignore_struct=None):
        """
        Distribute resources to a player based on the dice value rolled.
        
        Args:
            player (Player): The player to give resources to.
            dice_value (int, optional): The value of the dice rolled. Defaults to 0.
            ignore_struct (tuple, optional): Tuple containing hex and edge to ignore. Defaults to None.
        """
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
    
    def get_board_array(self):
        """
        Get the values of all hex blocks as a list.
        
        Returns:
            list: Values of all hex blocks.
        """
        return[hn.values() for hn in self.map_hexblocks]
    
    def get_hex_nums(self):
        """
        Get the numbers assigned to all hex blocks.
        
        Returns:
            list: List of all hex tile numbers.
        """
        return [hn.tile_num for hn in self.map_hexblocks]
    
    def get_hex_biomes(self):
        """
        Get the biomes assigned to all hex blocks.
        
        Returns:
             list: List of all hex biomes.
        """
        return [hn.biome for hn in self.map_hexblocks]
    
    def get_edges(self):
        """
        Get the values of all edges in the game.
        
        Returns:
            list: A list of edge values representing all edges in the game.
        """
        out = [0] * len(self.all_edges)
        for i,e in enumerate(self.all_edges):
            for ip,p in enumerate(self.players):
                if e.value == p.tag:
                    out[i] = ip + 1

        return out
    
    def get_sides(self):
        """
        Get the values of all sides in the game.
        
        Returns:
            list: A list of side values representing all sides in the game.
        """
        out = [0] * len(self.all_sides)
        for i,s in enumerate(self.all_sides):
            for ip,p in enumerate(self.players):
                if s.value == p.tag:
                    out[i] = ip + 1
        return out
    
    def get_vp(self):
        """
        Get the victory points (VP) for all players.
        
        Returns:
            list: A list of integers where each integer represents the VP of a player.
        """
        return [p.vp for p in self.players]
    
    def get_all_invs(self):
        """
        Get the inventory of all players.
        
        Returns:
            list: A list of player inventories, where each inventory is represented as a dictionary or similar structure.
        """
        return [p.inventory for p in self.players]
    
    def get_longest_road_owner(self):
        """
        Return the owner of the current longest road.
        
        Returns:
            int: The player number (1-indexed) who owns the longest road. 
                Returns 0 if no player owns the longest road.
        """
        for i,p in enumerate(self.players):
            if p == self.longest_road_owner:
                return i+1
        return 0