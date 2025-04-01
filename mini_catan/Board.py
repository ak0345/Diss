

import random
from mini_catan.enums import Biome, Structure
from mini_catan.Bank import Bank
from mini_catan.Hex import HexBlock
from mini_catan.Player import Player
from mini_catan.Die import Die

import logging
logging.basicConfig(level=logging.INFO, filename="games.log",filemode="a", format="[%(levelname)s | %(asctime)s | %(lineno)d] %(message)s")
def print(*args, **kwargs):
    logging.info(*args)

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
        self.current_player = 0

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

        #'biomes': array([2, 1, 4, 1, 0, 4, 3]
        # Uncomment below to keep map same
        #biome_distribution = [Biome.HILLS, Biome.FOREST, Biome.PASTURE, Biome.FOREST, Biome.DESERT, Biome.PASTURE, Biome.FIELDS]
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

        # 'hex_nums': array([5, 5, 1, 2, 6, 3, 5])
        # Uncomment below to keep map same
        #num_pool = [5, 5, 1, 2, 6, 3, 5]

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
        Calculate the longest road for a given player, including roads that span multiple hexes.
        
        This method builds a connectivity graph among the player's road segments (placed on hex sides)
        by checking whether the two endpoints (edges) of each road are the same or linked (across hex boundaries).
        It also makes sure that a shared endpoint is not blocked by an opponent's structure.
        
        Args:
            p (Player): The player whose road network is evaluated.
        
        Returns:
            int: The number of road segments in the longest continuous road.
        """
        from collections import defaultdict

        def same_intersection(e1, e2):
            """
            Determine if two edge components represent the same intersection.
            
            Two edges are considered the same if they are identical, or if one is linked to the other.
            """
            if e1 is e2:
                return True
            # Check if e1 links to e2
            if e1.links:
                if isinstance(e1.links, list):
                    if e2 in e1.links:
                        return True
                else:
                    if e1.links == e2:
                        return True
            # Check the reverse
            if e2.links:
                if isinstance(e2.links, list):
                    if e1 in e2.links:
                        return True
                else:
                    if e2.links == e1:
                        return True
            return False

        def get_endpoints(road):
            """
            Get the two endpoints (edge components) for a road segment.
            For a road on a hex side, the endpoints are the two adjacent edges.
            """
            parent_hex = road.get_parent()
            i = road.n
            return (parent_hex.edges[i], parent_hex.edges[(i + 1) % len(parent_hex.edges)])

        def endpoints_connected(eps1, eps2):
            """
            Return True if any endpoint in eps1 is connected (or identical) to any endpoint in eps2.
            """
            for e1 in eps1:
                for e2 in eps2:
                    if same_intersection(e1, e2):
                        return True
            return False

        def is_blocked(endpoints):
            """
            An endpoint is blocked if it has a structure placed by an opponent.
            """
            for ep in endpoints:
                if ep.value is not None and ep.value != p.tag:
                    return True
            return False

        # Gather all road segments (sides) that belong to the player.
        road_segments = [road for road in p.roads if road.value == p.tag]

        # Build the connectivity graph.
        graph = defaultdict(list)
        for road in road_segments:
            graph[road] = []

        # Connect roads if they share a common (and unblocked) intersection.
        for i, r1 in enumerate(road_segments):
            eps1 = get_endpoints(r1)
            for r2 in road_segments[i+1:]:
                eps2 = get_endpoints(r2)
                if endpoints_connected(eps1, eps2) and not (is_blocked(eps1) or is_blocked(eps2)):
                    graph[r1].append(r2)
                    graph[r2].append(r1)

        # Also incorporate direct road connections via the .links attribute.
        for r in road_segments:
            links = r.links
            if links:
                if not isinstance(links, list):
                    links = [links]
                for linked in links:
                    if linked and linked.value == p.tag:
                        if linked not in graph[r]:
                            graph[r].append(linked)
                        if r not in graph[linked]:
                            graph[linked].append(r)

        # Depth-first search to find the longest path (in terms of road segments).
        longest = 0

        def dfs(node, visited, length):
            nonlocal longest
            longest = max(longest, length)
            for neighbor in graph[node]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    dfs(neighbor, visited, length + 1)
                    visited.remove(neighbor)

        for road in road_segments:
            dfs(road, {road}, 0)

        return longest
    
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
                                p.longest_road_history.append(longest)
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
                    else:
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