
from enums import Structure


class HexBlock:
    """
    Represents a hexagonal block on the board, which includes sides and edges
    and can be associated with neighbors, resources, and structures.
    """
    class HexComponent:
        """
        Represents a component of the hex (side or edge), which can store links,
        value (player tag), and structure type (road or settlement).
        """
        def __init__(self, parent, n):
            """
            Initialize a HexComponent.

            Args:
                parent (HexBlock): The parent hex block to which this component belongs.
                n (int): The index of this component within its parent hex.
            """
            self.parent = parent
            self.n = n
            self.links = None
            self.value = None
            self.struct = None
        
        def set_links(self, links):
            """Set the links for this hex component."""
            self.links = links

        def get_parent(self):
            """Return the parent hex block."""
            return self.parent
        
        def set_value(self, value):
            """Set the value (player tag) for this hex component."""
            self.value = value

        def set_struct(self, struct):
            """Set the structure type for this hex component."""
            self.struct = struct


    def __init__(self, x, y):
        """
        Initialize a HexBlock.

        Args:
            x (int): The x-coordinate of the hex block.
            y (int): The y-coordinate of the hex block.
        """
        self.coords = (x, y)
        # Biome and tile num
        self.biome = None
        self.tile_num = None

        self.edges = [self.HexComponent(self, i) for i in range(self.edges)]
        self.sides = [self.HexComponent(self, i) for i in range(self.sides)]
        
    def set_sides_edges(self, n1, n2, n3, n4, n5, n6):
        """
        Set neighbors for this hex and establish links between sides and edges.

        Args:
            n1-n6 (HexBlock): Neighboring hex blocks (can be None if no neighbor).
        """

        # Neighbours
        self.n1 = n1
        self.n2 = n2
        self.n3 = n3
        self.n4 = n4
        self.n5 = n5
        self.n6 = n6

        # Set links for sides
        self.sides[0].set_links(self.n1.sides[3] if n1 else None) #s4
        self.sides[1].set_links(self.n2.sides[4] if n2 else None) #s5
        self.sides[2].set_links(self.n3.sides[5] if n3 else None) #s6
        self.sides[3].set_links(self.n4.sides[0] if n4 else None) #s1
        self.sides[4].set_links(self.n5.sides[1] if n5 else None) #s2
        self.sides[5].set_links(self.n6.sides[2] if n6 else None) #s3

        # Set links for edges
        self.edges[0].set_links([self.n6.edges[2] if n6 else None, self.n1.edges[4] if n1 else None]) # e3 e5
        self.edges[1].set_links([self.n1.edges[3] if n1 else None, self.n2.edges[5] if n2 else None]) # e4 e6
        self.edges[2].set_links([self.n2.edges[4] if n2 else None, self.n3.edges[0] if n3 else None]) # e5 e1
        self.edges[3].set_links([self.n3.edges[5] if n3 else None, self.n4.edges[1] if n4 else None]) # e6 e2
        self.edges[4].set_links([self.n4.edges[0] if n4 else None, self.n5.edges[2] if n5 else None]) # e1 e3
        self.edges[5].set_links([self.n5.edges[1] if n5 else None, self.n6.edges[3] if n6 else None]) # e2 e4


    def set_biome(self, biome):
        """Set the biome type for this hex block."""
        self.biome = biome

    def set_tile_num(self, tile_num):
        """Set the tile number for this hex block."""
        self.tile_num = tile_num
    
    def pos_is_empty(self, pos, struct):
        """
        Check if a given position on the hex is empty for a specific structure.

        Args:
            pos (HexCompEnum): The position to check.
            struct (Structure): The structure type to check (ROAD or SETTLEMENT).

        Returns:
            bool: True if the position is empty, False otherwise.
        """
        if struct == Structure.ROAD:
            return self.sides[pos.value].value is None
        elif struct == Structure.SETTLEMENT:
            return self.edges[pos.value % len(self.edges)].value is None
    
    def check_empty_edges(self, i, d):
        """
        Check if edges within distance `d` of index `i` are empty.

        Args:
            i (int): The index of the edge.
            d (int): The distance to check.

        Returns:
            bool: True if all edges within distance `d` are empty, False otherwise.
        """
        flag = True
        while d > 0:
            flag = flag and (self.edges[(i - d) % len(self.edges)].value is None) and (self.edges[(i + d) % len(self.edges)].value is None)
            if flag == False:
                return False
            d -= 1
        return True
    
    def check_sides(self, i, p):
        """
        Check if the player's tag is on adjacent sides to index `i`.

        Args:
            i (int): The index of the side.
            p: The player whose tag to check.

        Returns:
            bool: True if the player's tag is present, False otherwise.
        """
        return (self.sides[i % len(self.sides)].value == p.tag) or (self.sides[(i+1) % len(self.sides)].value == p.tag)
    
    def check_road_next_to_settlement_in_first_turn(self, road_pos, set_coords):
        """
        Check if a road being placed during the first turn is adjacent to a settlement.

        Args:
            road_pos (HexCompEnum): The position of the road to check.
            set_coords (tuple): A tuple containing the hex block and side position of the settlement.

        Returns:
            bool: True if the road is adjacent to the settlement, False otherwise.
        """

        # set_coords = (hex obj, side pos enum)
        settlement = set_coords[0].edges[set_coords[1].value % len(self.edges)]
        settlement_links = settlement.links
        
        #road   road_end1 -> |-------| <- road_end2
        road_end1 = self.edges[road_pos.value % len(self.edges)]
        road_end2 = self.edges[(road_pos.value + 1) % len(self.edges)]

        if settlement == road_end1 or settlement == road_end2:
            return True
        for link in settlement_links:
            if link:
                if link == road_end1 or link == road_end2:
                    return True
        return False


    def check_nearby(self, pos, struct, p, turn_number):
        """
        Check if placing a structure is valid based on proximity and adjacency rules.

        Args:
            pos (HexCompEnum): The position to place the structure.
            struct (Structure): The type of structure (ROAD or SETTLEMENT).
            p: The player attempting to place the structure.
            turn_number (int): The current turn number (affects settlement placement rules).

        Returns:
            bool: True if the position is valid for the structure, False otherwise.
        """

        if struct == Structure.ROAD:
            i = pos.value

            other_set_check = ( self.edges[i % len(self.edges)].value is not None and self.edges[i % len(self.edges)].value != p.tag or 
                    self.edges[(i + 1)% len(self.edges)].value is not None and self.edges[(i + 1) % len(self.edges)].value != p.tag )
            
            if other_set_check:
                return False
            
            # Check if any of the adjacent edges are occupied by the player's tag
            neighbor = False
            for i_check in [(i-1), i, (i+1) % len(self.sides)]:
                if self.sides[i_check].links:
                    link = self.sides[i_check].links
                    i_n = link.n
                    neighbor = neighbor or (link.parent.edges[i_n % len(self.edges)].value == p.tag or 
                        link.parent.edges[(i_n + 1) % len(self.edges)].value == p.tag or
                        link.parent.sides[(i_n - 1) % len(self.sides)].value is not None or
                        link.parent.sides[(i_n + 1) % len(self.sides)].value is not None)
                
            return (self.edges[i % len(self.edges)].value == p.tag or 
                    self.edges[(i + 1) % len(self.edges)].value == p.tag or 
                    self.sides[(i - 1) % len(self.sides)].value is not None or 
                    self.sides[(i + 1) % len(self.sides)].value is not None or neighbor)
        
        elif struct == Structure.SETTLEMENT:
            i = pos.value % len(self.edges)
            # Check adjacent sides
            if turn_number > 1:
                side_check = self.check_sides(i, p)
            else:
                side_check = True

            # Check edge spacing (distance rule)
            edge_check = self.check_empty_edges(i, 1)

            # Additionally, check linked edges through neighbors
            links = self.edges[i].links
            if links:
                for link in links:
                    if link:
                        # Check neighboring hexes’ sides and edges
                        if turn_number > 1:
                            side_check = side_check or link.get_parent().check_sides(link.n, p)
                        edge_check = edge_check and link.get_parent().check_empty_edges(link.n, 1)
            
            return side_check and edge_check


    def place_struct_in_pos(self, pos, struct, p):
        """
        Place a structure at a given position for a player.

        Args:
            pos (HexCompEnum): The position to place the structure.
            struct (Structure): The type of structure to place.
            p: The player placing the structure.
        """
        if struct == Structure.ROAD:
            i = pos.value
            self.sides[i].set_value(p.tag)
            self.sides[i].set_struct(struct)
            p.roads.append(self.sides[i])

            side_link = self.sides[i].links
            if side_link:
                side_link.get_parent().sides[side_link.n].set_value(p.tag)
                side_link.get_parent().sides[side_link.n].set_struct(struct)
                p.roads.append(side_link)

        elif struct == Structure.SETTLEMENT:
            i = pos.value % len(self.edges)

            self.edges[i].set_value(p.tag)
            self.edges[i].set_struct(struct)
            p.settlements.append(self.edges[i])

            edge_link = self.edges[i].links
            if edge_link:
                for link in edge_link:
                    if link:
                        link.get_parent().edges[link.n].set_value(p.tag)
                        link.get_parent().edges[link.n].set_struct(struct)
                        p.settlements.append(link)
            

    def values(self):
        """
        Get the current state of the sides and edges.

        Returns:
            tuple: A tuple of lists containing structure names and values for sides and edges.
        """
        new_e = list(map(lambda x: (x.struct.name if x.struct else None, x.value), self.sides))
        new_s = list(map(lambda x: (x.struct.name if x.struct else None, x.value), self.edges))
        return (new_e, new_s)