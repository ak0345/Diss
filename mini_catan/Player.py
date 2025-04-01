import random
import numpy as np
from mini_catan.enums import Structure

class Player:
    def __init__(self, name):
        """
        Initialize a new player with a name, a unique tag, and an inventory of resources.

        Args:
            name (str): The name of the player.
        """
        self.vp = 0 # Victory points
        self.longest_road = 0 
        #types_of_resources = 4
        self.name = name
        self.tag = name+str(random.randint(1, 6)) # Generate Unique tag for player
        # Inventory = number of [Wood, Brick, Sheep, Wheat]
        self.inventory = [4, 4, 2, 2] # [0] * types_of_resources
        self.settlements = [] # List of settlements owned by the player
        self.roads = [] # List of roads owned by the player
        self.trades_rejected = 0
        self.total_trades = 0
        self.second_settlement = None
        self.first_settlement = None
        self.longest_road_history = []

    def get_player_s2r(self):
        count = 0
        for s in self.settlements:
            road_count = 0
            for r in self.roads:
                if (r.get_parent().edges[r.n % len(r.get_parent().edges)] == s) or (r.get_parent().edges[(r.n + 1) % len(r.get_parent().edges)] == s):
                    road_count += 1
                if  r.links:
                    if (r.links.get_parent().edges[r.n % len(r.links.get_parent().edges)] == s) or (r.links.get_parent().edges[(r.n + 1) % len(r.links.get_parent().edges)] == s):
                        road_count += 1
                if road_count >= 2:
                    count += 1
                    break
        return count
        

    def cost_check(self, struct):
        """
        Check if the player has enough resources to build a given structure.

        Args:
            struct (Structure): The structure to be checked.

        Returns:
            bool: True if the player has enough resources, False otherwise.
        """
        check = [a - b for a, b in zip(self.inventory, struct.value)]
        for c in check:
            if c < 0:
                return False
        
        return True

    def build_struct(self, struct):
        """
        Deduct the resources required to build a structure from the player's inventory.

        Args:
            struct (Structure): The structure to be built.
        """
        self.del_from_inv(struct.value)

    def add_2_inv(self, items):
        """
        Add specified resources to the player's inventory.

        Args:
            items (list): The resources to be added.
        """
        self.inventory = [a + b for a, b in zip(self.inventory, items)]
    
    def half_inv(self):
        """
        Halve the resources in the player's inventory if all of them are >1 (used for certain game rules).
        """
        for i, a in enumerate(self.inventory):
            if a>1:
                self.inventory[i] = a//2

    def del_from_inv(self, items):
        """
        Remove specified resources from the player's inventory.

        Args:
            items (list): The resources to be removed.
        """
        self.inventory = [a - b for a, b in zip(self.inventory, items)]

    def trade_cost_check(self, p, my_items, p_items):
        return (
        np.all(np.array(my_items) <= np.array(self.inventory)) and 
        np.all(np.array(p_items) <= np.array(p.inventory))
    )

    def trade_I_with_p(self, p, my_items, p_items):
        """
        Perform a trade with another player.

        Args:
            p (Player): The other player involved in the trade.
            my_items (list): The resources the player is offering.
            p_items (list): The resources the player wants in return.

        Returns:
            bool: True if the trade is successful, False otherwise.
        """
        # Check if both parties have the necessary items for the trade
        if self.trade_cost_check(p, my_items, p_items):
            # Perform the trade
            p.del_from_inv(p_items)
            p.add_2_inv(my_items)

            self.del_from_inv(my_items)
            self.add_2_inv(p_items)
            return True  # Trade successful
        else:
            return False
        
    def trade_I_with_b(self, my_items, b_items):
        
        """
        Perform a bank trade at a 2:1 ratio.
        
        The player must offer resources from exactly one type and receive resources
        of exactly one (different) type. The total offered amount must be exactly twice 
        the total requested amount. For example, to receive 1 unit of a resource, the player 
        must offer 2 units of another resource.
        
        Args:
            my_items (list): A list (length 4) representing the quantity of each resource offered.
                            For instance, [2, 0, 0, 0] means offering 2 wood.
            b_items (list): A list (length 4) representing the quantity of each resource requested.
                            For instance, [0, 1, 0, 0] means requesting 1 brick.
        
        Returns:
            bool: True if the trade is successful, False otherwise.
        """
        # Ensure that exactly one resource type is being offered.
        offered_types = [i for i, x in enumerate(my_items) if x > 0]
        if len(offered_types) != 1:
            return False

        # Ensure that exactly one resource type is being requested.
        requested_types = [i for i, x in enumerate(b_items) if x > 0]
        if len(requested_types) != 1:
            return False

        # Optionally, ensure that the offered and requested resource types are not the same.
        if offered_types[0] == requested_types[0]:
            return False

        offered_amount = sum(my_items)
        requested_amount = sum(b_items)

        # Enforce the 2:1 ratio.
        if offered_amount != 2 * requested_amount:
            return False

        # Check if the player has enough of the offered resource.
        if not all(my <= inv for my, inv in zip(my_items, self.inventory)):
            return False

        # Perform the trade:
        self.del_from_inv(my_items)
        self.add_2_inv(b_items)
        return True

        
    def max_struct_check(self, struct):
        """
        Check if the player can build more of a specific structure based on game rules.

        Args:
            struct (Structure): The structure to be checked.

        Returns:
            bool: True if the player can build more of the structure, False otherwise.
        """
        if struct == Structure.SETTLEMENT:
            if len(self.settlements) < 6:
                return True
        if struct == Structure.ROAD:
            if len(self.roads) < 11:
                return True
        return False


    def inc_vp(self):
        """
        Increment the player's victory points by 1.
        """
        self.vp += 1

    def dec_vp(self):
        """
        Decrement the player's victory points by 1.
        """
        self.vp -= 1

