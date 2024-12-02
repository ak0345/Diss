import random
from enums import Structure

class Player:
    def __init__(self, name):
        self.vp = 0
        self.longest_road = 0
        types_of_resources = 4
        self.name = name
        self.tag = name+str(random.randint(1, 6))
        #inventory = number of [Wood, Brick, Sheep, Wheat]
        self.inventory = [0] * types_of_resources
        self.settlements = []
        self.roads = []

    def cost_check(self, struct):
        check = [a - b for a, b in zip(self.inventory, struct.value)]
        for c in check:
            if c < 0:
                return False
        
        return True

    def build_struct(self, struct):
        self.inventory = [a - b for a, b in zip(self.inventory, struct.value)]

    def add_2_inv(self, items):
        self.inventory = [a + b for a, b in zip(self.inventory, items)]
    
    def half_inv(self):
        self.inventory = [a//2 for a in self.inventory]

    def del_from_inv(self, items):
        self.inventory = [a - b for a, b in zip(self.inventory, items)]

    def trade_I_with_p(self, p, my_items, p_items):
        # Check if both parties have the necessary items for the trade
        if all(my <= inv for my, inv in zip(my_items, self.inventory)) and \
        all(pi <= inv for pi, inv in zip(p_items, p.inventory)):
            # Perform the trade
            p.del_from_inv(p_items)
            p.add_2_inv(my_items)

            self.del_from_inv(my_items)
            self.add_2_inv(p_items)
            return True  # Trade successful
        else:
            return False
        
    def max_struct_check(self, struct):
        if struct == Structure.SETTLEMENT:
            if len(self.settlements) < 6:
                return True
        if struct == Structure.ROAD:
            if len(self.roads) < 11:
                return True
        return False


    def inc_vp(self):
        self.vp += 1

    def dec_vp(self):
        self.vp -= 1

