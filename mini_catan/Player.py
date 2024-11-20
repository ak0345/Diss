import random

class Player:
    def __init__(self, name):
        self.vp = 0
        self.longest_road = 0
        types_of_resources = 4
        self.name = name
        self.tag = name+str(random.randint(1, 6))
        #inventory = number of [Wood, Brick, Sheep, Wheat]
        self.inventory = [0] * types_of_resources

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

    def del_from_inv(self, items):
        self.inventory = [a - b for a, b in zip(self.inventory, items)]

    def trade_I_with_p(self, p, my_items, p_items):
        #items = [W_n, B_n, S_n, W_n]
        p.del_from_inv(p_items)
        p.add_2_inv(my_items)

        self.del_from_inv(my_items)
        self.add_2_inv(p_items)

    def inc_vp(self):
        self.vp += 1

