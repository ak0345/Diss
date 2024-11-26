from enum import Enum

class Resource(Enum):
    WOOD = 0
    BRICK = 1
    SHEEP = 2
    WHEAT = 3

class Biome(Enum):
    FOREST = Resource.WOOD
    HILLS = Resource.BRICK
    FIELDS = Resource.WHEAT
    PASTURE = Resource.SHEEP
    DESERT = None

class Structure(Enum):
    ROAD = [1, 1, 0, 0]
    SETTLEMENT = [1, 1, 1, 1]

class HexCompEnum(Enum):
    S1 = 0
    S2 = 1
    S3 = 2
    S4 = 3
    S5 = 4
    S6 = 5
    E1 = 6
    E2 = 7
    E3 = 8
    E4 = 9
    E5 = 10
    E6 = 11
    def __add__(self, n):
        members = list(self.__class__)
        new_index = (self.value + n) % len(members)
        return members[new_index]