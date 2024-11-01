from enum import Enum

class Resource(Enum):
    WOOD = 'wood'
    BRICK = 'brick'
    SHEEP = 'sheep'
    WHEAT = 'wheat'

class Biome(Enum):
    FOREST = Resource.WOOD
    HILLS = Resource.BRICK
    FIELDS = Resource.WHEAT
    PASTURE = Resource.SHEEP
    DESERT = None

class Structure(Enum):
    ROAD = [1, 1, 0, 0]
    SETTLEMENT = [1, 1, 1, 1]

class HexEnum(Enum):
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