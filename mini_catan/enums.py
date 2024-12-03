from enum import Enum

class Resource(Enum):
    """
    An enumeration representing different types of resources.
    Each resource is associated with a unique integer value.
    """
    WOOD = 0
    BRICK = 1
    SHEEP = 2
    WHEAT = 3

class Biome(Enum):
    """
    An enumeration representing different biomes, each associated with a resource.
    DESERT is a special case with no associated resource.
    """
    FOREST = Resource.WOOD
    HILLS = Resource.BRICK
    FIELDS = Resource.WHEAT
    PASTURE = Resource.SHEEP
    DESERT = None

class Structure(Enum):
    """
    An enumeration representing different structures and their resource costs.
    
    Attributes:
        ROAD: Requires [1 WOOD, 1 BRICK, 0 SHEEP, 0 WHEAT].
        SETTLEMENT: Requires [1 WOOD, 1 BRICK, 1 SHEEP, 1 WHEAT].
    """
    ROAD = [1, 1, 0, 0]
    SETTLEMENT = [1, 1, 1, 1]

class HexCompEnum(Enum):
    """
    An enumeration for the components of a hex, representing sides and edges.
    
    Attributes:
        S1-S6: Represent the six sides of a hex.
        E1-E6: Represent the six edges of a hex.
    """
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
        """
        Allows circular addition within the enumeration.
        
        Args:
            n (int): The number to add to the current enum's value.
        
        Returns:
            HexCompEnum: The resulting enum member after circular addition.
        """
        members = list(self.__class__)
        new_index = (self.value + n) % len(members)
        return members[new_index]