import random

class Die:
    """
    A class representing a single die with a configurable number of sides.
    """
    def __init__(self, sides=6):
        """
        Initialize the die with a specified number of sides.
        
        Args:
            sides (int): The number of sides on the die. Defaults to 6.
        """
        self.sides = sides
        self.value = None

    def roll(self):
        """
        Roll the die and generate a random value between 1 and the number of sides.
        Can use other random functions as well or weighted random functions
        
        Returns:
            int: The result of the die roll.
        """
        self.value = random.randint(1, self.sides)
        return self.value

    def get_value(self):
        """
        Retrieve the current value of the die. If the die hasn't been rolled, returns None.
        
        Returns:
            int or None: The current value of the die, or None if it hasn't been rolled.
        """
        return self.value