class Bank:
    def __init__(self, resource_count):
        """
        Initialize the bank with a given amount of resources.
        
        Args:
            resource_count (int): The initial count for each type of resource.
        """
        self.name = "bank"
        types_of_resources = 4
        self.inventory = [resource_count] * types_of_resources

    def add_2_inv(self, items):
        """
        Add resources to the bank's inventory.
        
        Args:
            items (list[int]): A list of integers representing the quantities of resources to add.

        Returns:
            list[int]: Updated inventory after adding the resources.
        """
        return [a + b for a, b in zip(self.inventory, items)]

    def del_from_inv(self, items):
        """
        Remove resources from the bank's inventory.
        
        Args:
            items (list[int]): A list of integers representing the quantities of resources to remove.

        Returns:
            list[int]: Updated inventory after removing the resources.
        """
        return [a - b for a, b in zip(self.inventory, items)]