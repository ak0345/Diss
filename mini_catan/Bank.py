class Bank:
    def __init__(self, resource_count):
        self.name = "Bank"
        types_of_resources = 4
        self.inventory = [resource_count] * types_of_resources

    def add_2_inv(self, items):
        return [a + b for a, b in zip(self.inventory, items)]

    def del_from_inv(self, items):
        return [a - b for a, b in zip(self.inventory, items)]