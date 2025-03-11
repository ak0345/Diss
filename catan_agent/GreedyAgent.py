import numpy as np

class GreedyAgent:
    """
    A greedy agent for MiniCatan.
    """
    
    def __init__(self, player_index=0):
        self.attempts = 0
        self.attempt_threshold = 5  # After 5 failed attempts, cancel the move.
        self.cancel_next = False    # Flag to return a safe action on the next call.
        self.player_index = player_index  # 0 or 1
        self.action = -1

    def _get_inventory(self, obs):
        # For 2 players (each with 4 resources): indices 0-3 for player 0, 4-7 for player 1.
        start = self.player_index * 4
        return obs[start:start+4]
    
    def _get_other_inventory(self, obs):
        # Returns the other player's inventory.
        obs = np.array(obs)
        if self.player_index == 0:
            return obs[4:8]
        else:
            return obs[0:4]
    
    def _generate_valid_trade_action(self, inventory, other_inv=None):
        """
        Generates a trade action (a 2x4 array) for bank or player trades.
        
        For the offered resources, it ensures that the agent doesn't offer more than it can afford.
        For the requested resources (if other_inv is provided, i.e. for player trades), it ensures
        that the request does not exceed what the other player currently has.
        
        For each resource, we assume that the cost for the offered amount is twice its value.
        """
        # Check if the agent can afford at least one resource.
        if not np.any(inventory >= 2):
            return -1  # Cancel move if no resource is affordable.
        
        offer = np.zeros(4, dtype=np.int32)
        for i in range(4):
            # Maximum that can be offered is inventory[i] // 2.
            max_offer = (inventory[i] // 2) if inventory[i] >= 2 else 0
            if max_offer > 0:
                # Start at 1 so that 0 is never chosen.
                offer[i] = np.random.randint(1, min(10, max_offer + 1))
            else:
                offer[i] = 0
        
        # Ensure that at least one resource is offered.
        if not np.any(offer > 0):
            for i in range(4):
                if inventory[i] >= 2:
                    offer[i] = 1
                    break

        # For the request, if other_inv is provided (i.e. for a player trade),
        # limit each requested amount by the other player's available resources.
        if other_inv is not None:
            if not np.any(other_inv > 0):
                return -1  # Cancel move if opponent has nothing to trade.
            request = np.zeros(4, dtype=np.int32)
            for i in range(4):
                max_req = other_inv[i]
                if max_req > 0:
                    request[i] = np.random.randint(1, min(10, max_req + 1))
                else:
                    request[i] = 0
            if not np.any(request > 0):
                for i in range(4):
                    if other_inv[i] > 0:
                        request[i] = 1
                        break
        else:
            request = np.random.randint(0, 10, size=4, dtype=np.int32)
        
        return np.stack([offer, request])
    
    def act(self, obs):
        # Ensure obs is a numpy array.
        obs = np.array(obs)
        
        # If a cancellation was signaled from the previous attempt, now return a safe action.
        if self.cancel_next:
            self.cancel_next = False
            self.attempts = 0
            self.action = -1
            # Safe action: End Turn (action 4) in the main action space.
            return 4
        
        # Increment our attempt counter.
        self.attempts += 1
        if self.attempts >= self.attempt_threshold:
            # Cancel the move.
            self.cancel_next = True
            self.attempts = 0
            self.action = -1
            return -1
        
        # Decode key values from the observation.
        turn_number         = int(obs[80])
        b_trade_followup    = int(obs[81])
        p_trade_followup_1  = int(obs[82])
        p_trade_followup_2  = int(obs[83])
        p_trade_followup_3  = int(obs[84])
        reply_to_offer      = int(obs[85])
        #counter_sent        = int(obs[86])
        
        # Extract the agent's current inventory.
        inventory = self._get_inventory(obs)
        
        # Trade followup branches.
        if b_trade_followup:
            # Bank trade: ensure we can offer something.
            action = self._generate_valid_trade_action(inventory, other_inv=None)
            return action
        
        elif p_trade_followup_1:
            if reply_to_offer:
                # Reply to offer: action is Discrete(3).
                return np.random.randint(0, 3)
            else:
                # Player trade offer: include check on the other player's inventory.
                other_inventory = self._get_other_inventory(obs)
                action = self._generate_valid_trade_action(inventory, other_inv=other_inventory)
                return action
        
        elif p_trade_followup_2:
            # Counter offer response: action is Discrete(3).
            return np.random.randint(0, 3)
        
        elif p_trade_followup_3:
            if reply_to_offer:
                # Counter counter offer reply: action is Discrete(2).
                return np.random.randint(0, 2)
            else:
                # Player counter counter offer: include other player's inventory check.
                other_inventory = self._get_other_inventory(obs)
                action = self._generate_valid_trade_action(inventory, other_inv=other_inventory)
                return action
        
        elif self.action == 0:
            return np.random.randint(0, 24)
        
        elif self.action == 1:
            return np.random.randint(0, 30)
        
        else:
            # Not in any trade followup state.
            if turn_number == 0:
                # During initial build phase, randomly choose between a settlement action (Discrete(24))
                # and a road action (Discrete(30)). We assume initial builds are free.
                if np.random.rand() < 0.5:
                    return np.random.randint(0, 24)
                else:
                    return np.random.randint(0, 30)
            else:
                # Normal turn: sample from the main action space (Discrete(5)).
                self.action = np.random.randint(0, 5)
                # For building actions, check that we have enough resources.
                # Assume a road costs [1, 1, 0, 0] (Wood, Brick) and a settlement costs [1, 1, 1, 1].
                if self.action == 0:
                    # Build Road.
                    if inventory[0] < 1 or inventory[1] < 1:
                        return -1  # Cancel move.
                elif self.action == 1:
                    # Build Settlement.
                    if np.any(inventory < np.array([1, 1, 1, 1])):
                        return -1  # Cancel move.
                return self.action
