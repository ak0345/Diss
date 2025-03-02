import numpy as np
import random
from gymnasium import spaces

class RandomAgent:
    def __init__(self, seed=42):
        print("I am a random agent")
        
        np.random.seed(seed)

        self.action_space = spaces.Discrete(5)
        self.action_space_curr = None
        
        self.build_settlement_action_space = spaces.Discrete(24)

        self.build_road_action_space = spaces.Discrete(30)

        self.player_trade_action_space = spaces.Box(low=0, high=3, shape=(2, 4), dtype=np.int32)

        self.player_trade_offer_request_action_space = spaces.Discrete(3) # 0: Yes, 1: No, 2: Counter

        self.counter_offer_action_space = spaces.Box(low=0, high=3, shape=(2, 4), dtype=np.int32)

        self.counter_offer_response_action_space = spaces.Discrete(3) # 0: Yes, 1: No, 2: Counter the Counter

        self.counter_counter_offer_action_space = spaces.Box(low=0, high=3, shape=(2, 4), dtype=np.int32)

        self.counter_counter_offer_reply_action_space =  spaces.Discrete(2) # 0: Yes, 1: No

        self.bank_trade_action_space = spaces.Box(low=0, high=3, shape=(2, 4), dtype=np.int32)

        self.attempts = 0
    
    def act(self, observation):
        """
        Select an action based on the current observation and turn number.
        
        The observation is expected to be a dictionary that contains trade state flags.
        Depending on the trade sub-state, a corresponding action space is used.
        If no trade is in progress, the agent falls back to its base action selection.
        """
        # Check if we are in a trade sequence:
        if observation[9] == 1: # p_trade_followup_1
            # In the initial trade follow-up state:
            if observation[12] == 1: # reply_to_offer
                # Environment expects a reply to the trade offer (accept/reject/counter)
                return self.player_trade_offer_request_action_space.sample()
            elif observation[13] == 1: # counter_sent
                # We already sent a counter offer; now pick a counter offer action
                return self.counter_offer_action_space.sample()
            else:
                # Otherwise, initiate the trade offer
                return self.player_trade_action_space.sample()
        
        elif observation[10] == 1: # p_trade_followup_2
            # In the follow-up state after a counter offer has been sent,
            # respond with Yes/No/Counter (e.g., accept, reject, or counter the counter)
            return self.counter_offer_response_action_space.sample()
        
        elif observation[11] == 1: # p_trade_followup_3
            # In the final trade follow-up stage:
            if observation[12] == 1: # reply_to_offer
                # Respond to a counter-counter offer
                return self.counter_counter_offer_reply_action_space.sample()
            else:
                # Otherwise, send a counter-counter offer
                return self.counter_counter_offer_action_space.sample()
        
        # If not in any trade follow-up stage, follow the base action selection:
        if observation[8] > 0: # turn_number
            if self.action_space_curr is None:
                self.action_space_curr = self.action_space.sample()
                # If the agent decides to end its turn (action 4), reset for next turn.
                if self.action_space_curr == 4:  # End Turn
                    temp = self.action_space_curr
                    self.action_space_curr = None
                    return temp
                return self.action_space_curr
            
            elif self.action_space_curr == 0:
                # Build road action
                if self.attempts < 10:
                    self.attempts += 1
                else:
                    self.attempts = 0
                    self.action_space_curr = None
                    return -1  # Indicates failure after many attempts
                return self.build_road_action_space.sample()
            
            elif self.action_space_curr == 1:
                # Build settlement action
                if self.attempts < 10:
                    self.attempts += 1
                else:
                    self.attempts = 0
                    self.action_space_curr = None
                    return -1
                return self.build_settlement_action_space.sample()
            
            elif self.action_space_curr == 2:
                # Initiate player trade
                if self.attempts < 10:
                    self.attempts += 1
                else:
                    self.attempts = 0
                    self.action_space_curr = None
                    return -1
                return self.player_trade_action_space.sample()
            
            elif self.action_space_curr == 3:
                # Bank trade action
                if self.attempts < 10:
                    self.attempts += 1
                else:
                    self.attempts = 0
                    self.action_space_curr = None
                    return -1
                return self.bank_trade_action_space.sample()
        
        else:
            # On turn 0, simply try a road build action.
            return self.build_road_action_space.sample()