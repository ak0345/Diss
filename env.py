import mini_catan
import gymnasium as gym
import numpy as np
from random import randint

a = gym.make("MiniCatanEnv-v0")

a.reset()
while True:
    try:
        """
        #print("------------------------------------------------")
        b = randint(0,30)
        print("Action:", b)
        print(a.step(b))
        #print("------------------------------------------------")
        """
        a.reset()
        print("------------------------------------------------")
        print(a.step(2))
        print("------------------------------------------------")
        print(a.step(2))
        print("------------------------------------------------")
        print(a.step(9))
        print("------------------------------------------------")
        print(a.step(9))
        print("------------------------------------------------")
        print(a.step(22))
        print("------------------------------------------------")
        print(a.step(28))
        print("------------------------------------------------")
        print(a.step(15))
        print("------------------------------------------------")
        print(a.step(19))
        print("------------------------------------------------")
        print(a.step(3))
        print("------------------------------------------------")
        print(a.step(np.array([[0,1,0,0],[1,1,0,0]])))
        print("------------------------------------------------")
        print(a.step(0))
        print("------------------------------------------------")
        print(a.step(1))
        print("------------------------------------------------")
        print(a.step(3))
        print("------------------------------------------------")
        break
    except AssertionError as e:
        print(e)
        print("trying again teehee...")
        #break
        pass