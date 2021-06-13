# refer to https://github.com/lyu-xg/dec-hdrqn 

import random
import numpy as np
from collections import deque



# class ReplayBuffer:
#     def __init__(self, capacity=1e4) -> None:
#         self.capacity = capacity
#         self.index = 0
#         self.size = 0
#         self.buffer = deque()

#     def store(self, experience):
#         """ 
#         Store the experience 
        
        
#         """
#         if self.size < self.capacity:
#             self.buffer.append(experience)
#             self.size += 1
#         else:
#             self.buffer.popleft()
#             self.buffer.append(experience)


#     def sample(self, batch_size):
#         """ Sample a batch of experiences """
#         batch = []
#         if self.size < batch_size:
#             batch = random.sample(self.buffer, self.size)
#         else:
#             batch = random.sample(self.buffer, batch_size)
#         return batch


#     def reset(self):
#         self.buffer.clear()
#         self.size = 0
#         self.index = 0
#         pass


class ReplayBuffer:
    def __init__(self, obs_size, capacity=200000):
        self.obs_size = obs_size
        self.capacity = capacity
        
        self.actions = np.empty(self.capacity, dtype = np.uint8)
        self.rewards = np.empty(self.capacity, dtype = np.float64)
        self.obs = np.empty((self.capacity, self.obs_size), dtype = np.float16)
        self.next_obs = np.empty((self.capacity, self.obs_size), dtype = np.float16)


        self.count = 0
        self.index = 0
        

    def store(self, obs, action, reward, next_obs):
        self.actions[self.index] = action
        self.rewards[self.index] = reward
        self.obs[self.index] = obs
        self.next_obs[self.index] = next_obs
        self.count = max(self.count, self.index + 1)
        self.index = (self.index + 1) % self.capacity
        
   
           
    def sample(self, batch_size):
        if self.count < batch_size:
            indexes = range(0, self.count)
        else:
            indexes = random.sample(range(0,self.count), batch_size)

        obs = self.obs[indexes]
        next_obs = self.next_obs[indexes]
        actions = self.actions[indexes]
        rewards = self.rewards[indexes]

        return obs, actions, rewards, next_obs
   

