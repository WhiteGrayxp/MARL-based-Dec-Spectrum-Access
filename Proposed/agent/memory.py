import numpy as np



class ReplayBuffer:
    def __init__(self, num_agent = 4, episode_length = 100, obs_size=16, capacity=10000):
        """ Concurrent experience replay trajectory, three axes: episode, agent, and time step
            trace_length corrsponds to the length of captured segment
        """
        self.num_agent = num_agent
        self.episode_length = episode_length
        self.obs_size = obs_size
        self.capacity = capacity
        
        self.actions = np.empty((self.capacity, self.episode_length, self.num_agent), dtype = np.uint8)
        self.rewards = np.empty((self.capacity, self.episode_length, self.num_agent), dtype = np.float32)
        self.obs = np.empty((self.capacity, self.episode_length, self.num_agent, self.obs_size), dtype = np.float32)
        self.next_obs = np.empty((self.capacity, self.episode_length, self.num_agent, self.obs_size), dtype = np.float32)


        self.count = 0
        self.index = 0
        

    def store(self, obs, action, reward, next_obs):
        """ store an episode of experiences

        """
        assert action.shape == (self.episode_length, self.num_agent)
        assert obs.shape == (self.episode_length, self.num_agent, self.obs_size)

        self.actions[self.index] = action
        self.rewards[self.index] = reward
        self.obs[self.index] = obs
        self.next_obs[self.index] = next_obs

        self.count = max(self.count, self.index + 1)
        self.index = (self.index + 1) % self.capacity
        
   
           
    def sample(self, batch_size, trace_length):
        """ sample a mini-batch of experience trajectories

        Args:
            batch_size ([type]): [description]
            trace_length ([type]): [description]

        Returns:
            [type]: [description]
        """
        if self.count < batch_size:
            return
        else:
            episode_indexes = np.random.choice(range(0,self.count), batch_size, replace=False)
            start_time_steps = np.random.choice(range(0, self.episode_length - trace_length), batch_size)

        actions = np.empty((batch_size, trace_length, self.num_agent), dtype = np.uint8)
        rewards = np.empty((batch_size, trace_length, self.num_agent), dtype = np.float32)
        obs = np.empty((batch_size, trace_length, self.num_agent, self.obs_size), dtype = np.float32)
        next_obs = np.empty((batch_size, trace_length, self.num_agent, self.obs_size), dtype = np.float32)

        i = 0
        for episode_index, start_time_step in zip(episode_indexes, start_time_steps):
            actions[i] = self.actions[episode_index, start_time_step:start_time_step+trace_length, :]
            rewards[i] = self.rewards[episode_index, start_time_step:start_time_step+trace_length, :]
            obs[i] = self.obs[episode_index, start_time_step:start_time_step+trace_length, :, :]
            next_obs[i] = self.next_obs[episode_index, start_time_step:start_time_step+trace_length, :, :]
            i += 1

        assert i == batch_size

        return obs, actions, rewards, next_obs