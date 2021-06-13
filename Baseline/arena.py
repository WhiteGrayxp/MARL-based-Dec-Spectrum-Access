import torch
import os

import numpy as np
import random
from itertools import chain
from agent.memory import ReplayBuffer
from agent.qagent import DQN, QAgent


EPISODE_LENGTH = 100        #! new message generates every 100 steps

#! use GPU ('cuda: 0') by default
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device('cpu')

class Arena:
    """
    Several Q agent interact with environment here!
    """
    def __init__(self,
                 env,
                 num_agent,
                 batch_size=32,
                 learning_rate=0.0001,
                 epsilon=1.0,
                 final_epsilon=0.05,
                 gamma=0.95,
                 anneal_period=18000,
                 training= True,
                 model_path='./models') -> None:
        """
        Initialize the simulation arena.

        """
        self.num_agent = num_agent
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.epsilon = epsilon
        self.final_epsilon = final_epsilon
        self.gamma = gamma
        self.anneal_period = anneal_period
        self.training = training
        self.model_path = os.path.join(model_path, '{}_agents'.format(self.num_agent))

        self.env = env
        self.reward = []
        self.episode_reward = []

        self.agents = self.create_agents()
        self.observations = self.env.reset(episode=0, epsilon=1.)

        if self.training == False:
            self.epsilon = self.final_epsilon

    

    def reset(self, episode):
        """
        reset the simulator and agents
        """
        self.reward = []
        self.observations = self.env.reset(episode, self.epsilon)

        

    def create_agents(self):
        """ create agents, may use parameter sharing
        when self.training = False, this function will load the saved models
        """
        input_shape = self.env.observation_space
        output_shape = self.env.action_space
        agents = []
        if self.training:
            for i in range(self.num_agent):
                model = DQN(input_shape, output_shape)
                target_model = DQN(input_shape, output_shape)
                target_model.load_state_dict(model.state_dict())
                memory = ReplayBuffer(obs_size = input_shape)

                agent = QAgent(model, target_model, memory,
                                        self.env.action_space, self.batch_size,
                                        self.learning_rate, self.gamma,
                                        self.epsilon, training=self.training)
                agents.append(agent)
                
        else:
            #! load the saved (trained) models
            for i in range(self.num_agent):
                model = DQN(input_shape, output_shape)
                target_model = None
                memory = None
                path = 'individual_model/agent_{}/model'.format(i)
                path = os.path.join(self.model_path, path)
                model.load_state_dict(torch.load(path))
                model.eval()
                agent = QAgent(model, target_model, memory,
                                        self.env.action_space, self.batch_size,
                                        self.learning_rate, self.gamma,
                                        self.epsilon, training=self.training)
                agents.append(agent)
        return agents

    def step(self, episode):
        """ 
        complete one step interaction
        
        """
        actions = []
        for i, agent in enumerate(self.agents):
            actions.append(agent.act(self.observations[i]))

        new_observations, reward = self.env.step(np.array(actions), episode, self.epsilon)
        if self.training:
            for i, agent in enumerate(self.agents):
                agent.memory.store(self.observations[i], actions[i], reward, new_observations[i])
            self.reward.append(reward)
            #! new messages generate
            if self.env.t % EPISODE_LENGTH == 0:
                self.episode_reward.append(sum(self.reward))
                self.reward = []
        self.observations = new_observations


    def train(self):
        loss = 0.
        for i, agent in enumerate(self.agents):
            obs_batch, action_batch, reward_batch, next_obs_batch = agent.memory.sample(self.batch_size)
            loss += agent.learn(obs_batch, action_batch, reward_batch, next_obs_batch)
        
        return loss / self.num_agent

#//  does't need to be implemented here
    def evaluate(self):
        raise NotImplementedError
    

    def update_epsilon(self):
        self.epsilon = max(self.final_epsilon, self.epsilon - (1 - self.final_epsilon) / self.anneal_period)

        for agent in self.agents:
            agent.update_epsilon(self.epsilon)


    def inverse_joint(self, batch_joint):
        """ transform the batch of joint experience to individual transitions

        Args:
            batch ([type]): batch of joint experience trajectory
        """
        result = [[] for _ in range(self.num_agent)]
        for obs, act, reward, next_obs in chain(*batch_joint):
            for i in range(self.num_agent):
                #! flag bit indicates this transition is a padding
                flag = float(obs is not self.memory.zero_joint_obs)
                individual_transition = [obs[i], act[i], reward, next_obs[i], flag]
                result[i].append(individual_transition)
        #! now, shape of the 'result' is (num_agent, batch_size * trace_len, 5)
        result = np.array(result)
        result = result.reshape(self.num_agent, self.memory.batch_size, self.memory.trace_len, -1)
        result = result.transpose(0, 2, 1, 3)
        #! after some transformation, now, axis_0 of result corresonds to different agents,
        #! axis_1 corresponds to diffrent time step of the single trace
        #! axis_2 corresponds to different trace of the single batch
        #! axis_3 corresponds to single step transition (s, a, r, s), individual observaiton
        return result

    def update_target_model(self):
        assert self.training == True
        for agent in self.agents:
            agent.update_target_model()

    def save_models(self):
        for i, agent in enumerate(self.agents):
            path = 'individual_model/agent_{}/model'.format(i)
            file_path = os.path.join(self.model_path,  path)
            agent.save_model(file_path)

    #// since this function has been integrated into self.create_agents(), so not needed actually
    def load_models(self):
        raise NotImplementedError

