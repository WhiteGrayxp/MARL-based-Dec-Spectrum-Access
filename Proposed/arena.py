import torch
import os

import numpy as np
import random
from agent import qagent, rnn, memory

# MAX_EPSILON = 1.0
# MIN_EPSILON = 0.05
# INIT_HYSTERETIC = 0.2
# FINAL_HYSTERETIC = 0.8
# ANNEAL_PERIOD = 18000        #! after 2000 episodes, linear annealing stops
EPISODE_LENGTH = 100        #! new message generates every 100 steps


class Arena:
    """
    Several Q agent interact with environment here!
    """
    def __init__(self,
                 env,
                 memory,
                 num_agent=4,
                 episode_length=100,
                 trace_length=20,
                 batch_size=32,
                 learning_rate=0.0001,
                 epsilon=1.0,
                 final_epsilon=0.05,
                 gamma=0.95,
                 hysteretic=0.2,
                 final_hysteretic=0.8,
                 anneal_period=18000,
                 training= True,
                 parameter_sharing=False,
                 dueling=False,
                 benchmark='roundrobin',
                 model_path='./models') -> None:
        """
        Initialize the simulation arena.

        """
        self.num_agent = num_agent
        self.batch_size = batch_size
        self.episode_length = episode_length
        self.trace_length = trace_length
        self.learning_rate = learning_rate
        self.epsilon = epsilon
        self.final_epsilon = final_epsilon
        self.gamma = gamma
        self.init_hysteretic = hysteretic
        self.hysteretic = hysteretic
        self.final_hysteretic = final_hysteretic
        self.anneal_period = anneal_period
        self.training = training
        self.parameter_sharing = parameter_sharing
        self.dueling = dueling
        assert benchmark == 'roundrobin' or benchmark == 'random'
        self.benchmark = benchmark
        self.model_path = os.path.join(model_path, '{}_agents'.format(self.num_agent), benchmark)

        self.env = env
        #! concurrent experience replay trajectory
        # new_memory.ReplayBuffer(self.num_agent, self.episode_length, self.trace_length, self.env.action_space)
        self.memory = memory
        #! buffer for the observation, action, reward
        self.episode_reward = 0.
        self.buffer_index = 0
        self.obs_buffer = np.zeros((self.episode_length, self.num_agent, self.env.observation_space))
        self.act_buffer = np.zeros((self.episode_length, self.num_agent), dtype=np.int)
        self.reward_buffer = np.zeros((self.episode_length, self.num_agent))
        self.next_obs_buffer = np.zeros((self.episode_length, self.num_agent, self.env.observation_space))


        self.agents = self.create_agents()
        self.observations = self.env.reset()

        #! each agent occupies an orthogonal subchannel with maximum powe level
        #! Round-Robin as default
        if self.benchmark == 'roundrobin':
            # 记录轮盘数
            self.benchmark_action_needed = [i for i in range(self.num_agent)]       
        elif self.benchmark == 'random':
            self.benchmark_action_needed = None


    def reset(self):
        """
        reset the simulator and agents
        """
        self.buffer_index = 0
        if self.benchmark == 'roundrobin':
            self.benchmark_action_needed = [i for i in range(self.num_agent)] 
        self.observations = self.env.reset()
        self.obs_buffer = np.zeros((self.episode_length, self.num_agent, self.env.observation_space))
        self.act_buffer = np.zeros((self.episode_length, self.num_agent), dtype=np.int)
        self.reward_buffer = np.zeros((self.episode_length, self.num_agent))
        self.next_obs_buffer = np.zeros((self.episode_length, self.num_agent, self.env.observation_space))
        #! reset the hidden state
        for agent in self.agents:
            agent.init_hidden_state()
    
        

    def create_agents(self):
        """ create agents, may use parameter sharing
        when self.training = False, this function will load the saved models
        """
        input_shape = self.env.observation_space
        output_shape = self.env.action_space
        agents = []
        if self.training:
            #! create the first unique agent
            if self.dueling:
                model = rnn.DuelingRNN(input_shape, output_shape)
                target_model = rnn.DuelingRNN(input_shape, output_shape)
            else:
                model = rnn.RNN(input_shape, output_shape)
                target_model = rnn.RNN(input_shape, output_shape)
            target_model.load_state_dict(model.state_dict())
            
            for i in range(self.num_agent):
                agent = qagent.QAgent(model, target_model,
                                        self.env.action_space, self.batch_size,
                                        self.learning_rate, self.gamma,
                                        self.epsilon, self.hysteretic,
                                        training=self.training)
                agents.append(agent)
                if not self.parameter_sharing:
                    if self.dueling:
                        model = rnn.DuelingRNN(input_shape, output_shape)
                        target_model = rnn.DuelingRNN(input_shape, output_shape)
                    else:
                        model = rnn.RNN(input_shape, output_shape)
                        target_model = rnn.RNN(input_shape, output_shape)
                    target_model.load_state_dict(model.state_dict())
        else:
            #! load the saved (trained) models
            if self.dueling:
                model = rnn.DuelingRNN(input_shape, output_shape)
            else:
                model = rnn.RNN(input_shape, output_shape)
            target_model = None
            if self.parameter_sharing:
                path = os.path.join(self.model_path, 'shared_model/model')
            else:
                path = 'individual_model/agent_0/model'
                path = os.path.join(self.model_path, path)
            model.load_state_dict(torch.load(path))
            model.eval()
            
            for i in range(self.num_agent):
                agent = qagent.QAgent(model, target_model,
                                        self.env.action_space, self.batch_size,
                                        self.learning_rate, self.gamma,
                                        self.epsilon, self.hysteretic,
                                        training=self.training)
                agents.append(agent)
                if not self.parameter_sharing and i < self.num_agent - 1:
                    if self.dueling:
                        model = rnn.DuelingRNN(input_shape, output_shape)
                    else:
                        model = rnn.RNN(input_shape, output_shape)
                    target_model = None
                    path = 'individual_model/agent_{}/model'.format(i+1)
                    path = os.path.join(self.model_path, path)
                    model.load_state_dict(torch.load(path))
                    model.eval()

            pass
        return agents

    def step(self):
        """ 
        complete one step interaction
        
        """
        if self.buffer_index == 0:
            self.episode_reward = 0.
        actions = []
        for i, agent in enumerate(self.agents):
            actions.append(agent.act(self.observations[i]))

        actions_bench = None
        #! benchmark, Round-Robin
        if self.benchmark == 'roundrobin':
            self.benchmark_action_needed = self.benchmark_action_needed[1:] + [self.benchmark_action_needed[0]]
            actions_bench = np.array(self.benchmark_action_needed)
        elif self.benchmark == 'random':
            # actions_bench = np.random.randint(self.env.action_space, size=(self.num_agent,))
            actions_bench = np.array([random.randint(0, self.env.action_space - 1) for _ in range(self.num_agent)])
        #! apply the actions
        new_observations, reward = self.env.step(np.array(actions), actions_bench)
        
        #! store the experiences
        if self.training:
            self.obs_buffer[self.buffer_index] = self.observations
            self.act_buffer[self.buffer_index] = np.array(actions)
            self.reward_buffer[self.buffer_index, :] = reward          #! here reward is a float variable, not an array
            self.next_obs_buffer[self.buffer_index] = new_observations
            self.episode_reward += reward
            self.buffer_index += 1
            #! episode terminates
            if self.buffer_index % self.episode_length == 0:
                self.memory.store(self.obs_buffer, self.act_buffer, self.reward_buffer, self.next_obs_buffer)
                self.buffer_index = 0
                self.obs_buffer = np.zeros((self.episode_length, self.num_agent, self.env.observation_space))
                self.act_buffer = np.zeros((self.episode_length, self.num_agent), dtype=np.int)
                self.reward_buffer = np.zeros((self.episode_length, self.num_agent))
                self.next_obs_buffer = np.zeros((self.episode_length, self.num_agent, self.env.observation_space))
        self.observations = new_observations


    def train(self):
        """
        train all agents with sampled traces from replay buffer
        """
        loss = 0.
        #! ensure the replay buffer has enough sample
        if self.memory.count < self.batch_size:
            return loss
        #! sample the joint multi-agent experiences
        obs, actions, rewards, next_obs = self.memory.sample(self.batch_size, self.trace_length)
       
        #! when use parameter sharing, only ego agent learns
        if self.parameter_sharing:
            loss = self.agents[0].learn((obs[:, :, 0, :], actions[:, :, 0], rewards[:, :, 0], next_obs[:, :, 0, :]))
        else:
            for i, agent in enumerate(self.agents):
                loss += agent.learn((obs[:, :, i, :], actions[:, :, i], rewards[:, :, i], next_obs[:, :, i, :]))
            loss /= self.num_agent
        return loss

    

    def update_eps_hys(self, follow_eps = False):
        """ update epsilon and hysteretic rate after each episode

        Args:
            follow_eps (bool, optional):  if hysteretic rate corresponds to epsilon. Defaults to True.

        """
        self.epsilon = max(self.final_epsilon, self.epsilon - (1 - self.final_epsilon) / self.anneal_period)
        if follow_eps:
            self.hysteretic = 1 - self.epsilon
        else:
            self.hysteretic = min(self.final_hysteretic, self.hysteretic + (self.final_hysteretic - self.init_hysteretic) / self.anneal_period)

        for agent in self.agents:
            agent.update_eps_hys(self.epsilon, self.hysteretic)


   
    def update_target_model(self):
        assert self.training == True
        if self.parameter_sharing:
            self.agents[0].update_target_model()
        else:
            for agent in self.agents:
                agent.update_target_model()

    def save_models(self):
        if self.parameter_sharing:
            file_path = os.path.join(self.model_path, 'shared_model/model')
            self.agents[0].save_model(file_path)
        else:
            for i, agent in enumerate(self.agents):
                path = 'individual_model/agent_{}/model'.format(i)
                file_path = os.path.join(self.model_path,  path)
                agent.save_model(file_path)

    #// since this function has been integrated into self.create_agents(), so not needed actually
    def load_models(self):
        raise NotImplementedError

