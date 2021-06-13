from numpy.core.fromnumeric import mean
import torch
import torch.nn.functional as F
import numpy as np


class QAgent:
    def __init__(self,
                 model,
                 target_model,
                 action_space=16,
                 batch_size=32,
                 learning_rate=0.0001,
                 gamma=0.95,
                 epsilon=0.99,
                 hysteretic=0.2,
                 huber_beta=1.0,
                 training=True) -> None:
        """ initialize the Double DQN agent

        Args:
            model: evaluation network
            target_model: target network
            learning_rate: learning rate. Defaults to 0.001.
            gamma: discount factor. Defaults to 0.9.
            epsilon: exploration rate. Defaults to 0.5.
            hysteretic: hysteretic learning rate. Defaults to 0.4.
            huber_beta: use for Huber loss
        """
        #! RNN or dueling RNN
        self.model = model
        self.target_model = target_model
        #! hidden.hidden
        self.hidden = None

        self.action_space = action_space
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.hysteretic = hysteretic
        self.huber_beta = huber_beta
        self.training = training

        self.optimizer = torch.optim.Adam(params=self.model.parameters(), lr=self.learning_rate)

        #! initialize hidden.hidden
        self.init_hidden_state()



    def init_hidden_state(self):
        """reset hidden states of the GRU layer
        """
        self.hidden = self.model.get_initial_hidden(1)



    def update_target_model(self):
        """update the target network
        """
        self.target_model.load_state_dict(self.model.state_dict())


    def save_model(self, path):
        torch.save(self.model.state_dict(), path)


    def load_model(self, path):
        self.model.load_state_dict(torch.load(path))
        self.model.eval()


    def update_eps_hys(self, epsilon, hysteretic):
        """ update epsilon greedy rate and hysteretic rate """
        self.epsilon = epsilon
        self.hysteretic = hysteretic



    #! consider the available actions don't change
    def act(self, observation):
        """perform epsilon-greedy when training, max-Q when evaluation

        Args:
            observation ([type]): [description]
        """
        self.model.eval()
        observation = torch.FloatTensor(observation.reshape(1, 1, -1))

        qs, self.hidden = self.model(observation, self.hidden)
        action = None

        if self.training and np.random.rand() < self.epsilon:
            action = np.random.randint(self.action_space)
        else:
            action = np.argmax(qs.detach().numpy().squeeze())

        return action



    def learn(self, batch):
        """ learn from a batch of single agent traces

        Args:
            batch ([type]): [time_step, batch_size] * transitions
        """
        self.model.train()
        obs_batch, act_batch, reward_batch, next_obs_batch = batch

        observations = torch.FloatTensor(obs_batch)
        next_observations = torch.FloatTensor(next_obs_batch)
        actions = torch.LongTensor(act_batch).flatten()
        rewards = torch.FloatTensor(reward_batch).flatten()


        loss = 0.0
        
        predicted_qvalues, _ = self.model(observations, batch_size = self.batch_size)       #! shape = (batch, sequence, action)
        predicted_qvalues = predicted_qvalues.reshape(-1, self.action_space).squeeze(0)
        predicted_qvalues_actions = predicted_qvalues[range(predicted_qvalues.shape[0]), actions]
        #! double q-learning
        predicted_next_qvalues, _ = self.model(next_observations, batch_size = self.batch_size)
        predicted_next_qvalues = predicted_next_qvalues.reshape(-1, self.action_space).squeeze(0)
        next_actions =  predicted_next_qvalues.argmax(axis = -1) 

        target_qvalues, _ = self.target_model(next_observations, batch_size = self.batch_size)
        target_qvalues = target_qvalues.reshape(-1, self.action_space).squeeze(0)
        target_qvalues_actions = target_qvalues[range(target_qvalues.shape[0]), next_actions]

        #! no double q-learaning
        # predicted_next_qvalues, _ = self.target_model(next_observations, batch_size = self.batch_size);
        # predicted_next_qvalues = predicted_next_qvalues.reshape(-1, self.action_space).squeeze(0)
        # target_qvalues_actions = predicted_next_qvalues.max(-1)[0].reshape(self.batch_size, 1)

        td_errors = rewards + self.gamma * target_qvalues_actions - predicted_qvalues_actions
        td_errors = torch.max(td_errors, self.hysteretic * td_errors)

        

        # mean squared error loss to minimize
        loss = torch.mean(self.huber_loss(td_errors))
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return float(loss)

    def huber_loss(self, residual):
        K = self.huber_beta
        abs_residual = torch.abs(residual)

        # qradratic portion when residual is small
        small_res = 0.5 / K * torch.square(residual) * (abs_residual <= K).float()
        # linear portion when residual is bigger
        large_res = (abs_residual - 0.5 * K) * (abs_residual > K).float()
        return small_res + large_res
