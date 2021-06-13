import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F


#! use GPU by default, so make sure the model and ipu data both on GPU device
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device('cpu')


class DQN(nn.Module):
    def __init__(self, input_shape, output_shape):
        """
        
        """
        super().__init__()
        self.fc1 = nn.Linear(input_shape, 500)
        self.bn1 = nn.BatchNorm1d(500)
        self.fc2 = nn.Linear(500, 250)
        self.bn2 = nn.BatchNorm1d(250)
        self.fc3 = nn.Linear(250, 120)
        self.bn3 = nn.BatchNorm1d(120)
        self.fc4 = nn.Linear(120, output_shape)

    def forward(self, x):
        x = self.bn1(F.relu(self.fc1(x)))
        x = self.bn2(F.relu(self.fc2(x)))
        x = self.bn3(F.relu(self.fc3(x)))
        return self.fc4(x)



class QAgent:
    def __init__(self,
                 model,
                 target_model,
                 memory,
                 action_space,
                 batch_size=32,
                 learning_rate=0.0001,
                 gamma=0.95,
                 epsilon=0.99,
                 training=True) -> None:
        """
        Vanilla DQN
        """
        self.model = model
        self.target_model = target_model
        self.memory = memory
        self.action_space = action_space
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.training = training

        self.optimizer = torch.optim.RMSprop(params=self.model.parameters(), lr=self.learning_rate, momentum=0.95, eps=0.01)



    def update_target_model(self):
        """update the target network
        """
        self.target_model.load_state_dict(self.model.state_dict())


    def save_model(self, path):
        torch.save(self.model.state_dict(), path)


    def load_model(self, path):
        self.model.load_state_dict(torch.load(path))
        self.model.eval()


    def update_epsilon(self, epsilon):
        self.epsilon = epsilon




    #! consider the available actions don't change
    def act(self, observation):
        """perform epsilon-greedy when training, max-Q when evaluation

        Args:
            observation ([type]): [description]
        """
        self.model.eval()
        observation = torch.FloatTensor(observation).unsqueeze(dim=0)

        qs = self.model(observation)
        action = None

        if self.training and np.random.rand() < self.epsilon:
            action = np.random.randint(self.action_space)
        else:
            action = np.argmax(qs.cpu().detach().numpy())

        return action



    def learn(self, obs_batch, action_batch, reward_batch, next_obs_batch):

        obs = torch.FloatTensor(obs_batch)
        next_obs = torch.FloatTensor(next_obs_batch)
        act = torch.LongTensor(action_batch)
        reward = torch.FloatTensor(reward_batch)


        #! double q learning
        qs = self.model(obs)
        qs_actions = qs[torch.arange(qs.shape[0]), act]

        next_qs = self.model(next_obs)
        next_actions = next_qs.argmax(axis = -1)
        next_qs_target = self.target_model(next_obs)
        next_qs_actions = next_qs_target[torch.arange(next_qs_target.shape[0]), next_actions]

        targets = reward + self.gamma * next_qs_actions
        loss = F.smooth_l1_loss(qs_actions, targets.detach())

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return float(loss)




        



