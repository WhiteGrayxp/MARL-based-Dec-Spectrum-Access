import torch
import torch.nn as nn
import torch.nn.functional as F



class RNN(nn.Module):
    def __init__(self,
                 input_shape,
                 output_shape,
                 rnn_hidden_dim=128):
        super().__init__()
        self.rnn_hidden_dim = rnn_hidden_dim
        self.fc1 = nn.Linear(input_shape, 64)
        #! the input and output tensors are provided as (batch, seq, feature)
        self.rnn = nn.GRU(64, rnn_hidden_dim, batch_first = True)  
        self.fc2 = nn.Linear(rnn_hidden_dim, output_shape)

    def get_initial_hidden(self, batch_size):
        return torch.zeros(1, batch_size, self.rnn_hidden_dim)


    def forward(self, inputs, state = None, batch_size = 1):
        """ 
        one time step forward
        """
        if state is None:
            state = self.get_initial_hidden(batch_size)
        x = F.relu(self.fc1(inputs))
        output, state = self.rnn(x, state)
        output = self.fc2(output)

        return output, state





class DuelingRNN(nn.Module):
    def __init__(self,
                 input_shape,
                 output_shape,
                 rnn_hidden_dim=128):
        super().__init__()
        self.rnn_hidden_dim = rnn_hidden_dim
        self.fc1 = nn.Linear(input_shape, 64)
        #! the input and output tensors are provided as (batch, seq, feature)
        self.rnn = nn.GRU(64, rnn_hidden_dim, batch_first = True)

        self.value_stream = nn.Sequential(
            nn.Linear(rnn_hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

        self.advantage_stream = nn.Sequential(
            nn.Linear(rnn_hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, output_shape)
        )


    def get_initial_hidden(self, batch_size):
        return torch.zeros(1, batch_size, self.rnn_hidden_dim)


    def forward(self, inputs, state = None, batch_size = 1):
        """ 
        one time step forward
        """
        if state is None:
            state = self.get_initial_hidden(batch_size)
        outputs = F.relu(self.fc1(inputs))
        outputs, state = self.rnn(outputs, state)
        #! two streams
        values = self.value_stream(outputs)
        advantages = self.advantage_stream(outputs)

        qs = values + (advantages - torch.mean(advantages, dim = -1, keepdim = True))

        return qs, state
