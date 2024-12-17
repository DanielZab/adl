'''
The network module contains the implementation of the RNNModel. It is a modified version of the mlp-network available in https://github.com/sidhantls/ppo_lightning. 

The selected neural network is a RNN or variations thereof (LSTM, GRU) and is used in a Actor-Critic architecture. The RNNModel is used as a policy or value network as a standalone module, or in the ActorCategorical and ActorContinous classes. The ActorCategorical and ActorContinous classes are policy networks for discrete and continuous action spaces, respectively. The ActorCriticAgent class is an actor-critic agent used during trajectory collection.

This module contains several classes:

- RNNModel: A recurrent neural network model that can be used as a policy or value network.
- ActorCategorical: A policy network for discrete action spaces, which returns a distribution and an action given an observation.
- ActorContinous: A policy network for continuous action spaces, which also returns a distribution and an action given an observation.
- ActorCriticAgent: An actor-critic agent used during trajectory collection, which returns a distribution and an action given an observation.

'''

import logging
import torch
from torch import nn
from torch.distributions import Categorical, Normal, TransformedDistribution, AffineTransform, Distribution
from torch.utils.data import IterableDataset
from typing import Union, Tuple, Callable, Iterable, List


class RNNModel(nn.Module):
    '''
    A recurrent neural network model. It consists of a rnn part and a fully connected part. See README for more architectural details.
    '''

    def __init__(self, input_size: int, output_size: int, rec_hidden_size: int, fc_hidden_sizes: List[int],
                 rec_num_layers: int = 2, rec_nonlinearity: str = "tanh", fc_nonlinearity: str = "tanh",
                 rnn_type: str = 'RNN', dropout: float = 0.0):
        '''
        input_size: The input dimension of the network
        output_size: The output dimension of the network
        rec_hidden_size: The hidden size of the recurrent part of the network
        fc_hidden_sizes: A list of hidden sizes for the fully connected part of the network
        rec_num_layers: The number of layers in the recurrent part of the network
        rec_nonlinearity: The nonlinearity used in the recurrent part of the network
        fc_nonlinearity: The nonlinearity used in the fully connected part of the network
        rnn_type: The type of RNN used
        dropout: The dropout rate
        '''

        super(RNNModel, self).__init__()

        self.input_size = input_size
        self.output_size = output_size
        self.rec_num_layers = rec_num_layers
        self.rnn_type = rnn_type

        self.rec_hidden_size = rec_hidden_size

        # Choose RNN type: RNN, LSTM, or GRU
        if rnn_type == 'RNN':
            self.rnn = nn.RNN(input_size, rec_hidden_size, rec_num_layers, nonlinearity=rec_nonlinearity,
                              dropout=dropout, batch_first=False)
        elif rnn_type == 'LSTM':
            self.rnn = nn.LSTM(input_size, rec_hidden_size, rec_num_layers, dropout=dropout, batch_first=False)
        elif rnn_type == 'GRU':
            self.rnn = nn.GRU(input_size, rec_hidden_size, rec_num_layers, dropout=dropout, batch_first=False)
        else:
            raise ValueError(f"Unsupported RNN type: {rnn_type}")

        # Choose nonlinearity for fully connected layers
        if fc_nonlinearity == "tanh":
            self.fc_nonlinearity = nn.Tanh
        elif fc_nonlinearity == "sigmoid":
            self.fc_nonlinearity = nn.Sigmoid
        elif fc_nonlinearity == "relu":
            self.fc_nonlinearity = nn.ReLU
        else:
            raise ValueError(f"Unsupported non-recurrent nonlinearity: {fc_nonlinearity}")

        fc_layers = []

        fc_layers.append(nn.Linear(rec_hidden_size + self.input_size, fc_hidden_sizes[0]))
        fc_layers.append(self.fc_nonlinearity())

        for i in range(len(fc_hidden_sizes) - 1):
            fc_layers.append(nn.Linear(fc_hidden_sizes[i], fc_hidden_sizes[i + 1]))
            fc_layers.append(self.fc_nonlinearity())

        fc_layers.append(nn.Linear(fc_hidden_sizes[-1], output_size))

        self.fc = nn.Sequential(*fc_layers)

    def forward(self, x, hn) -> Tuple[torch.Tensor, torch.Tensor]:

        '''
        The forward pass of the network
        To ensure integrity, the shapes of the tensors are checked at each step
        The sequence length is assumed to be 1

        x: The input tensor
        hn: The hidden state of the network

        Returns:
        output: The output tensor
        hn: The new hidden state of the network
        '''

        # Move to correct device
        if self.rnn_type == 'LSTM':
            hn = (hn[0].to(x.device), hn[1].to(x.device))
        else:
            hn = hn.to(x.device)

        # Establish if batched or unbatched and swap first and second dimensions, as RNN expects (num_layers, batch, input_size) for hidden state, but we have (batch, num_layers, input_size)
        if len(x.shape) == 1:
            batch_size = 1
        else:
            if self.rnn_type == 'LSTM':
                hn = (hn[0].transpose(0, 1).contiguous(), hn[1].transpose(0, 1).contiguous())
            else:
                hn = hn.transpose(0, 1).contiguous()
            batch_size = x.shape[0]
        output, hn = self.rnn(torch.unsqueeze(x, 0), hn)

        if batch_size == 1:
            assert output.shape == (1, self.rec_hidden_size)
        else:
            assert output.shape == (1, batch_size, self.rec_hidden_size)

        output = torch.squeeze(output)

        if batch_size == 1:
            assert output.shape == (self.rec_hidden_size,)
        else:
            assert output.shape == (batch_size, self.rec_hidden_size)

        output = torch.cat((output, x), dim=-1)

        if batch_size == 1:
            assert output.shape == (self.rec_hidden_size + self.input_size,)
        else:
            assert output.shape == (batch_size, self.rec_hidden_size + self.input_size)

        output = self.fc(output)

        if batch_size == 1:
            assert output.shape == (self.output_size,)
        else:
            assert output.shape == (batch_size, self.output_size)

        return output, hn

    def init_hidden(self, batch_size) -> torch.Tensor:
        '''
        Initialize the initial hidden state of the network. The hidden state is a tensor of zeros and it is different for LSTM and RNN/GRU.

        Returns: hidden state
        '''
        if isinstance(self.rnn, nn.LSTM):
            first = torch.squeeze(torch.zeros(self.rec_num_layers, batch_size, self.rec_hidden_size))
            second = torch.squeeze(torch.zeros(self.rec_num_layers, batch_size, self.rec_hidden_size))

            # For batch size 1 and num_layers 1, we need to add an extra dimension
            if len(first.shape) < 2:
                first = torch.unsqueeze(first, 0)
                second = torch.unsqueeze(second, 0)
            return first, second
        else:
            temp = torch.squeeze(torch.zeros(self.rec_num_layers, batch_size, self.rec_hidden_size))
            # For batch size 1 and num_layers 1, we need to add an extra dimension
            if len(temp.shape) < 2:
                temp = torch.unsqueeze(temp, 0)
            return temp


class ActorCategorical(nn.Module):
    """
    Policy network, for discrete action spaces, which returns a distribution
    and an action given an observation
    """

    def __init__(self, actor_net: RNNModel):
        """
        actor_net: The RNNModel used as the policy network
        """
        super().__init__()

        self.actor_net = actor_net

    def forward(self, states, hn) -> Tuple[Categorical, torch.Tensor, torch.Tensor]:
        '''
        The forward pass of the network. The network returns a distribution and an action given an observation as well as the hidden state.
        '''
        logits, hn = self.actor_net(states, hn)

        # Create a categorical distribution based on network output
        pi = Categorical(logits=logits)

        # Shift the distribution to have the center (but not always the mean) at 0
        pi = TransformedDistribution(pi, AffineTransform(loc=-(self.actor_net.output_size // 2), scale=1.0))

        # Sample an action from the distribution
        actions = pi.sample().type(torch.int32)

        return pi, actions, hn

    def get_log_prob(self, pi: Categorical, actions: torch.Tensor) -> torch.Tensor:
        """
        Takes in a distribution and actions and returns log prob of actions
        under the distribution

        pi: torch distribution
        actions: actions taken by distribution

        Returns: log probability of the acition under pi
        """
        return pi.log_prob(actions)


class ActorContinous(nn.Module):
    """
    Policy network, for continous action spaces, which returns a distribution
    and an action given an observation
    """

    def __init__(self, actor_net, act_dim):
        """
        actor_net: The RNNModel used as the policy network
        act_dim: The dimension of the action space, in our case always 1
        """
        super().__init__()
        self.actor_net = actor_net
        log_std = -0.5 * torch.ones(act_dim, dtype=torch.float)
        self.log_std = torch.nn.Parameter(log_std)

    def forward(self, states, hn) -> Tuple[Normal, torch.Tensor, torch.Tensor]:
        '''
        The forward pass of the network.

        states: The input tensor
        hn: The hidden state of the network

        Returns: distribution, action, hidden state
        '''
        mu, hn = self.actor_net(states, hn)
        std = torch.exp(self.log_std)
        pi = Normal(loc=mu, scale=std)
        actions = pi.sample()

        return pi, actions, hn

    def get_log_prob(self, pi: Normal, actions: torch.Tensor):
        """
        Takes in a distribution and actions and returns log prob of actions
        under the distribution

        pi: torch distribution
        actions: actions taken by distribution

        Returns: log probability of the acition under pi
        """
        return pi.log_prob(actions).sum(axis=-1)


class ActorCriticAgent(object):
    """
    Actor Critic Agent used during trajectory collection. It returns a
    distribution and an action given an observation. Agent based on the
    implementations found here: https://github.com/Shmuma/ptan/blob/master/ptan/agent.py
    """

    def __init__(self, actor_net: nn.Module, critic_net: nn.Module):
        '''
        actor_net: The policy network
        critic_net: The value network
        '''
        self.actor_net = actor_net
        self.critic_net = critic_net

    @torch.no_grad()
    def __call__(self, state: torch.Tensor, device: str, hn_actor, hn_critic) -> Tuple[
        Distribution, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Takes in the current state and returns the agents policy, sampled
        action, log probability of the action, and value of the given state

        states: current state of the environment
        device: the device used for the current batch
        
        Returns: torch dsitribution, randomly sampled action, log probability of the action, value of the state, hidden state of the actor, hidden state of the critic
        """

        state = state.to(device=device)

        pi, actions, hn_a = self.actor_net(state, hn_actor)
        log_p = self.get_log_prob(pi, actions)

        value, hn_c = self.critic_net(state, hn_critic)

        return pi, actions, log_p, value, hn_a, hn_c

    def get_log_prob(self,
                     pi: Union[Categorical, Normal],
                     actions: torch.Tensor) -> torch.Tensor:
        """
        Takes in a distribution and actions and returns log prob of actions

        pi: torch distribution
        actions: actions taken by distribution
    
        Returns: log probability of the acition under pi
        """
        return self.actor_net.get_log_prob(pi, actions)


class ExperienceSourceDataset(IterableDataset):
    """
    Implementation from PyTorch Lightning Bolts:
    https://github.com/PyTorchLightning/pytorch-lightning-bolts/blob/master/pl_bolts/datamodules/experience_source.py

    Basic experience source dataset. Takes a generate_batch function that returns an iterator.
    The logic for the experience source and how the batch is generated is defined the Lightning model itself
    """

    def __init__(self, generate_batch: Callable):
        self.generate_batch = generate_batch

    def __iter__(self) -> Iterable:
        iterator = self.generate_batch()
        return iterator
