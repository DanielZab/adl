from typing import Union, Tuple, Callable, Iterable

import torch
from torch import nn
from torch.utils.data import IterableDataset
from torch.distributions import Categorical, Normal
import logging

# Source: https://github.com/sidhantls/ppo_lightning

class RNNModel(nn.Module):
    def __init__(self, rec_input_size, fc1_input_size, rec_hidden_size, fc1_hidden_sizes, fc2_hidden_sizes, rec_num_layers=2, rec_nonlinearity="tanh", fc1_nonlinearity="tanh", fc2_nonlinearity="tanh", rnn_type='RNN', dropout=0.0):
        super(RNNModel, self).__init__()

        # Choose RNN type: RNN, LSTM, or GRU
        if rnn_type == 'RNN':
            self.rnn = nn.RNN(rec_input_size, rec_hidden_size, rec_num_layers, nonlinearity=rec_nonlinearity, dropout=dropout, batch_first=True)
        elif rnn_type == 'LSTM':
            self.rnn = nn.LSTM(rec_input_size, rec_hidden_size, rec_num_layers, nonlinearity=rec_nonlinearity, dropout=dropout, batch_first=True)
        elif rnn_type == 'GRU':
            self.rnn = nn.GRU(rec_input_size, rec_hidden_size, rec_num_layers, nonlinearity=rec_nonlinearity, dropout=dropout, batch_first=True)
        else:
            raise ValueError(f"Unsupported RNN type: {rnn_type}")
        
        if fc1_nonlinearity == "tanh":
            self.fc1_nonlinearity = nn.Tanh
        elif fc1_nonlinearity == "sigmoid":
            self.fc1_nonlinearity = nn.Sigmoid
        elif fc1_nonlinearity == "relu":
            self.fc1_nonlinearity = nn.ReLU
        else:
            raise ValueError(f"Unsupported non-recurrent nonlinearity: {fc1_nonlinearity}")
        
        if fc2_nonlinearity == "tanh":
            self.fc2_nonlinearity = nn.Tanh
        elif fc2_nonlinearity == "sigmoid":
            self.fc2_nonlinearity = nn.Sigmoid
        elif fc2_nonlinearity == "relu":
            self.fc2_nonlinearity = nn.ReLU
        else:
            raise ValueError(f"Unsupported non-recurrent nonlinearity: {fc1_nonlinearity}")

        fc1_layers = []
        
        fc1_layers.append(nn.Linear(rec_hidden_size + fc1_input_size, fc1_hidden_sizes[0]))
        fc1_layers.append(self.fc1_nonlinearity())

        for i in range(len(fc1_hidden_sizes)-1):
            fc1_layers.append(nn.Linear(fc1_hidden_sizes[i], fc1_hidden_sizes[i+1]))
            fc1_layers.append(self.fc1_nonlinearity())

        nn.Linear(rec_hidden_size + fc1_hidden_sizes[-1], fc2_hidden_sizes[0])
        self.fc1 = nn.Sequential(*fc1_layers)

        fc2_layers = []

        for i in range(fc2_hidden_sizes - 1):
            fc2_layers.append(nn.Linear(fc2_hidden_sizes[i], fc2_hidden_sizes[i+1]))
            fc2_layers.append(self.fc2_nonlinearity())
        
        fc2_layers.append(nn.Linear(fc2_hidden_sizes[-1], 1))


def create_mlp(input_shape: Tuple[int], n_actions: int, hidden_sizes: list = [128, 128]):
    """
    Simple Multi-Layer Perceptron network
    """

    logging.debug(f"create_mlp called with {input_shape}, {n_actions}, {hidden_sizes}")
    net_layers = []
    net_layers.append(nn.Linear(input_shape, hidden_sizes[0]))
    net_layers.append(nn.Tanh())

    for i in range(len(hidden_sizes)-1):
        net_layers.append(nn.Linear(hidden_sizes[i], hidden_sizes[i+1]))
        net_layers.append(nn.Tanh())
    net_layers.append(nn.Linear(hidden_sizes[-1], n_actions))

    return nn.Sequential(*net_layers)


class ActorCategorical(nn.Module):
    """
    Policy network, for discrete action spaces, which returns a distribution
    and an action given an observation
    """

    def __init__(self, actor_net):
        """
        Args:
            input_shape: observation shape of the environment
            n_actions: number of discrete actions available in the environment
        """
        super().__init__()

        self.actor_net = actor_net

    def forward(self, states):
        logits = self.actor_net(states)
        pi = Categorical(logits=logits)
        actions = pi.sample()

        return pi, actions

    def get_log_prob(self, pi: Categorical, actions: torch.Tensor):
        """
        Takes in a distribution and actions and returns log prob of actions
        under the distribution
        Args:
            pi: torch distribution
            actions: actions taken by distribution
        Returns:
            log probability of the acition under pi
        """
        return pi.log_prob(actions)


class ActorContinous(nn.Module):
    """
    Policy network, for continous action spaces, which returns a distribution
    and an action given an observation
    """

    def __init__(self, actor_net, act_dim):
        """
        Args:
            input_shape: observation shape of the environment
            n_actions: number of discrete actions available in the environment
        """
        super().__init__()
        self.actor_net = actor_net
        log_std = -0.5 * torch.ones(act_dim, dtype=torch.float)
        self.log_std = torch.nn.Parameter(log_std)

    def forward(self, states):
        mu = self.actor_net(states)
        std = torch.exp(self.log_std)
        pi = Normal(loc=mu, scale=std)
        actions = pi.sample()

        return pi, actions

    def get_log_prob(self, pi: Normal, actions: torch.Tensor):
        """
        Takes in a distribution and actions and returns log prob of actions
        under the distribution
        Args:
            pi: torch distribution
            actions: actions taken by distribution
        Returns:
            log probability of the acition under pi
        """
        return pi.log_prob(actions).sum(axis=-1) 

class ActorCriticAgent(object):
    """
    Actor Critic Agent used during trajectory collection. It returns a
    distribution and an action given an observation. Agent based on the
    implementations found here: https://github.com/Shmuma/ptan/blob/master/ptan/agent.py

    """
    def __init__(self, actor_net: nn.Module, critic_net: nn.Module):
        self.actor_net = actor_net
        self.critic_net = critic_net

    @torch.no_grad()
    def __call__(self, state: torch.Tensor, device: str) -> Tuple:
        """
        Takes in the current state and returns the agents policy, sampled
        action, log probability of the action, and value of the given state
        Args:
            states: current state of the environment
            device: the device used for the current batch
        Returns:
            torch dsitribution and randomly sampled action
        """

        state = state.to(device=device)

        pi, actions = self.actor_net(state)
        log_p = self.get_log_prob(pi, actions)

        value = self.critic_net(state)

        return pi, actions, log_p, value

    def get_log_prob(self,
                     pi: Union[Categorical, Normal],
                     actions: torch.Tensor) -> torch.Tensor:
        """
        Takes in the current state and returns the agents policy, a sampled
        action, log probability of the action, and the value of the state
        Args:
            pi: torch distribution
            actions: actions taken by distribution
        Returns:
            log probability of the acition under pi
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
