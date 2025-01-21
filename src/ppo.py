"""
The PPO class is a PyTorch Lightning implementation of the Proximal Policy Optimization (PPO) algorithm. The PPO algorithm is a policy gradient method that aims to optimize the policy network by maximizing the expected return of the agent. It uses a clipped surrogate objective to prevent large policy updates that can lead to instability in training.

This module is a modification of the class available in https://github.com/sidhantls/ppo_lightning, which is based on https://github.com/openai/baselines/blob/master/baselines/ppo2/ppo2.py and https://github.com/PyTorchLightning/pytorch-lightning-bolts
"""

import gymnasium as gym
import pytorch_lightning as pl
import torch
import torch.optim as optim
from torch.distributions import Distribution
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader
from typing import List, Tuple, Generator

from environment import Config
from network import ActorCriticAgent, ActorCategorical, ActorContinous, RNNModel
from network import ExperienceSourceDataset

class PPO(pl.LightningModule):
    """
    Proximal Policy Optimization (PPO) class used to train the agent in the custom environment. The class contains the logic for training the policy (actor) and value (critic) network using the PPO algorithm. It also contains the logic for generating trajectory data to train the network, calculating the discounted rewards, and calculating the advantage and it handles the storage of the internal states of the network.
    """

    def __init__(
        self,
        env: gym.Env,
        config: Config,
        gamma: float = 0.99,
        lam: float = 0.95,
        lr_actor: float = 3e-4,
        lr_critic: float = 1e-3,
        max_episode_len: float = 1000,
        batch_size: int = 512,
        steps_per_epoch: int = 2048,
        nb_optim_iters: int = 4,
        clip_ratio: float = 0.2,
        rec_hidden_size: int = 128,
        fc_hidden_sizes: int = [128, 128],
        rec_num_layers: int = 2,
        rec_nonlinearity: str = "tanh",
        fc_nonlinearity: str = "tanh",
        rnn_type: str = "RNN",
        dropout: float = 0.0,
    ) -> None:

        """
        env: gym environment
        config: the environment configuration
        gamma: discount factor
        lam: advantage discount factor (lambda in the paper)
        lr_actor: learning rate of actor network
        lr_critic: learning rate of critic network
        max_episode_len: maximum number interactions (actions) in an episode
        batch_size:  batch_size when training network- can simulate number of policy updates performed per epoch
        steps_per_epoch: how many action-state pairs to rollout for trajectory collection per epoch
        nb_optim_iters: how many steps of gradient descent to perform on each batch
        clip_ratio: hyperparameter for clipping in the policy objective
        rec_hidden_size: hidden size of the recurrent layer
        fc_hidden_sizes: hidden sizes of the fully connected layers
        rec_num_layers: number of layers in the recurrent network
        rec_nonlinearity: nonlinearity of the recurrent network
        fc_nonlinearity: nonlinearity of the fully connected layers
        rnn_type: type of the recurrent network
        dropout: dropout rate
        """
        super().__init__()

        self.automatic_optimization = False

        # Hyperparameters
        self.lr_actor = lr_actor
        self.lr_critic = lr_critic
        self.steps_per_epoch = steps_per_epoch
        self.nb_optim_iters = nb_optim_iters
        self.batch_size = batch_size
        self.gamma = gamma
        self.lam = lam
        self.max_episode_len = max_episode_len
        self.clip_ratio = clip_ratio
        self.rec_hidden_size = rec_hidden_size
        self.fc_hidden_sizes = fc_hidden_sizes
        self.rec_num_layers = rec_num_layers
        self.rec_nonlinearity = rec_nonlinearity
        self.fc_nonlinearity = fc_nonlinearity
        self.rnn_type = rnn_type
        self.dropout = dropout
        self.config = config
        self.save_hyperparameters()

        self.env = env

        # value network
        self.critic = RNNModel(
            input_size=4,
            output_size=1,
            rec_hidden_size=rec_hidden_size,
            rec_num_layers=rec_num_layers,
            rec_nonlinearity=rec_nonlinearity,
            fc_hidden_sizes=fc_hidden_sizes,
            fc_nonlinearity=fc_nonlinearity,
            rnn_type=rnn_type,
            dropout=dropout,
        )

        # policy network (agent)

        if not config.CONTINUOUS_MODEL:
            actor_mlp = RNNModel(
                input_size=4,
                output_size=2 * config.MAX_BUY_LIMIT + 1,
                rec_hidden_size=rec_hidden_size,
                rec_num_layers=rec_num_layers,
                rec_nonlinearity=rec_nonlinearity,
                fc_hidden_sizes=fc_hidden_sizes,
                fc_nonlinearity=fc_nonlinearity,
                rnn_type=rnn_type,
                dropout=dropout,
            )
            self.actor = ActorCategorical(actor_mlp)
        else:
            act_dim = self.env.action_space.shape[0]
            assert (
                act_dim == 1
            ), "Only single action dimension supported for continuous action space"
            actor_mlp = RNNModel(
                input_size=4,
                output_size=act_dim,
                rec_hidden_size=rec_hidden_size,
                rec_num_layers=rec_num_layers,
                rec_nonlinearity=rec_nonlinearity,
                fc_hidden_sizes=fc_hidden_sizes,
                fc_nonlinearity=fc_nonlinearity,
                rnn_type=rnn_type,
                dropout=dropout,
            )
            self.actor = ActorContinous(actor_mlp, act_dim)

        # agent
        self.agent = ActorCriticAgent(self.actor, self.critic)

        self.batch_states = []
        self.batch_actions = []
        self.batch_adv = []
        self.batch_qvals = []
        self.batch_logp = []
        self.hns_actor = []
        self.hns_critic = []

        self.ep_rewards = []
        self.ep_values = []
        self.epoch_rewards = []
        self.portfolio_values = []

        self.episode_step = 0
        self.avg_ep_reward = 0
        self.avg_ep_len = 0
        self.avg_reward = 0
        self.avg_portofolio_value = 0
        self.max_portfolio_value = 0

        self.hn_actor = self.actor.actor_net.init_hidden(1)
        self.hn_critic = self.critic.init_hidden(1)
        
        if env is not None:
            self.state = self.convert_state_to_tensor(self.env.reset())

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[Distribution, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Passes in a state x through the network and returns the policy and a sampled action
        x: environment state

        Returns: policy, action, value, hidden state actor, hidden state critic
        """
        pi, action, hn_a = self.actor(x, self.hn_actor)
        value, hn_c = self.critic(x, self.hn_critic)

        return pi, action, value, hn_a, hn_c

    def convert_state_to_tensor(self, state: tuple) -> torch.Tensor:
        """
        Converts the state of the environment to a tensor which is accepted by the network. The values are log transformed to normalize the data.

        state: state of the environment

        Returns: network compatible tensor of the state
        """
        assert isinstance(state, tuple) and len(state) == 2

        obs = state[0]
        info = state[1]

        current_price = float(obs["current_price"])
        balance = float(obs["balance"])
        stocks_owned = float(obs["stocks_owned"])

        portfolio_value = float(info["portfolio_value"])

        return torch.log2(
            torch.FloatTensor([current_price, balance, stocks_owned, portfolio_value])
            + 1e-8
        )

    def discount_rewards(self, rewards: List[float], discount: float) -> List[float]:
        """
        Calculate the discounted rewards of all rewards in list

        rewards: list of rewards/advantages

        Returns: list of discounted rewards/advantages
        """

        assert isinstance(
            rewards[0], float
        ), f"Rewards should be of type float, got {type(rewards[0])}"

        cumul_reward = []
        sum_r = 0.0

        for r in reversed(rewards):
            sum_r = (sum_r * discount) + r
            cumul_reward.append(sum_r)

        return list(reversed(cumul_reward))

    def calc_advantage(
        self, rewards: List[float], values: List[float], last_value: float
    ) -> List[float]:
        """
        Calculate the advantage given rewards, state values, and the last value of episode

        rewards: list of episode rewards
        values: list of state values from critic
        last_value: value of last state of episode

        list of advantages
        """
        rews = rewards + [last_value]
        vals = values + [last_value]
        # GAE
        delta = [
            rews[i] + self.gamma * vals[i + 1] - vals[i] for i in range(len(rews) - 1)
        ]
        adv = self.discount_rewards(delta, self.gamma * self.lam)

        return adv

    def normalize_rewards(self, rewards: List[float] | float) -> List[float]:
        """
        Normalize the rewards using tanh

        rewards: list of rewards

        Returns: list of normalized rewards
        """
        rewards = torch.tensor(rewards)
        rewards = torch.tanh(rewards * 0.5)
        return rewards.tolist()

    def train_batch(
        self,
    ) -> Tuple[
        List[torch.Tensor],
        List[torch.Tensor],
        List[torch.Tensor],
        List[torch.Tensor],
        List[torch.Tensor],
        List[torch.Tensor],
        List[torch.Tensor],
    ]:
        """
        Contains the logic for generating trajectory data to train policy and value network

        Yields: Tuple of Lists containing tensors for states, actions, log probs, qvals, advantage, hidden actor state, hidden critic state
        """

        for step in range(self.steps_per_epoch):
            # get action from agent
            pi, action, log_prob, value, self.hn_actor, self.hn_critic = self.agent(
                self.state, self.device, self.hn_actor, self.hn_critic
            )

            # Perform action in environment
            obs, reward, done, info = self.env.step(action.cpu().numpy())

            reward = self.normalize_rewards(reward)

            next_state = (obs, info)

            self.episode_step += 1

            # Save trajectory data
            self.batch_states.append(self.state)
            self.batch_actions.append(action)
            self.batch_logp.append(log_prob)
            self.hns_actor.append(self.hn_actor)
            self.hns_critic.append(self.hn_critic)

            self.ep_rewards.append(reward)
            self.ep_values.append(value.item())

            self.state = self.convert_state_to_tensor(next_state)

            epoch_end = step == (self.steps_per_epoch - 1)
            terminal = len(self.ep_rewards) == self.max_episode_len

            # Check if end of trajectory
            if epoch_end or done or terminal:

                # if trajectory ends abtruptly, boostrap value of next state
                if (terminal or epoch_end) and not done:
                    with torch.no_grad():
                        _, _, _, value, self.hn_actor, self.hn_critic = self.agent(
                            self.state, self.device, self.hn_actor, self.hn_critic
                        )
                        last_value = value.item()
                        steps_before_cutoff = self.episode_step
                else:
                    last_value = 0
                    steps_before_cutoff = 0

                # calculate discounted cumulative reward
                self.batch_qvals += self.discount_rewards(
                    self.ep_rewards + [last_value], self.gamma
                )[:-1]

                # calcualte advantage
                self.batch_adv += self.calc_advantage(
                    self.ep_rewards, self.ep_values, last_value
                )

                self.epoch_rewards.append(sum(self.ep_rewards))
                self.portfolio_values.append(info["portfolio_value"])

                self.ep_rewards = []
                self.ep_values = []
                self.episode_step = 0

                # reset hidden states
                self.hn_actor = self.actor.actor_net.init_hidden(1)
                self.hn_critic = self.critic.init_hidden(1)

                self.state = self.convert_state_to_tensor(self.env.reset())

            # epoch is finished, yield data for training
            if epoch_end:
                train_data = zip(
                    self.batch_states,
                    self.batch_actions,
                    self.batch_logp,
                    self.batch_qvals,
                    self.batch_adv,
                    self.hns_actor,
                    self.hns_critic,
                )

                for state, action, logp_old, qval, adv, hn_a, hn_c in train_data:
                    yield state, action, logp_old, qval, adv, hn_a, hn_c

                self.batch_states.clear()
                self.batch_actions.clear()
                self.batch_adv.clear()
                self.batch_logp.clear()
                self.batch_qvals.clear()
                self.hns_actor.clear()
                self.hns_critic.clear()

                # logging
                self.avg_reward = sum(self.epoch_rewards) / self.steps_per_epoch

                # if epoch ended abruptly, exlude last cut-short episode to prevent stats skewness
                epoch_rewards = self.epoch_rewards
                if not done:
                    epoch_rewards = epoch_rewards[:-1]

                total_epoch_reward = sum(epoch_rewards)
                nb_episodes = len(epoch_rewards)

                self.avg_portofolio_value = sum(self.portfolio_values) / len(
                    self.portfolio_values
                )
                self.max_portfolio_value = max(self.portfolio_values)

                self.avg_ep_reward = total_epoch_reward / nb_episodes
                self.avg_ep_len = (
                    self.steps_per_epoch - steps_before_cutoff
                ) / nb_episodes

                self.epoch_rewards.clear()
                self.portfolio_values.clear()

    def actor_loss(self, state, hn, action, logp_old, qval, adv) -> torch.Tensor:
        """
        Calculate the actor loss using the clipped surrogate objective
        """
        pi, _, _ = self.actor(state, hn)
        logp = self.actor.get_log_prob(pi, action)
        ratio = torch.exp(logp - logp_old)
        clip_adv = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * adv
        loss_actor = -(torch.min(ratio * adv, clip_adv)).mean()
        return loss_actor

    def critic_loss(self, state, hn, action, logp_old, qval, adv) -> torch.Tensor:
        """
        Calculate the critic loss through mean squared error
        """
        value, _ = self.critic(state, hn)
        loss_critic = (qval - value).pow(2).mean()
        return loss_critic

    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx):
        """
        Carries out a single update to actor and critic network from a batch of replay buffer.

        batch: batch of replay buffer/trajectory data
        batch_idx: not used
        """
        state, action, old_logp, qval, adv, hn_a, hn_c = batch

        a_opt, c_opt = self.optimizers()

        # normalize advantages
        adv = (adv - adv.mean()) / adv.std()

        # Log important metrics
        self.log(
            "max_portfolio_value",
            float(self.max_portfolio_value),
            prog_bar=True,
            on_step=False,
            on_epoch=True,
            logger=True,
        )
        self.log(
            "avg_portfolio_value",
            float(self.avg_portofolio_value),
            prog_bar=True,
            on_step=False,
            on_epoch=True,
            logger=True,
        )
        self.log(
            "avg_ep_len",
            float(self.avg_ep_len),
            prog_bar=True,
            on_step=False,
            on_epoch=True,
            logger=True,
        )
        self.log(
            "avg_ep_reward",
            float(self.avg_ep_reward),
            prog_bar=True,
            on_step=False,
            on_epoch=True,
            logger=True,
        )
        self.log(
            "avg_reward",
            float(self.avg_reward),
            prog_bar=False,
            on_step=False,
            on_epoch=True,
            logger=True,
        )
        self.log(
            "avg_reward_atanh",
            float(torch.atanh(torch.tensor(self.avg_reward, dtype=torch.float32))),
            prog_bar=True,
            on_step=False,
            on_epoch=True,
            logger=True,
        )

        # Update actor
        a_opt.zero_grad()
        loss_actor = self.actor_loss(state, hn_a, action, old_logp, qval, adv)
        self.log(
            "loss_actor",
            float(loss_actor),
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        self.manual_backward(loss_actor)
        a_opt.step()

        # Update critic
        c_opt.zero_grad()
        loss_critic = self.critic_loss(state, hn_c, action, old_logp, qval, adv)
        self.log(
            "loss_critic",
            float(loss_critic),
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        self.manual_backward(loss_critic)
        c_opt.step()

    def configure_optimizers(self) -> Tuple[Optimizer, Optimizer]:
        """Initialize Adam optimizer"""
        optimizer_actor = optim.Adam(self.actor.parameters(), lr=self.lr_actor)
        optimizer_critic = optim.Adam(self.critic.parameters(), lr=self.lr_critic)

        return optimizer_actor, optimizer_critic

    def optimizer_step(self, *args, **kwargs):
        """
        Run 'nb_optim_iters' number of iterations of gradient descent on actor and critic
        for each data sample.
        """
        for i in range(self.nb_optim_iters):
            super().optimizer_step(*args, **kwargs)

    def _dataloader(self) -> DataLoader:
        """Initialize the Replay Buffer dataset used for retrieving experiences"""
        dataset = ExperienceSourceDataset(self.train_batch)
        dataloader = DataLoader(dataset=dataset, batch_size=self.batch_size)
        return dataloader

    def train_dataloader(self) -> DataLoader:
        """Get train loader"""
        return self._dataloader()

    def make_move(self, state: tuple, hn, device=None) -> Tuple[int, torch.Tensor]:
        """
        Make a move in the environment using the trained policy network
        state: state of the environment
        hn: hidden state of the network
        device: device to run the network on
        """
        if device:
            self.to(device)
        state = self.convert_state_to_tensor(state)
        pi, action, hn = self.actor(state, hn)
        return action, hn
