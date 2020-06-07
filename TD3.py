import copy
import math
import numpy as np
import torch
from torch.nn import Parameter, init
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Implementation of Twin Delayed Deep Deterministic Policy Gradients (TD3)
# Paper: https://arxiv.org/abs/1802.09477


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action, noise):
        super(Actor, self).__init__()
        self.noise = noise

        self.l1 = nn.Linear(state_dim, 256)
        self.l2 = nn.Linear(256, 256)
        # if self.noise:
        #     self.l3 = NoisyLinear(256, action_dim)
        # else:
        self.l3 = nn.Linear(256, action_dim)

        self.max_action = max_action

    def forward(self, state):
        a = F.relu(self.l1(state))
        a = F.relu(self.l2(a))
        return self.max_action * torch.tanh(self.l3(a))

    # def reset_noise(self):
    #     if self.noise:
            # self.l3.reset_noise()


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, dueling, noise):
        super(Critic, self).__init__()

        self.n_actions = action_dim
        self.dueling = dueling
        self.noise = noise

        # Q1 architecture
        self.l1 = nn.Linear(state_dim + action_dim, 256)
        self.l2 = nn.Linear(256, 256)
        if dueling:
            # We separate output stream into two streams
            # The one that calculates V(s)
            self.l3_1 = nn.Linear(256, 128)
            self.l4_1 = nn.Linear(128, 1)
            # The one that calculates A(s, a) - advantage
            self.l3_2 = nn.Linear(256, 128)
            self.l4_2 = nn.Linear(128, action_dim)
        elif noise:
            self.l3 = NoisyLinear(256, 1)
        else:
            self.l3 = nn.Linear(256, 1)

        # Q2 architecture
        self.l4 = nn.Linear(state_dim + action_dim, 256)
        self.l5 = nn.Linear(256, 256)
        if dueling:
            # We separate output stream into two streams
            # The one that calculates V(s)
            self.l6_1 = nn.Linear(256, 128)
            self.l7_1 = nn.Linear(128, 1)
            # The one that calculates A(s, a) - advantage
            self.l6_2 = nn.Linear(256, 128)
            self.l7_2 = nn.Linear(128, action_dim)
        elif noise:
            self.l6 = NoisyLinear(256, 1)
        else:
            self.l6 = nn.Linear(256, 1)

    def forward(self, state, action):
        sa = torch.cat([state, action], 1)

        q1 = F.relu(self.l1(sa))
        q1 = F.relu(self.l2(q1))
        if self.dueling:
            val = F.relu(self.l3_1(q1))
            val = self.l4_1(val)

            adv = F.relu(self.l3_2(q1))
            adv = self.l4_2(adv)
            # Q(s, a) = V(s) + A(s, a) - (1 / |A| * sumA(s, a'))
            q1 = val + adv - adv.mean(1).unsqueeze(1).expand(state.size(0), self.n_actions)
        else:
            q1 = self.l3(q1)

        q2 = F.relu(self.l4(sa))
        q2 = F.relu(self.l5(q2))
        if self.dueling:
            val = F.relu(self.l6_1(q2))
            val = self.l7_1(val)

            adv = F.relu(self.l6_2(q2))
            adv = self.l7_2(adv)
            # Q(s, a) = V(s) + A(s, a) - (1 / |A| * sumA(s, a'))
            q2 = val + adv - adv.mean(1).unsqueeze(1).expand(state.size(0), self.n_actions)
        else:
            q2 = self.l6(q2)

        return q1, q2

    def Q1(self, state, action, dueling):
        sa = torch.cat([state, action], 1)

        q1 = F.relu(self.l1(sa))
        q1 = F.relu(self.l2(q1))
        if dueling:
            val = F.relu(self.l3_1(q1))
            val = self.l4_1(val)

            adv = F.relu(self.l3_2(q1))
            adv = self.l4_2(adv)
            # Q(s, a) = V(s) + A(s, a) - (1 / |A| * sumA(s, a'))
            q1 = val + adv - adv.mean(1).unsqueeze(1).expand(state.size(0), self.n_actions)
        else:
            q1 = self.l3(q1)
        return q1

    def reset_noise(self):
        if self.noise:
            self.l3.reset_noise()
            self.l6.reset_noise()

        # for name, module in self.named_children():
        #     if 'fc' in name:
        #         module.reset_noise()


class TD3(object):
    def __init__(
            self,
            state_dim,
            action_dim,
            max_action,
            discount=0.99,
            tau=0.005,
            dueling=False,
            noisy=False,
            per=False,
            policy_noise=0.2,
            noise_clip=0.5,
            policy_freq=2
    ):

        self.actor = Actor(state_dim, action_dim, max_action, noisy).to(device)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=3e-4)

        self.critic = Critic(state_dim, action_dim, dueling, noisy).to(device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=3e-4)

        self.max_action = max_action
        self.discount = discount
        self.tau = tau
        self.dueling = dueling
        self.per = per
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_freq = policy_freq
        self.noisy = noisy

        self.total_it = 0

        print(f"Dueling: {dueling}, PER: {per}")

    def select_action(self, state):
        state = torch.FloatTensor(state.reshape(1, -1)).to(device)
        return self.actor(state).cpu().data.numpy().flatten()

    def train(self, replay_buffer, batch_size=100):
        self.total_it += 1

        # Sample replay buffer
        if self.per:
            idxs, state, action, reward, next_state, not_done, weights = replay_buffer.sample(batch_size)
        else:
            state, action, next_state, reward, not_done = replay_buffer.sample(batch_size)

        with torch.no_grad():
            # reseet
            if self.noisy:
                # self.actor_target.reset_noise()
                self.critic_target.reset_noise()
            # Select action according to policy and add clipped noise
            noise = (
                    torch.randn_like(action) * self.policy_noise
            ).clamp(-self.noise_clip, self.noise_clip)

            next_action = (
                    self.actor_target(next_state) + noise
            ).clamp(-self.max_action, self.max_action)

            # Compute the target Q value
            target_Q1, target_Q2 = self.critic_target(next_state, next_action)
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = reward + not_done * self.discount * target_Q

        # Get current Q estimates
        current_Q1, current_Q2 = self.critic(state, action)

        # Compute critic loss
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

        # Optimize the critic
        self.critic_optimizer.zero_grad()
        if self.per:
            (weights * critic_loss).mean().backward()  # Backpropagate importance-weighted minibatch loss
        else:
            critic_loss.backward()
        self.critic_optimizer.step()

        if self.per:
            # Update priorities of sampled transitions
            errors = np.abs((current_Q1 - target_Q).detach().cpu().numpy())
            replay_buffer.update_priorities(idxs, errors)

        # Delayed policy updates
        if self.total_it % self.policy_freq == 0:

            # Compute actor losse
            actor_loss = -self.critic.Q1(state, self.actor(state), self.dueling).mean()

            # Optimize the actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # Update the frozen target models
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    def save(self, filename):
        torch.save(self.critic.state_dict(), filename + "_critic")
        torch.save(self.critic_optimizer.state_dict(), filename + "_critic_optimizer")

        torch.save(self.actor.state_dict(), filename + "_actor")
        torch.save(self.actor_optimizer.state_dict(), filename + "_actor_optimizer")

    def load(self, filename):
        self.critic.load_state_dict(torch.load(filename + "_critic"))
        self.critic_optimizer.load_state_dict(torch.load(filename + "_critic_optimizer"))
        self.critic_target = copy.deepcopy(self.critic)

        self.actor.load_state_dict(torch.load(filename + "_actor"))
        self.actor_optimizer.load_state_dict(torch.load(filename + "_actor_optimizer"))
        self.actor_target = copy.deepcopy(self.actor)

    def reset_noise(self):
        if self.noisy:
            self.critic.reset_noise()


# Factorised NoisyLinear layer with bias
class NoisyLinear(nn.Module):
    def __init__(self, in_features, out_features, std_init=0.5):
        super(NoisyLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.std_init = std_init
        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
        self.register_buffer('weight_epsilon', torch.empty(out_features, in_features))
        self.bias_mu = nn.Parameter(torch.empty(out_features))
        self.bias_sigma = nn.Parameter(torch.empty(out_features))
        self.register_buffer('bias_epsilon', torch.empty(out_features))
        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self):
        mu_range = 1 / math.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.std_init / math.sqrt(self.in_features))
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.std_init / math.sqrt(self.out_features))

    def _scale_noise(self, size):
        x = torch.randn(size)
        return x.sign().mul_(x.abs().sqrt_())

    def reset_noise(self):
        epsilon_in = self._scale_noise(self.in_features)
        epsilon_out = self._scale_noise(self.out_features)
        self.weight_epsilon.copy_(epsilon_out.ger(epsilon_in))
        self.bias_epsilon.copy_(epsilon_out)

    def forward(self, input):
        if self.training:
            return F.linear(input, self.weight_mu + self.weight_sigma * self.weight_epsilon,
                            self.bias_mu + self.bias_sigma * self.bias_epsilon)
        else:
            return F.linear(input, self.weight_mu, self.bias_mu)
