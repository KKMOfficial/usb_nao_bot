from collections import deque
import numpy as np, random, model, copy, torch.nn as nn,torch.optim as optim,torch

max_memory_size = 0
actor_learning_rate = 0
critic_learning_rate = 0

class ReplayBuffer(object):
    def __init__(self, buffer_size):
        self.buffer_size = buffer_size
        self.count = 0
        self.buffer = deque()
    def add(self, s, a, r, t, s2):
        experience = (s, a, r, t, s2)
        if self.count < self.buffer_size: 
            self.buffer.append(experience)
            self.count += 1
        else:
            self.buffer.popleft()
            self.buffer.append(experience)
    def size(self):
        return self.count
    def sample_batch(self, batch_size):
        batch = []

        if self.count < batch_size:
            batch = random.sample(self.buffer, self.count)
        else:
            batch = random.sample(self.buffer, batch_size)

        s_batch = np.array([_[0] for _ in batch])
        a_batch = np.array([_[1] for _ in batch])
        r_batch = np.array([_[2] for _ in batch])
        t_batch = np.array([_[3] for _ in batch])
        s2_batch = np.array([_[4] for _ in batch])

        return s_batch, a_batch, r_batch, t_batch, s2_batch
    def clear(self):
        self.buffer.clear()
        self.count = 0


class OUNoise(object):
    def __init__(self, action_space, mu=0.0, theta=0.15, max_sigma=0.3, min_sigma=0.3, decay_period=100000):
        self.mu           = mu
        self.theta        = theta
        self.sigma        = max_sigma
        self.max_sigma    = max_sigma
        self.min_sigma    = min_sigma
        self.decay_period = decay_period
        self.action_dim   = action_space['dim']
        self.low          = action_space['low']
        self.high         = action_space['high']
        self.reset()       
    def reset(self):
        self.state = np.ones(self.action_dim) * self.mu        
    def evolve_state(self):
        x  = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(self.action_dim)
        self.state = x + dx
        return self.state    
    def get_action(self, action, t=0):
        ou_state = self.evolve_state()
        self.sigma = self.max_sigma - (self.max_sigma - self.min_sigma) * min(1.0, t / self.decay_period)
        return np.clip(action + ou_state, self.low, self.high)


class DDPGagent:
    def __init__(self, robot, actor_layers=[], critic_layers=[], actor_learning_rate=1e-4, critic_learning_rate=1e-3, gamma=0.99, tau=1e-2, max_memory_size=50000):
        # Params
        self.agent = robot
        self.dimOfStates = len(dictionary_flatter(self.agent.__observe__()))
        self.dimOfActions = len(self.agent.Motors)
        self.tau = tau
        self.gamma = gamma 

        # do modificions afterwards with respect to the main paper
        self.critic = model.MLP(units=[self.dimOfActions+self.dimOfStates]+critic_layers+[self.dimOfActions], hidden_layer_activation='relu', init_type='default', drop_out_prob=0.0)
        self.critic_target = copy.deepcopy(self.critic)

        self.actor = model.MLP(units=[self.dimOfStates]+actor_layers+[self.dimOfActions], hidden_layer_activation='relu', init_type='default', drop_out_prob=0.0)
        self.actor_target = copy.deepcopy(self.actor)

        # training
        self.memory = ReplayBuffer(max_memory_size)  
        self.critic_criterion = nn.MSELoss()
        self.actor_optimizer  = optim.Adam(self.actor.parameters(), lr=actor_learning_rate)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_learning_rate)

    def get_action(self, state):
        return self.actor.forward(state).detach().numpy()[0,0]
    
    def update_target_network_parameters(tau, network, target_network): 
        for target_param, param in zip(target_network.parameters(), network.parameters()):
            target_param.data.copy_(param.data * tau + target_param.data * (1.0 - tau))

    def update(self, batch_size):
        s_batch, a_batch, r_batch, _, s2_batch = self.replayBuffer.sample_batch(batch_size)
        states = torch.FloatTensor(s_batch)
        actions = torch.FloatTensor(a_batch)
        rewards = torch.FloatTensor(r_batch)
        next_states = torch.FloatTensor(s2_batch)

        # Critic loss  
        Qvals = self.critic.forward(states, actions)
        next_actions = self.actor_target.forward(next_states)
        next_Q = self.critic_target.forward(next_states, next_actions.detach())
        Qprime = rewards + self.gamma * next_Q
        critic_loss = self.critic_criterion(Qvals, Qprime)

        # Actor loss
        policy_loss = -self.critic.forward(states, self.actor.forward(states)).mean()

        # update networks
        self.actor_optimizer.zero_grad()
        policy_loss.backward()
        self.actor_optimizer.step()

        self.critic_optimizer.zero_grad()
        critic_loss.backward() 
        self.critic_optimizer.step()
        
        # update target networks
        self.update_target_network_parameters(self.tau, self.actor, self.actor_target)
        self.update_target_network_parameters(self.tau, self.critic, self.critic_target)

def dictionary_flatter(inList):
    flat_list = []
    for sublist in list(inList):
        flat_list += sublist if type(sublist)==list else [sublist]
    return flat_list


