from collections import deque
import numpy as np, random, model, copy, torch.nn as nn,torch.optim as optim,torch, requests,json,threading,os, pickle, time

max_memory_size = 0
actor_learning_rate = 0
critic_learning_rate = 0

WEBOTS_PATH = 'E:\\4_Installed_Softwares\\Webots\\msys64\\mingw64\\bin\\webots.exe'
EXPLORER_PATH = 'E:\\SBU\\Semester8\\FinalProject\\NAO\\worlds\\ExploreWorld20x20.wbt'
PORT_FILE_PATH = 'E:\\SBU\\Semester8\\FinalProject\\NAO\\data\\port_info.pkl'
URL_FILE_PATH = 'E:\\SBU\\Semester8\\FinalProject\\NAO\\data\\url_info.pkl'

class ReplayBuffer(object):
    def __init__(self, buffer_size):
        self.buffer_size = buffer_size
        self.count = 0
        self.buffer = deque()
    def add(self, s, a, r, s2):
        experience = (s, a, r, s2)
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
        self.dimOfStates = len(list_to_1d_np_array(self.agent.__observe__()))
        self.dimOfActions = len(self.agent.Motors)
        self.tau = tau
        self.gamma = gamma
        self.goalCoordinate = [3, 3, 0.332209]

        # do modificions afterwards with respect to the main paper
        self.critic = model.MLP(units=[self.dimOfActions+self.dimOfStates]+critic_layers+[self.dimOfActions], hidden_layer_activation='relu', init_type='default', drop_out_prob=0.0)
        self.critic_target = copy.deepcopy(self.critic)

        self.actor = model.MLP(units=[self.dimOfStates]+actor_layers+[self.dimOfActions], hidden_layer_activation='relu', init_type='default', drop_out_prob=0.0)
        self.actor_target = copy.deepcopy(self.actor)

        # training
        self.memory = ReplayBuffer(max_memory_size)
        self.trajectory = []
        self.critic_criterion = nn.MSELoss()
        self.actor_optimizer  = optim.Adam(self.actor.parameters(), lr=actor_learning_rate)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_learning_rate)

        # explorers initialization
        self.init_webot_process()

    def init_webot_process(self, number_of_explorers=1):
        with open (URL_FILE_PATH,'wb') as f:
            pickle.dump([],f)

        with open(PORT_FILE_PATH,'wb') as f:
            pickle.dump({'port_list':list(range(120, 1, 120+number_of_explorers+1)),'current_index':0}, f)

        for _ in range(number_of_explorers):
            os.system("{webots} --mode=fast --no-rendering --stdout --stderr --minimize --batch {explorer}".format(webots=WEBOTS_PATH, explorer=EXPLORER_PATH))
            time.sleep(5)

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

    def explore_request(self, host):
        # add other required values here!
        currnt_state = np.array(list_to_1d_np_array(self.agent.__getLastState__().values()))
        r = requests.post(host, json=json.dumps({'state':currnt_state,'action':self.get_action(currnt_state)}), timeout=None)
        print(r.status_code)
        print(r.text)
        return r

    def explore_env(self):
        with open(URL_FILE_PATH) as f:
            host_list = pickle.load(f)
        thread_list = [threading.Thread(target=self.explore_request, args=(host,)) for host in host_list]
        result_list = [thread.start() for thread in thread_list]
        for thread in thread_list: thread.join()
        print(len(result_list))

    def start_new_episode(self):
        # evaluate final trajectory value
        # state values stored in trajectory list are in form (s, a, r, t, s2)
        # state values stored in replybuffer list are in form (s, a, r, s2)
        for state in self.trajectory:
            state[2] += 25*state[3]/self.trajectory[-1][3]
            # store trajectory in replay buffer
            self.memory.add(state[0], state[1], state[2], state[3])
        # discard previous trajectory
        self.trajectory = []
    

def list_to_1d_np_array(inList):
    return [element for sublist in inList for element in sublist]

def partial_reward(next_coordinate, init_coordinate, goal_coordinate, prev_velocity):
    # reward_t = velocity_in_goal_direction
    # - 3*displacement_in_lateral_goal_direction**2 
    # - 50*center_of_mass_displacement 
    # + 25*total_episode_time_in_term_of_samples/total_episode_time \
    #             - 0.02*norm_2_squared_all_joint_velocity_previous_time
    init_goal = goal_coordinate[0:2] - init_coordinate[0:2]
    init_next = next_coordinate[0:2] - init_coordinate[0:2]
    cent_mass_disp = next_coordinate[2]-init_coordinate[2]
    goal_comp_norm = np.dot(init_next, init_goal)/np.linalg.norm(init_goal, ord=2)
    orth_goal_comp = init_next - goal_comp_norm*(init_goal/np.linalg.norm(init_goal, ord=2))
    orth_comp_norm = np.dot(init_next, orth_goal_comp)/np.linalg.norm(init_next, ord=2)
    prev_scnd_norm_jnts = np.linalg.norm(prev_velocity, ord=2)
    return goal_comp_norm\
         - 3*orth_comp_norm**2\
         - 50*cent_mass_disp\
         - 0.02*prev_scnd_norm_jnts**2
    



