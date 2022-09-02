from cmath import nan
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
from threading import Thread
import numpy as np,os,sys, copy, torch.nn as nn,torch.optim as optim,torch, pickle,requests,json,pandas as pd ,time,math,random
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
import MODEL
import UTILITY
import PARAMS
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

# buffer to store intermediate results from workers
multi_thread_results = []
# tensorboard counter
# enter latest critic loss value here
episode_log = PARAMS.TRAIN_LOG_START
# enter latest simulation info counter here 
episode_bias = PARAMS.SIMULATOR_LOG_START
# minimum trajectory batch size to be used in training
min_trajectory_step = PARAMS.MIN_TRAJECTORY_STEP

# logger
summary_writer = SummaryWriter(PARAMS.TENSORBOARD_SUMMARY_FOLDER)


class DDPGV2:
    """_summary_
    second version of the algorithm, improvements: 
    - random state initialization to prevent forgetting 
    - online memory to stablize training 
    - reuseable enhanced states to use in offline training 
    - optimize policy given states are independant, ordering does not matter!
    - requires good strategy for exploration/exploitation, epsilon greedy is inefficient and slow
    - update steps get ruined after first few thausand steps
    """ 
    def __init__(self, 
    supervisor, 
    load_target_actor_from_disk=False,
    target_actor_file_address=None,
    actor_layers=[],
    actor_dropout=[], 
    actor_normalize=True,
    actor_learning_rate=1e-4, 
    load_target_critic_from_disk=False,
    target_critic_file_address=None,
    critic_layers=[], 
    critic_dropout=[], 
    critic_normalize=True,
    critic_learning_rate=1e-3, 
    gamma=0.99, 
    tau=1e-2, 
    max_memory_size=PARAMS.MAX_BUFFER_SIZE, 
    noise_params={}, 
    training_mode=True, 
    load_buffer_from_file = False,
    buffer_file_address = None,
    load_history_from_file=False,
    history_file_address=None,
    batch_size = PARAMS.BATCH_SIZE,
    horizon = PARAMS.HORIZON,
    episodes = PARAMS.NUM_OF_EPISODES,
    use_simulator = PARAMS.USE_SIMULATOR):

        # Params
        self.agent = supervisor
        self.dimOfStates = len(UTILITY.list_to_1d_np_array(self.agent.__getCurrentState__().values()).tolist())
        self.dimOfActions = len(self.agent.__getActuators__())
        self.tau = tau
        self.gamma = gamma
        self.goalCoordinate = np.array([ 0, 10, 0.332209])
        self.initCoordinate = np.array([ 0,  0, 0.332209])
        self.use_simulator = use_simulator
        UTILITY.ALGORITHM = 'DDPGV2'


        # init actor target network
        if load_target_actor_from_disk:
            self.actor_target = torch.load(target_actor_file_address)
        else:
            self.actor_target = MODEL.MLP(units=[self.dimOfStates]+actor_layers+[self.dimOfActions], hidden_layer_activation=nn.ReLU(), drop_out_prob=[0.0]+actor_dropout+[0.0],optional_last_layer=nn.Sigmoid(),batch_normalization=actor_normalize)
        
        # init critic target network
        if load_target_critic_from_disk:
            self.critic_target = torch.load(target_critic_file_address)
        else:
            self.critic_target = MODEL.MLP(units=[self.dimOfActions+self.dimOfStates]+critic_layers+[1], hidden_layer_activation=nn.ReLU(), drop_out_prob=[0.0]+critic_dropout+[0.0],batch_normalization=critic_normalize)
        
        # init actor and critic networks
        self.actor = copy.deepcopy(self.actor_target)
        self.critic = copy.deepcopy(self.critic_target)


        if training_mode:

            # init population
            self.workers = PARAMS.WORKERS

            # use actor and critic in agent
            self.agent.actor = self.actor_target
            self.agent.critic = self.critic_target

            # store copy of networks on the disk
            torch.save(self.actor, target_actor_file_address)
            torch.save(self.critic, target_critic_file_address)

            # set algorithm parameters
            self.batch_size = batch_size
            self.episodes = episodes
            self.horizon = horizon
            
            # training
            # transition buffer for offline learning with large capacity
            if load_buffer_from_file:
                with open(buffer_file_address,"rb") as f:
                    self.memory = pickle.load(f)
            else :
                self.memory = UTILITY.ReplayBuffer(buffer_size=max_memory_size,tuple_size=5)
            # transition buffer for online leraning with low capacity
            self.online_memory = UTILITY.ReplayBuffer(buffer_size=PARAMS.ONLINE_BUFFER_SIZE,tuple_size=5)


            # noise
            self.noise = UTILITY.OUNoise(action_space={'dim':len(self.agent.__getActuators__()),'low':PARAMS.ACTOR_MIN_OUTPUT,'high':PARAMS.ACTOR_MAX_OUTPUT},
                    mu=noise_params['mu'],              # mean of the process
                    theta=noise_params['theta'],        # frequency
                    max_sigma=noise_params['sigma'],    # min volatility 
                    min_sigma=noise_params['sigma'],    # max volatility 
                    decay_period=PARAMS.DECAY_PERIOD)

            # history
            if load_history_from_file:
                with open(history_file_address,"rb") as f:
                    self.rewards = pickle.load(f)
            else:
                self.rewards = pd.DataFrame({
                    'total_reward':[],
                    'min_partial_rewards':[],
                    'max_partial_rewards':[],
                    'average_rewards':[],
                    'total_time_steps':[],
                    'total_distance':[],
                })

            # set training stuff
            self.trajectory = []
            self.critic_criterion = nn.MSELoss()
            self.actor_optimizer  = optim.Adam(self.actor.parameters(), lr=actor_learning_rate)
            self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_learning_rate)

        else:
            # use actor_target and critic_target in agent
            self.agent.actor = self.actor_target
            self.agent.critic = self.critic_target

            # set network modes to evaluation mode
            self.actor_target.eval()
            self.critic_target.eval()
    def update(self, batch_size, MEMORY_BUFFER):
        """Will update the networks according to paper.

        Args:
            batch_size (int): Size of samples batch to train the networks.
        """

        # set actor and critic to training mode
        self.actor.train()
        self.critic.train()
        self.actor_target.train()
        self.critic_target.train()

        batch = MEMORY_BUFFER.sample_batch(batch_size)
        states = torch.FloatTensor(batch[0])
        actions = torch.FloatTensor(batch[1])
        rewards = torch.FloatTensor(batch[2])
        next_states = torch.FloatTensor(batch[3])
        done_list = torch.FloatTensor(batch[4])

        
        # Critic loss  
        Qvals = self.critic.forward(torch.cat([states, actions], dim=1))
        next_actions = self.actor_target.forward(next_states)
        next_Q = self.critic_target.forward(torch.cat([next_states, next_actions.detach()], dim=1))
        Qprime = torch.reshape(rewards, [-1,1]) + (1 - torch.reshape(done_list, [-1,1])) * self.gamma * next_Q
        critic_loss = self.critic_criterion(Qvals, Qprime)

        # update critic network
        self.critic_optimizer.zero_grad()
        critic_loss.backward() 
        self.critic_optimizer.step()

        # Actor loss
        policy_loss = -self.critic.forward(torch.cat([states, self.actor.forward(states)], dim=1)).mean()

        # update actor network
        self.actor_optimizer.zero_grad()
        policy_loss.backward()
        self.actor_optimizer.step()

        # update target networks
        UTILITY.update_target_network_parameters(self.tau, self.actor, self.actor_target)
        UTILITY.update_target_network_parameters(self.tau, self.critic, self.critic_target)
    def train(self):

        time.sleep(1)
        # store agent's initial state to use later
        self.agent.supervisor.getFromDef(self.agent.supervisor.getName()).saveState(stateName='initialState')

        for episode in range(1,self.episodes):

            # update worker networks
            for worker,worker_url in PARAMS.WORKERS.items() :
                r = requests.post(worker_url+'/update_networks', json=json.dumps({'actor_network':PARAMS.TARGET_ACTOR_NETWORK_BACKUP_ADDRESS,'critic_network':PARAMS.TARGET_CRITIC_NETWORK_BACKUP_ADDRESS}), timeout=None)
                if r.status_code != 200: raise Exception('unsuccessfull <update_networks> request1!')

            # gather samples from workers
            global multi_thread_results
            multi_thread_results = [None for _ in range(len(PARAMS.WORKERS))]
            thread_list = [Thread(target=UTILITY.sample_batch_request, args=(worker_url,multi_thread_results,)) for worker,worker_url in PARAMS.WORKERS.items()]
            for thread in thread_list : thread.start()
            for thread in thread_list : thread.join()
            for result in multi_thread_results : self.online_memory.list_add(list_of_samples=result[1]['sample_list'])
            for result in multi_thread_results : self.memory.list_add(list_of_samples=result[1]['sample_list'])

            print('online buffer size is {}.'.format(self.online_memory.size()))
            # online agent training
            self.update(batch_size=self.batch_size,MEMORY_BUFFER=self.online_memory)
            # reset online memory
            self.online_memory.clear()

            # offline agent training
            self.update(batch_size=self.batch_size,MEMORY_BUFFER=self.memory)


            # log and back-up section
            performance_log = self.agent.__eval__()
            self.rewards = self.rewards.append(performance_log, ignore_index=True) 

            # backup variables
            if episode in PARAMS.BACKUP_CHECK_POINTES:
                torch.save(self.actor_target,PARAMS.TARGET_ACTOR_NETWORK_BACKUP_ADDRESS)
                torch.save(self.critic_target,PARAMS.TARGET_CRITIC_NETWORK_BACKUP_ADDRESS)
                with open('replay_buffer[size={},num={}].pkl'.format(self.memory.size(),episode/PARAMS.EPISODE_INTERVAL),'wb') as f:
                    pickle.dump(copy.deepcopy(self.memory), f)
                with open(PARAMS.HISTORY_BACKUP_ADDRESS,'wb') as f:
                    pickle.dump(self.rewards, f)

            # log current episode information
            print("episode: {}, reward: {}, average_reward: {}, walked_distance: {}, total_steps: {}, buffer_size: {}, max_reward: {}, min_reward: {}, average_episode_reward: {} \n".format(
                episode, 
                np.round(performance_log['total_reward'], decimals=2), 
                np.mean(self.rewards['total_reward'][-10:]), 
                performance_log['total_distance'], 
                performance_log['total_time_steps'], 
                self.memory.size(), 
                performance_log['max_partial_rewards'],
                performance_log['min_partial_rewards'], 
                performance_log['average_rewards']))
    

class TD3V1:
    def __init__(self,
                supervisor,
                actor_layers=[],
                actor_dropout=[], 
                actor_normalize=True,
                actor_learning_rate=1e-4,
                critic_layers=[], 
                critic_dropout=[], 
                critic_normalize=True,
                critic_learning_rate=1e-3,
                max_memory_size=PARAMS.MAX_BUFFER_SIZE,
                episodes=PARAMS.NUM_OF_EPISODES,
                noise_clip = PARAMS.NOISE_CLIP,
                training_mode=PARAMS.TRAINING_MODE,
                policy_noise = PARAMS.POLICY_NOISE,
                policy_freq = PARAMS.POLICY_FREQUENCY,
                gamma=0.99, 
                tau=1e-2, 
                noise_params={},):
                
        self.agent = supervisor

        if training_mode:
            # TD3 parameters
            self.noise_clip = noise_clip
            self.policy_noise = policy_noise
            self.policy_freq = policy_freq

            self.actor = MODEL.MLP(units=[self.dimOfStates]+actor_layers+[self.dimOfActions], hidden_layer_activation=nn.ReLU(), drop_out_prob=[0.0]+actor_dropout+[0.0],optional_last_layer=nn.Tanh(),batch_normalization=actor_normalize)
            self.critic1 = MODEL.MLP(units=[self.dimOfActions+self.dimOfStates]+critic_layers+[1], hidden_layer_activation=nn.ReLU(), drop_out_prob=[0.0]+critic_dropout+[0.0],batch_normalization=critic_normalize)
            self.critic2 = MODEL.MLP(units=[self.dimOfActions+self.dimOfStates]+critic_layers+[1], hidden_layer_activation=nn.ReLU(), drop_out_prob=[0.0]+critic_dropout+[0.0],batch_normalization=critic_normalize)
            self.actor_target = copy.deepcopy(self.actor)
            self.critic1_target = copy.deepcopy(self.critic1)
            self.critic2_target = copy.deepcopy(self.critic2)

            self.memory = UTILITY.ReplayBuffer(max_memory_size)
            self.critic_criterion = nn.MSELoss()
            self.actor_optimizer  = optim.Adam(self.actor.parameters(), lr=actor_learning_rate)
            self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_learning_rate)
    def update(self, episode, batch_size, MEMORY_BUFFER):
        # set actor and critic to training mode
        self.actor.train()
        self.critic.train()
        self.actor_target.train()
        self.critic_target.train()

        # take a sample from the buffer
        s_batch, a_batch, r_batch, s2_batch = MEMORY_BUFFER.sample_batch(batch_size)
        states = torch.FloatTensor(s_batch)
        actions = torch.FloatTensor(a_batch)
        rewards = torch.FloatTensor(r_batch)
        next_states = torch.FloatTensor(s2_batch)

        # add clipped noise to the batch actions
        noise = torch.FloatTensor(a_batch).data.normal_(0, self.policy_noise)
        noise = noise.clamp(-self.noise_clip, self.noise_clip)
        actions = (self.actor_target(next_states) + noise).clamp(-self.max_action, self.max_action)

        # Compute the target Q value
        target_Q1 = self.critic1_target(next_states, actions)
        target_Q2 = self.critic2_target(next_states, actions)
        target_Q = torch.min(target_Q1, target_Q2)
        target_Q = torch.reshape(rewards, [-1,1]) + self.gamma * target_Q

        # Get current Q estimates
        current_Q1 = self.critic.forward(torch.cat([states, actions], dim=1))
        current_Q2 = self.critic.forward(torch.cat([states, actions], dim=1))

        # Compute critic loss
        critic_loss = self.critic_criterion(current_Q1, target_Q) + self.critic_criterion(current_Q2, target_Q)

        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        if episode % self.policy_freq == 0:

            # Compute policy loss
            policy_loss = -self.critic1.forward(torch.cat([states, self.actor.forward(states)], dim=1)).mean()

            # Optimize the actor 
            self.actor_optimizer.zero_grad()
            policy_loss.backward()
            self.actor_optimizer.step()

            # Update the frozen target models
            UTILITY.update_target_network_parameters(self.tau, self.actor, self.actor_target)
            UTILITY.update_target_network_parameters(self.tau, self.critic1, self.critic1_target)
            UTILITY.update_target_network_parameters(self.tau, self.critic2, self.critic2_target)
    def train(self):
        pass


class PPOV1:
    def __init__(self,
                supervisor,
                m_net_units,
                m_net_drop_out_prob,
                m_net_normalize,
                s_net_units,
                s_net_drop_out_prob,
                s_net_actor_normalize,
                actor_learning_rate,
                critic_layers,
                critic_dropout,
                critic_normalize,
                critic_learning_rate,
                gamma=0.99,
                clip_ratio=0.1,
                load_target_actor_from_disk=False,
                target_actor_file_address=None,
                load_target_critic_from_disk=False,
                target_critic_file_address=None,
                training_mode=True,
                horizon = PARAMS.HORIZON,
                episodes = PARAMS.NUM_OF_EPISODES,):
        # PPO parameters
        UTILITY.ALGORITHM = 'PPOV1'
        self.agent = supervisor
        self.dimOfStates = len(UTILITY.list_to_1d_np_array(self.agent.__getCurrentState__().values()).tolist())
        self.dimOfActions = len(self.agent.__getActuators__())
        self.gamma = gamma
        self.clip_ratio = clip_ratio

        # states, actions, rewards, values, logp
        # self.memory = UTILITY.ReplayBuffer(buffer_size=max_memory_size,tuple_size=5)

        # init actor target network
        if load_target_actor_from_disk:
            self.actor = torch.load(target_actor_file_address)
        else:
            self.actor = MODEL.StochActor(m_net_units=[self.dimOfStates]+m_net_units+[self.dimOfActions],
                                        m_net_hidden_layer_activation=nn.ReLU(),
                                        m_net_drop_out_prob=[0.0]+m_net_drop_out_prob+[0.0],
                                        m_net_normalize=m_net_normalize,
                                        m_net_optional_last_layer=nn.Sigmoid(),
                                        s_net_units=[self.dimOfStates]+s_net_units+[self.dimOfActions],
                                        s_net_hidden_layer_activation=nn.ReLU(),
                                        s_net_drop_out_prob=[0.0]+s_net_drop_out_prob+[0.0],
                                        s_net_actor_normalize=s_net_actor_normalize,
                                        s_net_optional_last_layer=nn.Sigmoid())
            torch.save(self.actor,PARAMS.TARGET_ACTOR_NETWORK_BACKUP_ADDRESS)
            print(self.actor)
        # init critic target network
        if load_target_critic_from_disk:
            self.critic = torch.load(target_critic_file_address)
        else:
            self.critic = MODEL.MLP(units=[self.dimOfStates]+critic_layers+[1], hidden_layer_activation=nn.ReLU(), drop_out_prob=[0.0]+critic_dropout+[0.0],batch_normalization=critic_normalize)
            torch.save(self.critic,PARAMS.TARGET_CRITIC_NETWORK_BACKUP_ADDRESS)
            print(self.critic)


        # raw reward counter, find more about it in train function log section
        self.data_log_counter = 0

        self.agent.actor = self.actor
        self.agent.critic = self.critic

        if training_mode:

            # init population
            self.workers = PARAMS.WORKERS

            # set algorithm parameters
            self.episodes = episodes
            self.horizon = horizon

            # memory buffer
            with open(PARAMS.BEST_SAMPLES_FILE_ADDRESS,'rb') as f:
                self.best_trajectory_memory = pickle.load(f)

            # set training stuff
            self.trajectory = []
            self.actor_optimizer  = optim.Adam(self.actor.parameters(), lr=actor_learning_rate)
            self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_learning_rate)
        else:
            # set network modes to evaluation mode
            self.actor.eval()
            self.critic.eval()   
    def update(self, MEMORY_BUFFER, current_episode=0):
        # perform update 
        # bring actor critic to gpu
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.actor.to(device)
        self.critic.to(device)

        # ORDER IS IMPORTANT!
        for index,episode in enumerate(MEMORY_BUFFER):
            # index for logger
            i = index + current_episode

            # each episode containes (state, action, reward, value, logp) tuples
            states = torch.nan_to_num(torch.as_tensor([experience[0] for experience in episode])).to(device)
            actions = torch.nan_to_num(torch.as_tensor([experience[1] for experience in episode])).to(device)
            rewards = np.nan_to_num(np.array([0.0 if np.isnan(experience[2]) else experience[2] for experience in episode]+[0.0]))
            values = np.nan_to_num(np.array([0.0 if np.isnan(experience[3]) else experience[3] for experience in episode]+[0.0]))
            logp_olds = torch.nan_to_num(torch.as_tensor([experience[4] for experience in episode])).to(device)

            if torch.isinf(states).any() or torch.isinf(actions).any() or torch.isinf(logp_olds).any() or np.isinf(rewards).any() or np.isinf(values).any():
                continue

            if torch.isnan(states).any() or torch.isnan(actions).any() or torch.isnan(logp_olds).any() or np.isnan(rewards).any() or np.isnan(values).any():
                continue

            # calculate advantage
            w = np.arange(rewards.size - 1)
            w = w - w[:, np.newaxis]
            w = np.triu(self.gamma ** w.clip(min=0)).T
            advantages = rewards[:-1] + self.gamma * values[1:] - values[:-1]
            advantages = (advantages.reshape(-1, 1) * w).sum(axis=0)

            # reward
            rewards_to_go = ((self.gamma*rewards[:-1]).reshape(-1, 1) * w).sum(axis=0)

            # normalize advantages
            advantages = torch.as_tensor((advantages - advantages.mean()) / np.std(advantages)).to(device)

            # update policy network
            # print('states = {}'.format(states))
            # print('actions = {}'.format(actions))
            _, logps = self.actor(states, actions)
            ratio = torch.exp(logps - logp_olds)

            # approx_kl = torch.mean(logp_olds-logps)         # a sample estimate for KL-divergence, easy to compute
            # approx_ent = pi.entropy().mean()                # a sample estimate for entropy, also easy to compute

            # summary_writer.add_histogram("approximate_kl_div",approx_kl, global_step=i)
            # summary_writer.add_histogram("approximate_entropy",approx_ent, global_step=i)
            
            clip_adv = torch.clamp(ratio, 1-self.clip_ratio, 1+self.clip_ratio) * advantages
            loss_pi = -(torch.min(ratio * advantages, clip_adv)).mean()

            # print('> loss_pi = {}'.format(loss_pi))
            if(torch.isinf(loss_pi) or torch.isnan(loss_pi)):
                continue

            # summary_writer.add_scalar('ratio',torch.clamp(ratio, 1-self.clip_ratio, 1+self.clip_ratio).mean(),i)
            self.actor_optimizer.zero_grad()
            loss_pi.backward()
            summary_writer.add_histogram("gradients/actor",torch.cat([p.data.view(-1) for p in self.actor.parameters()]), global_step=i)
            self.actor_optimizer.step()

            # update critic network
            loss_v = ((self.critic(states).squeeze(-1) - torch.as_tensor(rewards_to_go).to(device))**2).mean()

            # print('> loss_v = {}'.format(loss_v))
            if(torch.isinf(loss_v) or torch.isnan(loss_v)):
                continue

            self.critic_optimizer.zero_grad()
            loss_v.backward()
            summary_writer.add_histogram("gradients/critic",torch.cat([p.data.view(-1) for p in self.critic.parameters()]), global_step=i)
            self.critic_optimizer.step()

            # update training log list
            summary_writer.add_scalar('critic_loss',float(loss_v),i)
            # summary_writer.add_scalar('policy_loss',float(loss_pi),i)  
        # bring back actor critic to cpu
        device = torch.device('cpu')
        self.actor.to(device)
        self.critic.to(device)

    def train(self):
        global episode_log,episode_bias,min_trajectory_step
        time.sleep(1)

        # set actor and critic to training mode
        self.actor.train()
        self.critic.train()


        # store agent's initial state to use later
        self.agent.supervisor.getFromDef(self.agent.supervisor.getName()).saveState(stateName='initialState')
        if PARAMS.NETWORK_INITIAL_TRAINING:
            # print([tup[0] for tup in sorted_list[0:PARAMS.OFFLINE_EPISODES]])
            for offline_episodes in range(PARAMS.OFFLINE_ITERATIONS):
                with open(PARAMS.BEST_SAMPLES_FILE_ADDRESS,'rb') as f:
                    best_trajectory_memory = pickle.load(f)
                sorted_list = sorted([(len(episode),episode) for episode in best_trajectory_memory], key=lambda tup: tup[0], reverse=True)
                best_trajectory_memory = random.sample(sorted_list[0:PARAMS.OFFLINE_SUBSET],PARAMS.OFFLINE_EPISODES)
                log = np.array([x[0] for x in best_trajectory_memory])
                print('mean={}, std={}, list={}'.format(np.mean(log),np.std(log),log))
                self.update(MEMORY_BUFFER=[x[1] for x in best_trajectory_memory],current_episode=episode_log+offline_episodes)
                torch.save(self.actor,PARAMS.TARGET_ACTOR_NETWORK_BACKUP_ADDRESS)
                torch.save(self.critic,PARAMS.TARGET_CRITIC_NETWORK_BACKUP_ADDRESS)
                episode_log += PARAMS.OFFLINE_EPISODES+1
            exit()

        for episode in range(1,self.episodes):

            # update worker networks
            for worker,worker_url in PARAMS.WORKERS.items() :
                r = requests.post(worker_url+'/update_networks', json=json.dumps({'actor_network':PARAMS.TARGET_ACTOR_NETWORK_BACKUP_ADDRESS,'critic_network':PARAMS.TARGET_CRITIC_NETWORK_BACKUP_ADDRESS}), timeout=None)
                if r.status_code != 200: raise Exception('unsuccessfull <update_networks> request1!')

            # gather samples from workers
            global multi_thread_results
            multi_thread_results = [None for _ in range(len(PARAMS.WORKERS))]
            thread_list = [Thread(target=UTILITY.sample_batch_request, args=(worker_url,multi_thread_results,)) for worker,worker_url in PARAMS.WORKERS.items()]
            for thread in thread_list : thread.start()
            for thread in thread_list : thread.join()
            worker_batch = [result[1]['sample_list'] for result in multi_thread_results]
            log_info_batch = [result[1]['log_list'] for result in multi_thread_results]

            # make list of batches to train
            trajectory_batch = []
            for worker_episodes in worker_batch:
                for episode_list in worker_episodes:
                    trajectory_batch += [episode_list]

            # filter useless episodes
            trajectory_batch = [__episode for __episode in trajectory_batch if len(__episode)>min_trajectory_step]

            # update filter size dynamic
            # min_trajectory_step = min_trajectory_step+min(PARAMS.UPDATE_RADIUS,(len(trajectory_batch)-PARAMS.UPDATE_STEP)) if len(trajectory_batch)>PARAMS.UPDATE_STEP else min_trajectory_step

            # update filter size static
            min_trajectory_step = min_trajectory_step+3 if len(trajectory_batch)>PARAMS.UPDATE_STEP else min_trajectory_step

            print('min_trajectory_size = {}'.format(min_trajectory_step))

            # make list of batches to log
            trajectory_log_batch = []
            for log_list in log_info_batch:
                for log in log_list:
                    trajectory_log_batch += [list(log.values())]

            trajectory_values = list(np.array(copy.deepcopy(trajectory_log_batch)).mean(axis=0))
            trajectory_keys = list(log_info_batch[0][0].keys())
            log_dictionary = {k:v for (k,v) in zip(trajectory_keys,trajectory_values)}

            for _episode in trajectory_batch:
                if len(_episode) >= PARAMS.BEST_SAMPLE_THRESHOLD:
                    self.best_trajectory_memory += [_episode]
            with open(PARAMS.BEST_SAMPLES_FILE_ADDRESS,'wb') as f:
                pickle.dump(self.best_trajectory_memory, f)

            log = [len(_episode) for _episode in trajectory_batch]
            print('mean={}, std={}, list={}'.format(np.mean(log),np.std(log),log))
            
            print('>>{}'.format(len(self.best_trajectory_memory)))

            # agent update
            self.update(MEMORY_BUFFER=trajectory_batch,current_episode=episode_log+episode)

            # update log counter
            episode_log += len(trajectory_batch)

            # log network layers shape
            self.actor.mu_net.draw_layers(name='actor',summary_writer=summary_writer)
            self.critic.draw_layers(name='critic',summary_writer=summary_writer)

            # backup variables
            torch.save(self.actor,PARAMS.TARGET_ACTOR_NETWORK_BACKUP_ADDRESS)
            torch.save(self.critic,PARAMS.TARGET_CRITIC_NETWORK_BACKUP_ADDRESS)


            # log current episode information
            summary_writer.add_scalars(f'simulation/trajectory_info', {
                'total_distance':log_dictionary['total_distance'],
                'total_time_step':log_dictionary['total_time_steps'],
            }, episode_bias+episode)

            summary_writer.add_scalars(f'simulation/reward_info', {
                'total_reward':log_dictionary['total_reward'],
            }, episode_bias+episode)

            summary_writer.add_scalars(f'simulation/force_info', {
                'force_cost':log_dictionary['force_cost'],
                'righ_foot_force':log_dictionary['righ_foot_force'],
                'left_foot_force':log_dictionary['left_foot_force'],
            }, episode_bias+episode)
    