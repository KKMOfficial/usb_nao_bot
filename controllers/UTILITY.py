from multiprocessing import Process
import os,sys,numpy as np, torch,subprocess,random,time,copy,json,requests
from collections import deque
import torch.nn as nn
from math import sqrt
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
import PARAMS
import MODEL

class ReplayBuffer(object):
    def __init__(self, buffer_size, tuple_size):
        self.buffer_size = buffer_size
        self.tuple_size = tuple_size
        self.count = 0
        self.buffer = deque()        
    def add(self, experience):
        if self.count < self.buffer_size: 
            self.buffer.append(experience)
            self.count += 1
        else:
            self.buffer.popleft()
            self.buffer.append(experience)
    def size(self):
        return self.count
    def sample_batch(self, batch_size, random_sampling=True):
        if random_sampling:
            if self.count < batch_size:
                samples = random.sample(self.buffer, self.count)
            else:
                samples = random.sample(self.buffer, batch_size)
        
        else:
            if self.count < batch_size:
                samples = list(self.buffer[-self.count:])
            else:
                samples = list(self.buffer[-batch_size:])

        if self.tuple_size!=-1:
            batch = [None for _ in range(self.tuple_size)]
            # arrange batch groups
            for group in range(self.tuple_size):
                batch[group] = np.array([_[group] for _ in samples])
            return batch
        
        return samples
    def clear(self):
        self.buffer.clear()
        self.count = 0
    def set_size(self, new_size):
        self.buffer_size = new_size
    def list_add(self, list_of_samples):
        if self.tuple_size!=-1:
            for experience in list_of_samples:
                self.add(experience)
        else:
            self.add(list_of_samples)
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
        return np.clip(action + ou_state, 0.0, 1.0)

def list_to_1d_np_array(inList):
    return_value = []
    for element in inList:
        if type(element) is list:
            return_value += element
        else:
            return_value += [element]
    return np.array(return_value)
def partial_reward(state):
    # make sure it moves!
    righ_foot_force = sqrt(np.dot(np.array(state['RightFootForce']),np.array(state['RightFootForce'])))
    left_foot_force = sqrt(np.dot(np.array(state['LeftFootForce']),np.array(state['LeftFootForce'])))
    force_cost = -np.clip(a=(righ_foot_force+left_foot_force)/100,a_min=0,a_max=1)
    # make sure it lives!
    alive_bonus = 5
    # make sure it moves correctly!
    lin_vel_cost = 5000 * state['Velocity'][0]
    prep_vel_cost = 0#-1000 * abs(state['Velocity'][1])
    # control cost
    control_cost = -0.01*np.array(state['ControlForce']).sum()

    return lin_vel_cost + alive_bonus + prep_vel_cost + force_cost + control_cost,[lin_vel_cost,prep_vel_cost,force_cost,control_cost,righ_foot_force,left_foot_force,lin_vel_cost + alive_bonus + prep_vel_cost + force_cost + control_cost]

def convert_to_motor_valid_representation(input, robot, lcoeff=PARAMS.LOWER_MOTOR_MULTIPLIER, ucoeff=PARAMS.UPPER_MOTOR_MULTIPLIER):
    # x is always in (0,1) range
    fin_val = []
    for x,(motor_name,motor) in zip(input,robot.__getActuators__().items()):
        l = motor.getMinPosition()/lcoeff
        u = motor.getMaxPosition()/ucoeff
        fin_val += [l+x*(u-l)]
    return fin_val
def __run_external_controller__():
      subprocess.run(["start", "/wait", "cmd", "/K", "E:\\4_Installed_Softwares\\Python3.9\\python.exe E:\\SBU\\Semester8\\FinalProject\\HIVE_MIND\\controllers\\WORKER_AGENT_SUPERVISOR\\WORKER_AGENT_SUPERVISOR.py"], shell=True)
def __start_simulation__(worker_list):
    for worker in worker_list:        
        Process(target=__run_external_controller__, args=()).start()
        print('{worker_url} is online...'.format(worker_url=worker))
def update_target_network_parameters(tau, network, target_network): 
    """Will update controller target networks according to paper

    Args:
        tau (float): Tau parameter used to update target networks, refer to paper for more information.
        network (tensor): trained network which will be used to update target network.
        target_network (tensor): stable network which will be trained using trained unstable network.
    """
    for target_param, param in zip(target_network.parameters(), network.parameters()):
        target_param.data.copy_(param.data * tau + target_param.data * (1.0 - tau))
def sample_batch_request(worker_url,multi_thread_results):
    """Request to worker agent to gather samples and return them as list of elements of form (s,a,r,s2).

    Args:
        worker_url (string): Worker url of form "url/port".
    """
    # Make a request to worker controller to gather samples with described number of episodes
    r = requests.post(worker_url+'/generate_samples', json=json.dumps({'episodes':PARAMS.WORKER_EPISODES}), timeout=None)
    if r.status_code != 200: raise Exception('unsuccessfull <generate_samples> request1!')

    # Store request result inside global buffer
    multi_thread_results[multi_thread_results.index(None)]=(r.status_code, json.loads(r.text))


