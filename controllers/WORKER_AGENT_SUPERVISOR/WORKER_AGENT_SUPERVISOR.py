#  still needs debugging.
#  documentation completed.
#  code cleaned and refactored.

from flask import Flask,request,json
import random, numpy as np, torch
import os,sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
import SUPERVISOR
import UTILITY
import PARAMS

robot = SUPERVISOR.Nao()

if PARAMS.ALGORITHM=='DDPGV2':
    noise = UTILITY.OUNoise(action_space={'dim':len(robot.__getActuators__()),'low':PARAMS.ACTOR_MIN_OUTPUT,'high':PARAMS.ACTOR_MAX_OUTPUT},
                        mu=PARAMS.NOISE_LIST[0]['mu'],              # mean of the process
                        theta=PARAMS.NOISE_LIST[0]['theta'],        # frequency
                        max_sigma=PARAMS.NOISE_LIST[0]['sigma'],    # min volatility 
                        min_sigma=PARAMS.NOISE_LIST[0]['sigma'],    # max volatility 
                        decay_period=PARAMS.DECAY_PERIOD)

app = Flask(__name__)

@app.route("/update_networks",methods=['POST'])
def update_networks():
    """Will update worker actor and critic networks using provided arguments.

    Returns:
        dict: Emptry dictonary will be returned.
    """
    # initiate values
    req_in = json.loads(request.json)

    # update actor network 
    robot.actor = torch.load(req_in['actor_network'])

    # update critic network
    robot.critic = torch.load(req_in['critic_network'])

    # return empty dictonary
    return {}

@app.route("/generate_samples",methods=['POST'])
def generate_samples(episodes=None):
    """Will generate list of samples of form (s,a,r,s2) using specified actor and critic networks in the agent.

    Args:
        episodes (int): Number of the episodes to run worker and collect samples.

    Raises:
        KeyError: Input request must contain a dictionary with episodes:int element otherwise a KeyError will be raised.

    Returns:
        Dict: Dictionary with sample_list key and value equals to sample_list of form (s,a,r,s2) that will be used to trian the controller target networks.
    """
    # initiate values
    req_in = json.loads(request.json)
    robot.actor.eval()
    robot.critic.eval()
    if PARAMS.ALGORITHM=='DDPGV2':
        noise.reset()

    # load argument from input json request
    try : episodes = req_in['episodes']
    except : raise KeyError('Request must contain a dictoinary with episodes:int element.')

    trajectory_list = []
    log_list = []
    episodes_reward_list = []

    for _ in range(episodes):

        # Set the location and pose of the robot to initiate state
        robot.__setCurrentState__('initialState')

        if PARAMS.WORKER_RANDOM_INITIALIZATION==True:
            # We will make initial state random and undependent of previous trajectory to minimize forgetting and maximize exploration
            # uniform initialization is a good way to go. we will perform two random actions after initialization.
            if (random.randint(1,4)%2==0):
                for _ in range(random.randint(2,4)):
                    a = noise.get_action(np.random.rand(len(robot.__getActuators__())))
                    robot.__act__(a)
                    robot.__stepSimulaiton__()
            else:
                robot.__setRandomPosiiton__()

        # sample a trajectory
        trajectory,log_info,episode_steps_rewards_list = robot.__eval__()

        # store trajectory
        trajectory_list += [trajectory]
        log_list += [log_info]
        episodes_reward_list += [episode_steps_rewards_list]

    return {'sample_list':trajectory_list, 'log_list': log_list, 'list_of_episodes_steps_rewards' : episodes_reward_list}

def main():

    # Store initiate state to use in future sampling process
    robot.__stepSimulaiton__()
    
    # Store init position to ensure zero velocity and acceleration
    robot.supervisor.getFromDef(robot.supervisor.getName()).saveState(stateName='initialState')

    # Start the flask server app and wait for main_agent requests
    app.run(host='localhost', port=int(robot.supervisor.getCustomData()))

if __name__ == "__main__":
    main()