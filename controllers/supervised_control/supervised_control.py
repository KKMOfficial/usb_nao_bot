"""supervised_control controller."""
import nao_superviser as ns, numpy as np, sys,matplotlib.pyplot as plt, requests, json,time
import nao_rl_framework as nrlf
from flask import jsonify

if __name__=='__main__':

    ddpgAgent = nrlf.DDPGagent(ns.Nao(),
                            actor_layers=[],
                            critic_layers=[],
                            actor_learning_rate=1e-4,
                            critic_learning_rate=1e-3,
                            gamma=0.99,
                            tau=1e-2,
                            max_memory_size=50000)


    # ddpgAgent.agent.__updateSimulator__()
    # ddpgAgent.agent.__resumeFastSimulation__()
    # ddpgAgent.agent.__stepSimulaiton__()
    # print(ddpgAgent.agent.__jointsEstimatedRotation__())
    ddpgAgent.agent.__stopSimulation__()
    # ddpgAgent.agent.__quitSimulation__()
    # ddpgAgent.agent.__robotCleanUp__()


    # ddpgAgent.agent.__storeCurrentState__()
    # r = requests.post('http://localhost:123/explore', json=json.dumps({'state':nrlf.list_to_1d_np_array(ddpgAgent.agent.__getLastState__().values()),'action':[1,2,3]}), timeout=None)
    # print(r.status_code)
    # print(r.text)

    # noise = nrlf.OUNoise(action_space={},
    #                     mu=0.0,
    #                     theta=0.15,
    #                     max_sigma=0.3,
    #                     min_sigma=0.3,
    #                     decay_period=100000)

    # batch_size = 128
    # rewards = []
    # avg_rewards = []

    # for episode in range(50):
    #     # define new random target point here
    #     ddpgAgent.agent.__resetSimulation__()
    #     # stablize robot in the environment (simple wait for some time)

    #     # observe environment
    #     state = nrlf.dictionary_flatter(ddpgAgent.agent.__observe__())
    #     noise.reset()
    #     episode_reward = 0
        
    #     for step in range(500):
    #         action = ddpgAgent.agent.get_action(state)
    #         action = noise.get_action(action, t=step)
    #         # new_state, reward, done, _ = env.step(action) 
    #         ddpgAgent.agent.__act__(command=action)
    #         ddpgAgent.agent.__updateSimulator__()
    #         ddpgAgent.agent.__resumeFastSimulation__()
    #         ddpgAgent.agent.__stepSimulaiton__()
    #         ddpgAgent.agent.__stopSimulation__()
    #         new_state = nrlf.dictionary_flatter(ddpgAgent.agent.__observe__())
    #         reward, done = ddpgAgent.agent.__reward__()
    #         ddpgAgent.memory.add(state, action, reward, done, new_state)
            
    #         if len(ddpgAgent.memory) > batch_size:
    #             ddpgAgent.update(batch_size)        
            
    #         state = new_state
    #         episode_reward += reward

    #         if done:
    #             sys.stdout.write("episode: {}, reward: {}, average _reward: {} \n".format(episode, np.round(episode_reward, decimals=2), np.mean(rewards[-10:])))
    #             break

    #     rewards.append(episode_reward)
    #     avg_rewards.append(np.mean(rewards[-10:]))

    # plt.plot(rewards)
    # plt.plot(avg_rewards)
    # plt.plot()
    # plt.xlabel('Episode')
    # plt.ylabel('Reward')
    # plt.show()