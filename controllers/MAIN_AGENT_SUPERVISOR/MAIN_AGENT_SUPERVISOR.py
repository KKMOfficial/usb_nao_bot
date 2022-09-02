#  still needs debugging.
#  documentation completed.
#  code cleaned and refactored.
# E:\4_Installed_Softwares\Python3.9\python.exe E:\SBU\Semester8\FinalProject\HIVE_MIND\controllers\MAIN_AGENT_SUPERVISOR\MAIN_AGENT_SUPERVISOR.py

import sys,os
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
import ALGORITHMS
import SUPERVISOR
import UTILITY
import PARAMS

if __name__=='__main__':

    # define supervisor
    nao_supervisor = SUPERVISOR.Nao()

    # Instantiating DDPG algorithm to train the agent controller.
    # To improve the algorithm and reduce forgetting, reduce critic deduction error by initiating agent with random poses.
    # ddpg_alg  = ALGORITHMS.DDPGV2(
    #             supervisor = nao_supervisor,
    #             load_target_actor_from_disk=PARAMS.LOAD_TARGET_ACTOR_NETWORK_FROM_DISK,
    #             target_actor_file_address=PARAMS.TARGET_ACTOR_NETWORK_BACKUP_ADDRESS,
    #             actor_layers = PARAMS.ACTOR_STRUCTURE,
    #             actor_dropout = PARAMS.ACTOR_DROPOUT,
    #             load_target_critic_from_disk=PARAMS.LOAD_TARGET_CRITIC_NETWORK_FROM_DISK,
    #             target_critic_file_address=PARAMS.TARGET_CRITIC_NETWORK_BACKUP_ADDRESS,
    #             critic_layers = PARAMS.CRITIC_STRUCTURE,
    #             critic_dropout = PARAMS.CRITIC_DROPOUT,
    #             actor_learning_rate = PARAMS.ACTOR_LEARNING_RATE,
    #             critic_learning_rate = PARAMS.CRITIC_LEARNING_RATE,
    #             gamma = PARAMS.GAMMA,
    #             tau = PARAMS.TAU,
    #             max_memory_size=PARAMS.MAX_BUFFER_SIZE,
    #             noise_params=PARAMS.NOISE_LIST[0],
    #             training_mode=PARAMS.TRAINING_MODE,
    #             actor_normalize=PARAMS.ACTOR_NOMALIZE,
    #             critic_normalize=PARAMS.CRITIC_NORMALIZE,
    #             load_buffer_from_file = PARAMS.LOAD_BUFFER_FROM_FILE,
    #             buffer_file_address = PARAMS.BUFFER_BACKUP_ADDRESS,
    #             load_history_from_file=PARAMS.LOAD_HISTORY_FROM_FILE,
    #             history_file_address=PARAMS.HISTORY_BACKUP_ADDRESS)
    ppo_alg = ALGORITHMS.PPOV1(
            supervisor = nao_supervisor,
            m_net_units=PARAMS.MU_NET_STRUCTURE,
            m_net_drop_out_prob=PARAMS.MU_NET_DROPOUT,
            m_net_normalize=PARAMS.ACTOR_NOMALIZE,
            s_net_units=PARAMS.SIGMA_NET_STRUCTURE,
            s_net_drop_out_prob=PARAMS.SIGMA_NET_DROPOUT,
            s_net_actor_normalize=PARAMS.ACTOR_NOMALIZE,
            actor_learning_rate = PARAMS.ACTOR_LEARNING_RATE,
            critic_layers = PARAMS.CRITIC_STRUCTURE,
            critic_dropout = PARAMS.CRITIC_DROPOUT,
            critic_normalize = PARAMS.CRITIC_NORMALIZE,
            critic_learning_rate = PARAMS.CRITIC_LEARNING_RATE,
            gamma = PARAMS.GAMMA,
            load_target_actor_from_disk = PARAMS.LOAD_TARGET_ACTOR_NETWORK_FROM_DISK,
            target_actor_file_address = PARAMS.TARGET_ACTOR_NETWORK_BACKUP_ADDRESS,
            load_target_critic_from_disk = PARAMS.LOAD_TARGET_CRITIC_NETWORK_FROM_DISK,
            target_critic_file_address = PARAMS.TARGET_CRITIC_NETWORK_BACKUP_ADDRESS,
            training_mode=PARAMS.TRAINING_MODE,
            horizon = PARAMS.HORIZON,
            episodes = PARAMS.NUM_OF_EPISODES,
            clip_ratio=PARAMS.CLIP_RATIO,)

    # train actor
    if PARAMS.TRAINING_MODE:
        # ddpg_alg.train()
        ppo_alg.train()

    # evaluate actor
    else:
        # ddpg_alg.supervisor.__eval__()
        ppo_alg.supervisor.__eval__()

                