#  still needs debugging.
#  does not need documentation.
#  code cleaned and refactored.

import numpy as np
import os,sys,copy,torch
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
import PARAMS
import UTILITY
sys.path.append(PARAMS.WEBOTS_PYTHON_BIN)
from controller import Supervisor
from math import sqrt

class Nao():
    def __init__(self):
        self.supervisor = Supervisor()
        self.DEF = self.supervisor.getName()
        self.actor = None
        self.critic = None
        self.stateCheckPoint = None
        self.initCoordinate = np.array(self.supervisor.getFromDef(self.DEF).getField('translation').getSFVec3f())
        self.goalCoordinate = np.array([self.initCoordinate[0]+10, self.initCoordinate[1], self.initCoordinate[2]])
        og_vec = self.goalCoordinate-self.initCoordinate
        self.origin_goal = np.sqrt(np.dot(og_vec,og_vec))
        self.timestep = int(self.supervisor.getBasicTimeStep())
        self.__initMotors__()
        self.__initSensors__()
        self.Motors = self.__getActuators__()
        self.Sensors = self.__getSensors__()
        self.WB_SUPERVISOR_SIMULATION_MODE_PAUSE = 0
        self.SIMULATION_MODE_REAL_TIME = 1
        self.WB_SUPERVISOR_SIMULATION_MODE_FAST = 2
        self.EXIT_SUCCESS = 0
        self.EXIT_FAILURE = 1
        # these variables are used to determine current state
        # (using position to estimate velocity for current state)
        self.prevCoordinate = copy.deepcopy(self.initCoordinate)
        self.currentCoordinate = copy.deepcopy(self.initCoordinate)
        self.prevJointPose = copy.deepcopy(np.array(list(self.__getJointPose__().values())))
        self.currentJointPose = copy.deepcopy(np.array(list(self.__getJointPose__().values())))
        # store controls of the whole trajectory
        self.prevControls = [[0]*len(self.__getActuators__())]
    def __stopSimulation__(self):
        self.supervisor.simulationSetMode(self.WB_SUPERVISOR_SIMULATION_MODE_PAUSE)
    def __resumeSimulation__(self):
        self.supervisor.simulationSetMode(self.SIMULATION_MODE_REAL_TIME)
    def __resumeFastSimulation__(self):
        self.supervisor.simulationSetMode(self.WB_SUPERVISOR_SIMULATION_MODE_FAST)
    def __quitSimulation__(self):
        self.supervisor.simulationQuit(self.EXIT_SUCCESS)
    def __robotCleanUp__(self):
        self.supervisor.__del__()
    def __stepSimulaiton__(self, n=1):
        self.prevCoordinate = np.array(self.__getGPSValue__())
        self.prevJointPose = copy.deepcopy(np.array(list(self.__getJointPose__().values())))
        self.supervisor.step(n*self.timestep)
        self.currentCoordinate = np.array(self.__getGPSValue__())
        self.currentJointPose = copy.deepcopy(np.array(list(self.__getJointPose__().values())))
    def __getActuators__(self):
        return {
            # 'HeadPitch':self.supervisor.getDevice('HeadPitch'),
            # 'HeadYaw':self.supervisor.getDevice('HeadYaw'),

            'RShoulderRoll':self.supervisor.getDevice('RShoulderRoll'),
            'RShoulderPitch':self.supervisor.getDevice('RShoulderPitch'),
            'RElbowRoll':self.supervisor.getDevice('RElbowRoll'),
            'RElbowYaw':self.supervisor.getDevice('RElbowYaw'),
            'RWristYaw':self.supervisor.getDevice('RWristYaw'),

            'RHipYawPitch':self.supervisor.getDevice('RHipYawPitch'),
            'RHipPitch':self.supervisor.getDevice('RHipPitch'),
            'RHipRoll':self.supervisor.getDevice('RHipRoll'),
            'RKneePitch':self.supervisor.getDevice('RKneePitch'),
            'RAnklePitch':self.supervisor.getDevice('RAnklePitch'),
            'RAnkleRoll':self.supervisor.getDevice('RAnkleRoll'),

            'LShoulderRoll':self.supervisor.getDevice('LShoulderRoll'),
            'LShoulderPitch':self.supervisor.getDevice('LShoulderPitch'),
            'LElbowRoll':self.supervisor.getDevice('LElbowRoll'),
            'LElbowYaw':self.supervisor.getDevice('LElbowYaw'),
            'LWristYaw':self.supervisor.getDevice('LWristYaw'),

            'LHipYawPitch':self.supervisor.getDevice('LHipYawPitch'),
            'LHipPitch':self.supervisor.getDevice('LHipPitch'),
            'LHipRoll':self.supervisor.getDevice('LHipRoll'),
            'LKneePitch':self.supervisor.getDevice('LKneePitch'),
            'LAnklePitch':self.supervisor.getDevice('LAnklePitch'),
            'LAnkleRoll':self.supervisor.getDevice('LAnkleRoll'),
        }
    def __initMotors__(self):
        for motorName,motor in self.__getActuators__().items():
            motor.setPosition(0.0)
            motor.setVelocity(motor.getMaxVelocity())
            motor.enableTorqueFeedback(self.timestep)
    def __getSensors__(self):
          # 'CameraBottom':self.supervisor.getDevice('CameraBottom'),
          # 'CameraTop':self.supervisor.getDevice('CameraTop'),
        return {
            'GPS':self.supervisor.getDevice('gps'),
            'RFBL':self.supervisor.getDevice('RFBL'),
            'RFBR':self.supervisor.getDevice('RFBR'),
            'LFBL':self.supervisor.getDevice('LFBL'),
            'LFBR':self.supervisor.getDevice('LFBR'),
            'RFoot/ForceSensor':self.supervisor.getDevice('RFsr'),
            'LFoot/ForceSensor':self.supervisor.getDevice('LFsr'),
            'Accelerometer':self.supervisor.getDevice('accelerometer'),
            'HeadYawS':self.supervisor.getDevice('HeadYawS'),
            'HeadPitchS':self.supervisor.getDevice('HeadPitchS'),
            'RHipYawPitchS':self.supervisor.getDevice('RHipYawPitchS'),
            'RHipRollS':self.supervisor.getDevice('RHipRollS'),
            'RHipPitchS':self.supervisor.getDevice('RHipPitchS'),
            'RKneePitchS':self.supervisor.getDevice('RKneePitchS'),
            'RAnklePitchS':self.supervisor.getDevice('RAnklePitchS'),
            'RAnkleRollS':self.supervisor.getDevice('RAnkleRollS'),
            'LHipYawPitchS':self.supervisor.getDevice('LHipYawPitchS'),
            'LHipRollS':self.supervisor.getDevice('LHipRollS'),
            'LHipPitchS':self.supervisor.getDevice('LHipPitchS'),
            'LKneePitchS':self.supervisor.getDevice('LKneePitchS'),
            'LAnklePitchS':self.supervisor.getDevice('LAnklePitchS'),
            'LAnkleRollS':self.supervisor.getDevice('LAnkleRollS'),
            'RShoulderPitchS':self.supervisor.getDevice('RShoulderPitchS'),
            'RShoulderRollS':self.supervisor.getDevice('RShoulderRollS'),
            'RElbowYawS':self.supervisor.getDevice('RElbowYawS'),
            'RElbowRollS':self.supervisor.getDevice('RElbowRollS'),
            'RWristYawS':self.supervisor.getDevice('RWristYawS'),
            'LShoulderPitchS':self.supervisor.getDevice('LShoulderPitchS'),
            'LShoulderRollS':self.supervisor.getDevice('LShoulderRollS'),
            'LElbowYawS':self.supervisor.getDevice('LElbowYawS'),
            'LElbowRollS':self.supervisor.getDevice('LElbowRollS'),
            'LWristYawS':self.supervisor.getDevice('LWristYawS'),
            'IMU':self.supervisor.getDevice('inertial unit'),
            'Gyro':self.supervisor.getDevice('gyro'),
        }
    def __initSensors__(self):
        for sensorName,sensor in self.__getSensors__().items():
            sensor.enable(self.timestep)
    def __observe__(self):
        return self.__getJointPose__() | {
            # joint positions
            # center of mass height
            'COMZ':list(self.supervisor.getDevice('gps').getValues())[2],
            # external force
            'RightFootForce':list(self.supervisor.getDevice('RFsr').getValues()),
            'LeftFootForce':list(self.supervisor.getDevice('LFsr').getValues()),
            # center of mass angular velocity
            'Gyro':self.supervisor.getDevice('gyro').getValues()[0:2],
            # center of mass acceleration
            'Accelerometer':list(self.supervisor.getDevice('accelerometer').getValues()),
            # center of mass angle
            'IMU':[self.supervisor.getDevice('inertial unit').getRollPitchYaw()[0],self.supervisor.getDevice('inertial unit').getRollPitchYaw()[2]],
        }
    def __getCurrentState__(self):
        Ɛ = 0.0000001
        currentState = self.__observe__()

        init_goal = self.goalCoordinate - self.initCoordinate
        prev_goal = self.goalCoordinate - self.prevCoordinate
        prev_cur = self.currentCoordinate - self.prevCoordinate
        cur_goal = self.goalCoordinate - self.currentCoordinate
        # cur_init = self.initCoordinate - self.currentCoordinate

        prev_goal_unit = prev_goal/(np.sqrt(np.dot(prev_goal,prev_goal))+Ɛ)

        goal_comp_norm = np.dot(prev_cur, prev_goal_unit)
        orth_goal_vect = prev_cur - goal_comp_norm*prev_goal_unit
        orth_goal_unit = orth_goal_vect/(np.sqrt(np.dot(orth_goal_vect,orth_goal_vect))+Ɛ)
        orth_comp_norm = np.dot(prev_cur, orth_goal_unit)
        cent_mass_disp = self.currentCoordinate[2] - self.prevCoordinate[2]
        # cos_direction = np.dot(prev_cur,prev_goal)/((np.sqrt(np.dot(prev_cur,prev_cur))*np.sqrt(np.dot(prev_goal,prev_goal)))+Ɛ)

        s_d = goal_comp_norm/self.timestep
        p_d = orth_comp_norm/self.timestep
        n_d = cent_mass_disp/self.timestep

        # currentState['InitNormalGoalDirection']=list(init_goal/np.sqrt(np.dot(init_goal,init_goal)))
        # currentState['PrevNormalGoalDirection']=list(prev_goal/np.sqrt(np.dot(prev_goal,prev_goal)))
        current_goal_unit=list(cur_goal/np.sqrt(np.dot(cur_goal,cur_goal)))
        currentState['normalDirectionToGoal']=list((np.sqrt(np.dot(cur_goal,cur_goal))/np.sqrt(np.dot(init_goal,init_goal)))*np.array(current_goal_unit))
        # currentState['PrevNormalDistaceFromGoal']=np.sqrt(np.dot(prev_goal,prev_goal))/np.sqrt(np.dot(init_goal,init_goal))
        currentState['Velocity']=[s_d, p_d, n_d]
        currentState['JointVelocity']=list((self.currentJointPose-self.prevJointPose)/self.timestep)
        currentState['ControlForce']=list((np.array(self.prevControls[-PARAMS.EFFECTIVE_REWARDS:])**2).sum(axis=0))
        # currentState['cos_direction']=cos_direction
        return currentState
    def __getGPSValue__(self):
        # GPS update frequency is not right probably! 
        return self.Sensors['GPS'].getValues()
    def __getJointPose__(self):
        return {
            'RHipYawPitchS':self.supervisor.getDevice('RHipYawPitchS').getValue(),
            'RHipRollS':self.supervisor.getDevice('RHipRollS').getValue(),
            'RHipPitchS':self.supervisor.getDevice('RHipPitchS').getValue(),
            'RKneePitchS':self.supervisor.getDevice('RKneePitchS').getValue(),
            'RAnklePitchS':self.supervisor.getDevice('RAnklePitchS').getValue(),
            'RAnkleRollS':self.supervisor.getDevice('RAnkleRollS').getValue(),
            'LHipYawPitchS':self.supervisor.getDevice('LHipYawPitchS').getValue(),
            'LHipRollS':self.supervisor.getDevice('LHipRollS').getValue(),
            'LHipPitchS':self.supervisor.getDevice('LHipPitchS').getValue(),
            'LKneePitchS':self.supervisor.getDevice('LKneePitchS').getValue(),
            'LAnklePitchS':self.supervisor.getDevice('LAnklePitchS').getValue(),
            'LAnkleRollS':self.supervisor.getDevice('LAnkleRollS').getValue(),
            'RShoulderPitchS':self.supervisor.getDevice('RShoulderPitchS').getValue(),
            'RShoulderRollS':self.supervisor.getDevice('RShoulderRollS').getValue(),
            'RElbowYawS':self.supervisor.getDevice('RElbowYawS').getValue(),
            'RElbowRollS':self.supervisor.getDevice('RElbowRollS').getValue(),
            'RWristYawS':self.supervisor.getDevice('RWristYawS').getValue(),
            'LShoulderPitchS':self.supervisor.getDevice('LShoulderPitchS').getValue(),
            'LShoulderRollS':self.supervisor.getDevice('LShoulderRollS').getValue(),
            'LElbowYawS':self.supervisor.getDevice('LElbowYawS').getValue(),
            'LElbowRollS':self.supervisor.getDevice('LElbowRollS').getValue(),
            'LWristYawS':self.supervisor.getDevice('LWristYawS').getValue(),
            'HeadYawS':self.supervisor.getDevice('HeadYawS').getValue(),
            'HeadPitchS':self.supervisor.getDevice('HeadPitchS').getValue(),
        }
    def __act__(self, command, lcoeff=PARAMS.LOWER_MOTOR_MULTIPLIER, ucoeff=PARAMS.UPPER_MOTOR_MULTIPLIER):
        # store act inside control array
        self.prevControls += [list(command.squeeze())]
        # convert inputs in range(0,1) to valid motor value
        fin_val = []
        for x,(motor_name,motor) in zip(list(command.squeeze()),self.__getActuators__().items()):
            l = motor.getMinPosition()/lcoeff
            u = motor.getMaxPosition()/ucoeff
            fin_val += [l+x*(u-l)]
        command = fin_val
    
        # use valid values
        for index, motor in enumerate(self.__getActuators__().values()):
            motor.setPosition(float(command[index]))
    def __getAction__(self, state):
        if PARAMS.ALGORITHM=='DDPGV2':
            return {'action':self.actor.forward(state).detach().numpy()[:]}
        elif PARAMS.ALGORITHM=='PPOV1':
            with torch.no_grad():
                pi = self.actor._distribution(state)
                if PARAMS.ALGO_DETEMINISTIC:
                    return {'action':pi.mean.numpy()}
                else:
                    a = torch.clip(pi.sample(),0,1)
                logp_a = pi.log_prob(a).sum(axis=-1)
            return {'action':a.numpy(), 'logp_a':logp_a.numpy()}
    def __getValue__(self, state):
        # PPOV1 agent only
        return self.critic(state)
    def __episodeIsDone__(self, state):
        Ɛ = 0.0000001
        # check if robot has fallen
        IS_FALL = state['COMZ'] < PARAMS.MIN_Z_FALL
        # current normal distance from goal
        normal_goal_distance = np.sqrt(np.dot(state['normalDirectionToGoal'],state['normalDirectionToGoal']))
        # check if robot got too far from goal
        # IS_WONDERED = normal_goal_distance > PARAMS.MAX_INIT_GOAL_RADIUS
        # reach the goal 
        IS_REACHED_THE_GOAL = normal_goal_distance < PARAMS.MIN_DISTANCE

        BEND_OVER = abs(state['IMU'][1])>PARAMS.MAX_TORSO_ANGLE

        prev_goal = self.goalCoordinate - self.prevCoordinate
        curr_goal = self.goalCoordinate - self.currentCoordinate
        delta = curr_goal-prev_goal

        IS_WONDERED = np.sqrt(np.dot(delta,delta))*(-1 if np.sqrt(np.dot(curr_goal,curr_goal))<np.sqrt(np.dot(prev_goal,prev_goal))else 1)>(PARAMS.MAX_INIT_GOAL_RADIUS-1)
        # if self.supervisor.getName()=='NAO4':
        #     print( np.sqrt(np.dot(delta,delta))*(-1 if np.sqrt(np.dot(curr_goal,curr_goal))<np.sqrt(np.dot(prev_goal,prev_goal))else 1))
        #     print(PARAMS.MAX_INIT_GOAL_RADIUS-1)
        #     print('=========================')
        


        return 1 if (IS_REACHED_THE_GOAL) else -1 if (IS_WONDERED or BEND_OVER or IS_FALL) else 0
    def __setCurrentState__(self, stateName, reset=True):
        self.supervisor.getFromDef(self.supervisor.getName()).loadState(stateName=stateName)
        self.currentCoordinate = self.__getGPSValue__()
        self.currentJointPose = copy.deepcopy(np.array(list(self.__getJointPose__().values())))
        if reset: 
            self.prevJointPose = copy.deepcopy(np.array(list(self.__getJointPose__().values())))
            self.prevCoordinate = self.__getGPSValue__()
            self.prevControls = [[0]*len(self.__getActuators__())]
    def __eval__(self):
        """Will assign stable target networks to main agent and measure it's performance and provide history log.

        Returns:
            Dict: Dictionary of performance log that will be stored in the main rewards variable
        """
        
        # initiate varialbes
        trajectory = []
        total_reward = 0.0
        
        trajectory_final_coordinate = self.initCoordinate

        # set agent location to initial state
        self.__setCurrentState__(stateName='initialState')

        # set agent networks to evaluation mode
        self.actor.eval()
        self.critic.eval()

        # update agent and simulator
        # don't move it to while condition cuz gps needs one turn to be updated!
        self.__stepSimulaiton__()

        # Obtaine init environment state
        s = self.__getCurrentState__()

        # get reward function info
        reward_info = []

        # simulator state array
        episode_end = 0

        # drive agent using target networks until it's fails
        while episode_end < 1:
            with torch.no_grad():
                # Get current value in PPO algorithm
                if PARAMS.ALGORITHM=='PPOV1':
                    with torch.no_grad():
                        v = self.__getValue__((torch.from_numpy(UTILITY.list_to_1d_np_array(s.values())).type(torch.FloatTensor))[None,:])

                # emulate do-while loop
                failed_in_next_state = 0

                # store current state
                self.supervisor.getFromDef(self.supervisor.getName()).saveState(stateName='{}-final_state'.format(self.supervisor.getName()))

                # try to correct current state
                max_try = PARAMS.NUMBER_OF_RECOVERIES(depth=len(reward_info),period=PARAMS.EFFECTIVE_REWARDS)
                while failed_in_next_state <= max_try:
                    # Generate action given state
                    a = self.__getAction__((torch.from_numpy(UTILITY.list_to_1d_np_array(s.values())).type(torch.FloatTensor))[None,:])

                    # Perform act on the environment
                    self.__act__(a['action'])
                    self.__stepSimulaiton__()

                    # Stop act after one time step
                    self.__act__(torch.as_tensor([[0] * len(self.Motors)]))

                    # Obtaine next environment state
                    s2 = self.__getCurrentState__()

                    # update episode end
                    episode_end = self.__episodeIsDone__(s2)
                    
                    # Update loop condition value
                    if episode_end == -1 : failed_in_next_state+=1
                    else: break

                    # return to healthy state if needed
                    if failed_in_next_state< max_try:
                        self.__setCurrentState__(stateName='{}-final_state'.format(self.supervisor.getName()), reset=False)

                # get next state location
                trajectory_final_coordinate = np.array(self.__getGPSValue__())

                # Calculate transition reward
                r,r_info = UTILITY.partial_reward(s2)

                # add r_info to reward_info
                reward_info += [r_info]

                # generate logs
                total_reward += r

                # update trajectory information
                # it ACTUALLY depends on wether algorithm is deterministic or not!
                if PARAMS.ALGORITHM=='DDPGV2':
                    trajectory += [(UTILITY.list_to_1d_np_array(s.values()).tolist(),
                                    a['action'],
                                    r,
                                    UTILITY.list_to_1d_np_array(s2.values()).tolist(),
                                    self.__episodeIsDone__(s2))]
                elif PARAMS.ALGORITHM=='PPOV1':
                    trajectory += [(UTILITY.list_to_1d_np_array(s.values()).tolist(),
                                    a['action'].tolist(),
                                    r,
                                    float(v[0][0]),
                                    a['logp_a'].tolist())]

                # update current state
                s = s2

                # end trial if still fails
                if(episode_end<0):
                    break
                        
        
        # calculate total displacement from start state to final state
        displacement = np.array(trajectory_final_coordinate[0:2]-self.initCoordinate[0:2])

        # calculate mean of trajectory different rewards
        mean_reward_info = list(np.array(reward_info).mean(axis=0))

        return trajectory, {
            'total_reward':total_reward,
            'total_time_steps':len(trajectory),
            'total_distance':np.sqrt(np.dot(displacement,displacement))* (-1 if np.linalg.norm(self.goalCoordinate-np.array(self.__getGPSValue__()),ord=2) >  np.linalg.norm(self.goalCoordinate-self.initCoordinate,ord=2) else 1),
            'lin_vel_cost':mean_reward_info[0],
            'prep_vel_cost':mean_reward_info[1],
            'force_cost':mean_reward_info[2],
            'control_cost':mean_reward_info[3],
            'righ_foot_force':mean_reward_info[4],
            'left_foot_force':mean_reward_info[5],
        }, reward_info



