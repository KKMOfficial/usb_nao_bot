from cmath import inf
from controller import Robot, Supervisor, Keyboard, Node

class Nao():
    def __init__(self):

        self.supervisor = Supervisor()
        self.stateVector = []
        self.initialState = []
        self.timestep = int(self.supervisor.getBasicTimeStep())
        self.Motors = self.__initMotors__()
        self.Sensors = self.__initSensors__()
        self.WB_SUPERVISOR_SIMULATION_MODE_PAUSE = 0
        self.SIMULATION_MODE_REAL_TIME = 1
        self.WB_SUPERVISOR_SIMULATION_MODE_FAST = 2
        self.episodeGoalCoordinate = None

        
    def __stopSimulation__(self):
        self.supervisor.simulationSetMode(self.WB_SUPERVISOR_SIMULATION_MODE_PAUSE)

    def __resumeSimulation__(self):
        self.supervisor.simulationSetMode(self.SIMULATION_MODE_REAL_TIME)
    
    def __resumeFastSimulation__(self):
        self.supervisor.simulationSetMode(self.WB_SUPERVISOR_SIMULATION_MODE_FAST)
    
    def __updateSimulator__(self):
        self.supervisor.step(0)
    
    def __stepSimulaiton__(self):
        self.supervisor.step(self.timestep)

    def __resetSimulation__(self):
        self.supervisor.simulationReset()

    def __storeCurrentState__(self):
        self.stateVector += [
            {
                # maybe we add velocities in future
                'NaoPosition':self.supervisor.getFromDef('NAO').getPosition(),
                'NaoOrientation':self.supervisor.getFromDef('NAO').getOrientation(),
                # Head Pitch
                # Head Yaw
                'HeadOrientation':self.supervisor.getFromDef('NAO.HeadYaw.HeadEndPoint').getOrientation(),
                # R-Shoulder Roll
                # R-Shoulder Pitch
                'R-Shoulder':self.supervisor.getFromDef('NAO.RShoulderPitch.RShoulder').getOrientation(),
                # R-Elbow Roll
                # R-Elbow Yaw
                'R-Elbow':self.supervisor.getFromDef('NAO.RShoulderPitch.RShoulder.RElbowYaw.RElbow').getOrientation(),
                # R-Wrist Yaw
                'R-Wrist':self.supervisor.getFromDef('NAO.RShoulderPitch.RShoulder.RElbowYaw.RElbow.RElbowRoll.RElbowEnd.RWristYaw.RWrist').getOrientation(),
                # R-Hip Yaw-Pitch
                'R-HipYP':self.supervisor.getFromDef('NAO.RHipYawPitch.RHipYP').getOrientation(),
                # R-Hip Pitch
                'R-HipPitch':self.supervisor.getFromDef('NAO.RHipYawPitch.RHipYP.RHipRoll.RHipR.RHipPitch.RHipP').getOrientation(),
                # R-Hip Roll
                'R-HipRoll':self.supervisor.getFromDef('NAO.RHipYawPitch.RHipYP.RHipRoll.RHipR').getOrientation(),
                # R-Knee Pitch
                'R-KneePitch':self.supervisor.getFromDef('NAO.RHipYawPitch.RHipYP.RHipRoll.RHipR.RHipPitch.RHipP.RKneePitch.RKneeP').getOrientation(),
                # R-Ankle Pitch
                'R-AnklePitch':self.supervisor.getFromDef('NAO.RHipYawPitch.RHipYP.RHipRoll.RHipR.RHipPitch.RHipP.RKneePitch.RKneeP.RAnklePitch.RAnkleP').getOrientation(),
                # R-Ankle Roll
                'R-AnkleRoll':self.supervisor.getFromDef('NAO.RHipYawPitch.RHipYP.RHipRoll.RHipR.RHipPitch.RHipP.RKneePitch.RKneeP.RAnklePitch.RAnkleP.RAnkleRoll.RAnkleR').getOrientation(),
                # L-Shoulder Roll
                # L-Shoulder Pitch
                'L-Shoulder':self.supervisor.getFromDef('NAO.LShoulderPitch.LShoulder').getOrientation(),
                # L-Elbow Roll
                # L-Elbow Yaw
                'L-Elbow':self.supervisor.getFromDef('NAO.LShoulderPitch.LShoulder.LElbowYaw.LElbow').getOrientation(),
                # L-Wrist Yaw
                'L-Elbow':self.supervisor.getFromDef('NAO.LShoulderPitch.LShoulder.LElbowYaw.LElbow.LElbowRoll.LElbowEnd.LWristYaw.LWrist').getOrientation(),
                # L-Hip Yaw-Pitch
                'L-HipYP':self.supervisor.getFromDef('NAO.LHipYawPitch.LHipYP').getOrientation(),
                # L-Hip Pitch
                'L-HipPitch':self.supervisor.getFromDef('NAO.LHipYawPitch.LHipYP.LHipRoll.LHipR.LHipPitch.LHipP').getOrientation(),
                # L-Hip Roll
                'L-HipRoll':self.supervisor.getFromDef('NAO.LHipYawPitch.LHipYP.LHipRoll.LHipR').getOrientation(),
                # L-Knee Pitch
                'L-KneePitch':self.supervisor.getFromDef('NAO.LHipYawPitch.LHipYP.LHipRoll.LHipR.LHipPitch.LHipP.LKneePitch.LKneeP').getOrientation(),
                # L-Ankle Pitch
                'L-AnklePitch':self.supervisor.getFromDef('NAO.LHipYawPitch.LHipYP.LHipRoll.LHipR.LHipPitch.LHipP.LKneePitch.LKneeP.LAnklePitch.LAnkleP').getOrientation(),
                # L-Ankle Roll
                'L-AnkleRoll':self.supervisor.getFromDef('NAO.LHipYawPitch.LHipYP.LHipRoll.LHipR.LHipPitch.LHipP.LKneePitch.LKneeP.LAnklePitch.LAnkleP.LAnkleRoll.LAnkleR').getOrientation(),
            }
        ]

    def __printLastState__(self):
        print([(key,len(val)) for key,val in self.stateVector[-1].items()])

    def __setState__(self, stateIndex=-1):
        self.supervisor.getFromDef('NAO').getField('translation').setSFVec3f(self.stateVector[stateIndex]['NaoPosition'])
        self.supervisor.getFromDef('NAO').getField('rotation').setSFRotation(self.stateVector[stateIndex]['NaoOrientation'])
        self.supervisor.getFromDef('NAO.HeadYaw.HeadEndPoint').getField('rotation').setSFRotation(self.stateVector[stateIndex]['HeadOrientation'])
        self.supervisor.getFromDef('NAO.RShoulderPitch.RShoulder').getField('rotation').setSFRotation(self.stateVector[stateIndex]['R-Shoulder'])
        self.supervisor.getFromDef('NAO.RShoulderPitch.RShoulder.RElbowYaw.RElbow').getField('rotation').setSFRotation(self.stateVector[stateIndex]['R-Elbow'])
        self.supervisor.getFromDef('NAO.RShoulderPitch.RShoulder.RElbowYaw.RElbow.RElbowRoll.RElbowEnd.RWristYaw.RWrist').getField('rotation').setSFRotation(self.stateVector[stateIndex]['R-Wrist'])
        self.supervisor.getFromDef('NAO.RHipYawPitch.RHipYP').getField('rotation').setSFRotation(self.stateVector[stateIndex]['R-HipYP'])
        self.supervisor.getFromDef('NAO.RHipYawPitch.RHipYP.RHipRoll.RHipR.RHipPitch.RHipP').getField('rotation').setSFRotation(self.stateVector[stateIndex]['R-HipPitch'])
        self.supervisor.getFromDef('NAO.RHipYawPitch.RHipYP.RHipRoll.RHipR').getField('rotation').setSFRotation(self.stateVector[stateIndex]['R-HipRoll'])
        self.supervisor.getFromDef('NAO.RHipYawPitch.RHipYP.RHipRoll.RHipR.RHipPitch.RHipP.RKneePitch.RKneeP').getField('rotation').setSFRotation(self.stateVector[stateIndex]['R-KneePitch'])
        self.supervisor.getFromDef('NAO.RHipYawPitch.RHipYP.RHipRoll.RHipR.RHipPitch.RHipP.RKneePitch.RKneeP.RAnklePitch.RAnkleP').getField('rotation').setSFRotation(self.stateVector[stateIndex]['R-AnklePitch'])
        self.supervisor.getFromDef('NAO.RHipYawPitch.RHipYP.RHipRoll.RHipR.RHipPitch.RHipP.RKneePitch.RKneeP.RAnklePitch.RAnkleP.RAnkleRoll.RAnkleR').getField('rotation').setSFRotation(self.stateVector[stateIndex]['R-AnkleRoll'])
        self.supervisor.getFromDef('NAO.LShoulderPitch.LShoulder').getField('rotation').setSFRotation(self.stateVector[stateIndex]['L-Shoulder'])
        self.supervisor.getFromDef('NAO.LShoulderPitch.LShoulder.LElbowYaw.LElbow').getField('rotation').setSFRotation(self.stateVector[stateIndex]['L-Elbow'])
        self.supervisor.getFromDef('NAO.LShoulderPitch.LShoulder.LElbowYaw.LElbow.LElbowRoll.LElbowEnd.LWristYaw.LWrist').getField('rotation').setSFRotation(self.stateVector[stateIndex]['L-Elbow'])
        self.supervisor.getFromDef('NAO.LHipYawPitch.LHipYP').getField('rotation').setSFRotation(self.stateVector[stateIndex]['L-HipYP'])
        self.supervisor.getFromDef('NAO.LHipYawPitch.LHipYP.LHipRoll.LHipR.LHipPitch.LHipP').getField('rotation').setSFRotation(self.stateVector[stateIndex]['L-HipPitch'])
        self.supervisor.getFromDef('NAO.LHipYawPitch.LHipYP.LHipRoll.LHipR').getField('rotation').setSFRotation(self.stateVector[stateIndex]['L-HipRoll'])
        self.supervisor.getFromDef('NAO.LHipYawPitch.LHipYP.LHipRoll.LHipR.LHipPitch.LHipP.LKneePitch.LKneeP').getField('rotation').setSFRotation(self.stateVector[stateIndex]['L-KneePitch'])
        self.supervisor.getFromDef('NAO.LHipYawPitch.LHipYP.LHipRoll.LHipR.LHipPitch.LHipP.LKneePitch.LKneeP.LAnklePitch.LAnkleP').getField('rotation').setSFRotation(self.stateVector[stateIndex]['L-AnklePitch'])
        self.supervisor.getFromDef('NAO.LHipYawPitch.LHipYP.LHipRoll.LHipR.LHipPitch.LHipP.LKneePitch.LKneeP.LAnklePitch.LAnkleP.LAnkleRoll.LAnkleR').getField('rotation').setSFRotation(self.stateVector[stateIndex]['L-AnkleRoll'])
    
    def __initMotors__(self):
        motorDictionary = {
            'HeadPitch':self.supervisor.getDevice('HeadPitch'),
            'HeadYaw':self.supervisor.getDevice('HeadYaw'),
            'R-Shoulder Roll':self.supervisor.getDevice('RShoulderRoll'),
            'R-Shoulder Pitch':self.supervisor.getDevice('RShoulderPitch'),
            'R-Elbow Roll':self.supervisor.getDevice('RElbowRoll'),
            'R-Elbow Yaw':self.supervisor.getDevice('RElbowYaw'),
            'R-Wrist Yaw':self.supervisor.getDevice('RWristYaw'),
            'R-Hip Yaw Pitch':self.supervisor.getDevice('RHipYawPitch'),
            'R-Hip Pitch':self.supervisor.getDevice('RHipPitch'),
            'R-Hip Roll':self.supervisor.getDevice('RHipRoll'),
            'R-Knee Pitch':self.supervisor.getDevice('RKneePitch'),
            'R-Ankle Pitch':self.supervisor.getDevice('RAnklePitch'),
            'R-Ankle Roll':self.supervisor.getDevice('RAnkleRoll'),
            'L-Shoulder Roll':self.supervisor.getDevice('LShoulderRoll'),
            'L-Shoulder Pitch':self.supervisor.getDevice('LShoulderPitch'),
            'L-Elbow Roll':self.supervisor.getDevice('LElbowRoll'),
            'L-Elbow Yaw':self.supervisor.getDevice('LElbowYaw'),
            'L-Wrist Yaw':self.supervisor.getDevice('LWristYaw'),
            'L-Hip Yaw Pitch':self.supervisor.getDevice('LHipYawPitch'),
            'L-Hip Pitch':self.supervisor.getDevice('LHipPitch'),
            'L-Hip Roll':self.supervisor.getDevice('LHipRoll'),
            'L-Knee Pitch':self.supervisor.getDevice('LKneePitch'),
            'L-Ankle Pitch':self.supervisor.getDevice('LAnklePitch'),
            'L-Ankle Roll':self.supervisor.getDevice('LAnkleRoll'),
        }
        for motorName,motor in motorDictionary.items():
            motor.setPosition(inf)
            motor.setVelocity(0.0)
        return motorDictionary

    def __initSensors__(self):
        sensorDictionary = {
            'CameraBottom':self.supervisor .getDevice('CameraBottom'),
            'CameraTop':self.supervisor .getDevice('CameraTop'),
            'GPS':self.supervisor .getDevice('gps'),
            'RFoot/Bumper/Left':self.supervisor.getDevice('RFoot/Bumper/Left'),
            'RFoot/Bumper/Right':self.supervisor.getDevice('RFoot/Bumper/Right'),
            'LFoot/Bumper/Left':self.supervisor.getDevice('LFoot/Bumper/Left'),
            'LFoot/Bumper/Right':self.supervisor.getDevice('LFoot/Bumper/Right'),
            'RFoot/ForceSensor':self.supervisor.getDevice('RFsr'),
            'LFoot/ForceSensor':self.supervisor.getDevice('LFsr'),
            'Accelerometer':self.supervisor.getDevice('accelerometer')
        }
        for sensorName,sensor in sensorDictionary.items():
            sensor.enable(self.timestep)
        return sensorDictionary

    def __observe__(self):
        return {
            'RFRB':self.Sensors['RFoot/Bumper/Right'].getValue(),
            'RFLB':self.Sensors['RFoot/Bumper/Left'].getValue(),
            'LFRB':self.Sensors['LFoot/Bumper/Right'].getValue(),
            'LFLB':self.Sensors['LFoot/Bumper/Left'].getValue(),
            'RFFS':self.Sensors['RFoot/ForceSensor'].getValues(),
            'LFFS':self.Sensors['LFoot/ForceSensor'].getValues(),
            'GPS':self.Sensors['GPS'].getValues(),
            'ACCL':self.Sensors['Accelerometer'].getValues()
        }

    def __act__(self, command):
        """will perform <command> in following <time_step>."""
        for index, motor in enumerate(self.Motors):
            motor.setVelocity(command[index])

    def __jointsEstimatedRotation__(self):
        pass

    def __reward__(self, state):
        # termination conditions
        is_done_at_t = None # will use force sensors of the soles
        # reward function criterion
        reward_t = velocity_in_goal_direction - 3*displacement_in_lateral_goal_direction**2 - 50*center_of_mass_displacement + 25*total_episode_time_in_term_of_samples/total_episode_time \
                    - 0.02*norm_2_squared_all_joint_velocity_previous_time
        return reward_t, is_done_at_t

    def __startNewEpisode__(self):
        # pick goal coordinate here
        self.episodeGoalCoordinate = None
        pass

