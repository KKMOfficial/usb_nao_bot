from cmath import inf
from controller import Robot, Supervisor, Keyboard, Node
import copy

class Nao():
    def __init__(self):

        self.supervisor = Supervisor()
        self.stateVector = []
        self.initialState = {}
        self.timestep = int(self.supervisor.getBasicTimeStep())
        self.Motors = self.__initMotors__()
        self.Sensors = self.__initSensors__()
        self.PosSensors = self.__initPositionSensors__()
        self.WB_SUPERVISOR_SIMULATION_MODE_PAUSE = 0
        self.SIMULATION_MODE_REAL_TIME = 1
        self.WB_SUPERVISOR_SIMULATION_MODE_FAST = 2
        self.EXIT_SUCCESS = 0
        self.EXIT_FAILURE = 1
      
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

    def __getLastState__(self):
        return copy.deepcopy(self.stateVector[-1])

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
            'RFoot/Bumper/Right':self.Sensors['RFoot/Bumper/Right'].getValue(),
            'RFoot/Bumper/Left':self.Sensors['RFoot/Bumper/Left'].getValue(),
            'LFoot/Bumper/Right':self.Sensors['LFoot/Bumper/Right'].getValue(),
            'LFoot/Bumper/Left':self.Sensors['LFoot/Bumper/Left'].getValue(),
            'RFoot/ForceSensor':self.Sensors['RFoot/ForceSensor'].getValues(),
            'LFoot/ForceSensor':self.Sensors['LFoot/ForceSensor'].getValues(),
            'GPS':self.Sensors['GPS'].getValues(),
            'Accelerometer':self.Sensors['Accelerometer'].getValues()
        }

    def __act__(self, command):
        for index, motor in enumerate(self.Motors):
            motor.setVelocity(command[index])

    def __initPositionSensors__(self):
        posSensorDictionary = {
            'LAnklePitchS':self.supervisor.getDevice('LAnklePitchS'),
            'LAnkleRollS':self.supervisor.getDevice('LAnkleRollS'),
            'LElbowRollS':self.supervisor.getDevice('LElbowRollS'),
            'LElbowYawS':self.supervisor.getDevice('LElbowYawS'),
            'LHipPitchS':self.supervisor.getDevice('LHipPitchS'),
            'LHipRollS':self.supervisor.getDevice('LHipRollS'),
            'LHipYawPitchS':self.supervisor.getDevice('LHipYawPitchS'),
            'LKneePitchS':self.supervisor.getDevice('LKneePitchS'),
            'LShoulderPitchS':self.supervisor.getDevice('LShoulderPitchS'),
            'LShoulderRollS':self.supervisor.getDevice('LShoulderRollS'),
            'LWristYawS':self.supervisor.getDevice('LWristYawS'),
            'RAnklePitchS':self.supervisor.getDevice('RAnklePitchS'),
            'RAnkleRollS':self.supervisor.getDevice('RAnkleRollS'),
            'RElbowRollS':self.supervisor.getDevice('RElbowRollS'),
            'RElbowYawS':self.supervisor.getDevice('RElbowYawS'),
            'RHipPitchS':self.supervisor.getDevice('RHipPitchS'),
            'RHipRollS':self.supervisor.getDevice('RHipRollS'),
            'RHipYawPitchS':self.supervisor.getDevice('RHipYawPitchS'),
            'RKneePitchS':self.supervisor.getDevice('RKneePitchS'),
            'RShoulderPitchS':self.supervisor.getDevice('RShoulderPitchS'),
            'RShoulderRollS':self.supervisor.getDevice('RShoulderRollS'),
            'RWristYawS':self.supervisor.getDevice('RWristYawS'),
        }
        for sensorName,sensor in posSensorDictionary.items():
            sensor.enable(self.timestep)
        return posSensorDictionary

    def __jointsEstimatedRotation__(self):
        """values measured in radian unit"""
        return {
            'LAnklePitchS':self.PosSensors['LAnklePitchS'].getValue(),
            'LAnkleRollS':self.PosSensors['LAnkleRollS'].getValue(),
            'LElbowRollS':self.PosSensors['LElbowRollS'].getValue(),
            'LElbowYawS':self.PosSensors['LElbowYawS'].getValue(),
            'LHipPitchS':self.PosSensors['LHipPitchS'].getValue(),
            'LHipRollS':self.PosSensors['LHipRollS'].getValue(),
            'LHipYawPitchS':self.PosSensors['LHipYawPitchS'].getValue(),
            'LKneePitchS':self.PosSensors['LKneePitchS'].getValue(),
            'LShoulderPitchS':self.PosSensors['LShoulderPitchS'].getValue(),
            'LShoulderRollS':self.PosSensors['LShoulderRollS'].getValue(),
            'LWristYawS':self.PosSensors['LWristYawS'].getValue(),
            'RAnklePitchS':self.PosSensors['RAnklePitchS'].getValue(),
            'RAnkleRollS':self.PosSensors['RAnkleRollS'].getValue(),
            'RElbowRollS':self.PosSensors['RElbowRollS'].getValue(),
            'RElbowYawS':self.PosSensors['RElbowYawS'].getValue(),
            'RHipPitchS':self.PosSensors['RHipPitchS'].getValue(),
            'RHipRollS':self.PosSensors['RHipRollS'].getValue(),
            'RHipYawPitchS':self.PosSensors['RHipYawPitchS'].getValue(),
            'RKneePitchS':self.PosSensors['RKneePitchS'].getValue(),
            'RShoulderPitchS':self.PosSensors['RShoulderPitchS'].getValue(),
            'RShoulderRollS':self.PosSensors['RShoulderRollS'].getValue(),
            'RWristYawS':self.PosSensors['RWristYawS'].getValue(),
        }



