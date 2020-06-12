import os
import math
import numpy as np
 
import gym
from gym import spaces
from gym.utils import seeding
 
import pybullet as p
import pybullet_data
 
#the observation space consists of 3 continous values:-
#robot inclination around the x-axis
#angular velocity of the robot
#angular velocity of the wheels of the robot

class BalancebotEnv(gym.Env):
    metadata = {'render.modes': ['human', 'rgb_array'],
                'video.frames_per_second' : 50}
 
    def __init__(self):
        self._observation = []
        self.action_space = spaces.Discrete(9)  #our action space consisting of 9 discrete actions - i.e. different changes in velocity
        self.observation_space = spaces.Box(np.array([-math.pi, -math.pi, -5]), np.array([math.pi, math.pi, 5])) #observation space
        self.physicsClient = p.connect(p.GUI)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())  
        self._seed()
 
    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
 
    def _step(self, action):
        self._assign_throttle(action)                            
        p.stepSimulation()                                            #actions taking place
        self._observation = self._compute_observation()               #calculating the observation after taking actions
        reward = self._compute_reward()                               #calculating the reward we get for taking that action in that particular state
        done = self._compute_done()                                   #checking whether the episode has finished or not
 
        self._envStepCounter += 1
 
        return np.array(self._observation), reward, done, {}
 
    def _reset(self):
        self.vt = 0     #angular velocity of the wheel, this will be updated so as to balance the bot
        self.vd = 0
        self._envStepCounter = 0
 
        p.resetSimulation()
        p.setGravity(0,0,-10) # m/s^2
        p.setTimeStep(0.01) # sec
        planeId = p.loadURDF("plane.urdf")
        cubeStartPos = [0,0,0.001]
        cubeStartOrientation = p.getQuaternionFromEuler([0,0,0])
 
        path = os.path.abspath(os.path.dirname(__file__))
        self.botId = p.loadURDF(os.path.join(path, "balancebot_simple.xml"),cubeStartPos,cubeStartOrientation)
 
        self._observation = self._compute_observation()
        return np.array(self._observation)
 
    def _assign_throttle(self, action):
        dv = 0.1                          
        deltav = [-10.*dv,-5.*dv, -2.*dv, -0.1*dv, 0, 0.1*dv, 2.*dv,5.*dv, 10.*dv][action]      #determines what change in angular velocity should take place depending on the action given
        vt = self.vt + deltav
        self.vt = vt
        p.setJointMotorControl2(bodyUniqueId=self.botId, 
                                jointIndex=0, 
                                controlMode=p.VELOCITY_CONTROL, 
                                targetVelocity=vt)
        p.setJointMotorControl2(bodyUniqueId=self.botId, 
                                jointIndex=1, 
                                controlMode=p.VELOCITY_CONTROL, 
                                targetVelocity=-vt)
 
    def _compute_observation(self):
        cubePos, cubeOrn = p.getBasePositionAndOrientation(self.botId)
        cubeEuler = p.getEulerFromQuaternion(cubeOrn)
        linear, angular = p.getBaseVelocity(self.botId)              
        return [cubeEuler[0],angular[0],self.vt]    #returning the observations
 
    def _compute_reward(self):
        _, cubeOrn = p.getBasePositionAndOrientation(self.botId)
        cubeEuler = p.getEulerFromQuaternion(cubeOrn)
        # could also be pi/2 - abs(cubeEuler[0])
        return (1 - abs(cubeEuler[0])) * 0.1 -  abs(self.vt - self.vd) * 0.01    #the reward calculated
 
    def _compute_done(self):
        cubePos, _ = p.getBasePositionAndOrientation(self.botId)
        return cubePos[2] < 0.15 or self._envStepCounter >= 1500
 
    def _render(self, mode='human', close=False):
        pass