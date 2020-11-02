import numpy as np
from .config import *


class Environment(object):
    def __init__(
        self,
        env_size=ENVIRONMENT_SIZE,
        init_distance=INIT_DISTANCE,
        source_size=SOURCE_SIZE,
        agent_size=AGENT_SIZE,
        velocity=VELOCITY,
    ):

        self.env_size = env_size
        self.init_distance = init_distance
        self.source_size = source_size
        self.agent_size = agent_size
        self.vel = velocity

        self.pos = None
        self.s_pos = None
        self.s1_pos = None  # added
        self.s2_pos = None  # added
        self.theta = None
        self.reset()

    def reset(self):
        ''' Reset agent position and orientation, source position '''
        rand_loc = np.random.rand() * (2 * np.pi)
        fx = self.env_size / 2 + (self.init_distance * np.cos(rand_loc))
        fy = self.env_size / 2 + (self.init_distance * np.sin(rand_loc))

        self.pos = [fx, fy]  # positon of the back end of the chemotaxis/agent
        self.s_pos = [self.env_size / 2, self.env_size / 2] # source position
        # added s1 and s2 positions
        self.s1_pos = [self.env_size / 3, self.env_size / 2] # source 1 position
        self.s2_pos = [self.env_size * 2/3 , self.env_size / 2] # source 2 position
        self.theta = np.random.rand() * (2 * np.pi)  # orientation of the agent
        self.observe()

    def observe(self):
        ''' Calcuate the chemical gradient '''
        fx = self.pos[0] + (self.agent_size * np.cos(self.theta)) # fx, fy are front end position
        fy = self.pos[1] + (self.agent_size * np.sin(self.theta))
        f_dis = self.dis(fx, fy, self.s_pos[0], self.s_pos[1])  # front end to source distance 
        b_dis = self.dis(self.pos[0], self.pos[1], self.s_pos[0], self.s_pos[1])  # back end to source distance
        if f_dis > b_dis:
            o = NEG_GRADIENT
        else:
            o = POS_GRADIENT
        ### added new calculation for s1 and s2 ###
        f1_dis = self.dis(fx, fy, self.s1_pos[0], self.s1_pos[1])
        f2_dis = self.dis(fx, fy, self.s2_pos[0], self.s2_pos[1])
        b1_dis = self.dis(self.pos[0], self.pos[1], self.s1_pos[0], self.s1_pos[1])
        b2_dis = self.dis(self.pos[0], self.pos[1], self.s2_pos[0], self.s2_pos[1])
        if f1_dis > b1_dis and f2_dis > b2_dis:
            o = NEG_GRADIENT
        else:
            o = POS_GRADIENT
        return o

    def act(self, a):
        ''' Run or tumble then observe (calculate gradient) '''
        
        # if a == RUN and self.distance() > self.source_size:
        dis1, dis2 = self.distance() # distance between front and back and the gradient center

        # Run and check bounds, or tumble.
        if a == RUN and dis1 > self.source_size and dis2 > self.source_size:
            self.pos[0] += self.vel * np.cos(self.theta) # update position.  NOT stored in a history.
            self.pos[1] += self.vel * np.sin(self.theta)
            self.check_bounds()
        elif a == TUMBLE:
            self.theta = np.random.rand() * (2 * np.pi)

        return self.observe()

    def distance(self):
        ''' Distance between the agent and the source '''
        # return self.dis(self.pos[0], self.pos[1], self.s_pos[0], self.s_pos[1])
        dis1 = self.dis(self.pos[0], self.pos[1], self.s1_pos[0], self.s1_pos[1])
        dis2 = self.dis(self.pos[0], self.pos[1], self.s2_pos[0], self.s2_pos[1])
        return dis1, dis2


    def check_bounds(self):
        if self.pos[0] > self.env_size:
            self.pos[0] = self.env_size
        if self.pos[0] < 0:
            self.pos[0] = 0
        if self.pos[1] > self.env_size:
            self.pos[1] = self.env_size
        if self.pos[1] < 0:
            self.pos[1] = 0

    @staticmethod
    def dis(x1, y1, x2, y2):
        ''' Euclidean distance '''
        return np.sqrt(((x1 - x2) * (x1 - x2)) + ((y1 - y2) * (y1 - y2)))
