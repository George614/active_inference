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
        granularity=GRANULARITY,
    ):

        self.env_size = env_size
        self.init_distance = init_distance
        self.source_size = source_size
        self.agent_size = agent_size
        self.vel = velocity
        self.granularity = granularity

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
        self.observe(self.pos)

    def observe(self, prev_pos):
        ''' Calcuate the change of distance '''
        prev_dis = self.dis(prev_pos[0], prev_pos[1], self.s_pos[0], self.s_pos[1])  # 
        cur_dis = self.dis(self.pos[0], self.pos[1], self.s_pos[0], self.s_pos[1])  # 
        if prev_dis > cur_dis:
            o = CHANGE_CLOSER
        elif prev_dis < cur_dis:
            o = CHANGE_FARTHER
        else:
            o = CHANGE_NONE
        return o

    def act(self, a):
        ''' Go different directions then observe (calculate change of distance) '''
        prev_pos = np.copy(self.pos)
        if self.distance() > self.source_size:
            if a == GO_LEFT:
                self.theta -= self.granularity
            elif a == GO_RIGHT:
                self.theta += self.granularity
            self.pos[0] += self.vel * np.cos(self.theta)
            self.pos[1] += self.vel * np.sin(self.theta)
            self.check_bounds()

        return self.observe(prev_pos)

    def distance(self):
        ''' Distance between the agent and the source '''
        return self.dis(self.pos[0], self.pos[1], self.s_pos[0], self.s_pos[1])
        # dis1 = self.dis(self.pos[0], self.pos[1], self.s1_pos[0], self.s1_pos[1])
        # dis2 = self.dis(self.pos[0], self.pos[1], self.s2_pos[0], self.s2_pos[1])
        # return dis1, dis2


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
