import numpy as np
from numpy import linalg as LA
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

        self.pos = None    # agent position
        self.s_pos = None  # source position
        self.theta = None  # agent's orientation
        self.phi = None  # visual angle
        self.reset()

    def reset(self):
        ''' Reset agent position and orientation, source position '''
        rand_loc = np.random.rand() * (2 * np.pi)
        fx = self.env_size / 2 + (self.init_distance * np.cos(rand_loc))
        fy = self.env_size / 2 + (self.init_distance * np.sin(rand_loc))
        self.pos = [fx, fy]  # positon of the back end of the chemotaxis/agent
        self.s_pos = [self.env_size / 2, self.env_size / 2] # source position
        self.theta = np.random.rand() * (2 * np.pi)  # orientation of the agent
        self.observe()

    def observe(self):
        ''' Calcuate in which visual region of the agent that the target falls '''
        vec_agent_to_source = self.vec_norm(np.asarray(self.s_pos) - np.asarray(self.pos))
        vec_agent_heading = np.asarray([np.cos(self.theta), np.sin(self.theta)])
        self.phi = np.arccos(np.dot(vec_agent_to_source, vec_agent_heading))
        for i, vrange in enumerate(VISUAL_RANGES):
            if self.phi >= vrange[0] and self.phi <= vrange[1]:
                o = i
                return o
        return VISUAL_R5

    def act(self, a):
        ''' Go different directions then observe (calculate visual region) '''
        if self.distance() > self.source_size:
            if a == GO_LEFT:
                self.theta -= self.granularity
            elif a == GO_RIGHT:
                self.theta += self.granularity
            self.pos[0] += self.vel * np.cos(self.theta)
            self.pos[1] += self.vel * np.sin(self.theta)
            self.check_bounds()
       
        return self.observe()

    def distance(self):
        ''' Distance between the agent and the source '''
        return self.dis(self.pos[0], self.pos[1], self.s_pos[0], self.s_pos[1])

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

    @staticmethod
    def vec_norm(vec):
        return vec / LA.norm(vec)
