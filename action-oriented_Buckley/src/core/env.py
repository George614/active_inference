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
        representation=OBV_OPTION,
    ):

        self.env_size = env_size
        self.init_distance = init_distance
        self.source_size = source_size
        self.agent_size = agent_size
        self.vel = velocity
        self.granularity = granularity
        self.representation = representation

        self.pos = None
        self.s_pos = None
        self.s1_pos = None  # added
        self.s2_pos = None  # added
        self.theta = None  # agent's orientation
        self.phi = None  # approach angle
        self.step_count = None  # time steps used so far
        self.steps_episode = []  # total time steps used per episode
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
        vec_agent_to_source = self.vec_norm(np.asarray(self.s_pos) - np.asarray(self.pos))
        vec_agent_heading = np.asarray([np.cos(self.theta), np.sin(self.theta)])
        self.phi = np.arccos(np.dot(vec_agent_to_source, vec_agent_heading))
        self.step_count = 0
        
        self.observe(self.pos, self.phi)

    def observe(self, prev_pos=None, prev_angle=None):
        ''' Calcuate the change of distance / approach angle '''
        if self.representation == CHANGE_DISTANCE or self.representation == CHANGE_BOTH:
            prev_dis = self.dis(prev_pos[0], prev_pos[1], self.s_pos[0], self.s_pos[1])
            cur_dis = self.dis(self.pos[0], self.pos[1], self.s_pos[0], self.s_pos[1])
        if self.representation == CHANGE_ANGLE or self.representation == CHANGE_BOTH:
            vec_agent_to_source = self.vec_norm(np.asarray(self.s_pos) - np.asarray(self.pos))
            vec_agent_heading = np.asarray([np.cos(self.theta), np.sin(self.theta)])
            self.phi = np.arccos(np.dot(vec_agent_to_source, vec_agent_heading))
        # get the observation based on representation of choice
        if self.representation == CHANGE_DISTANCE:
            if prev_dis > cur_dis:
                o = CHANGE_CLOSER
            elif prev_dis < cur_dis:
                o = CHANGE_FARTHER
            else:
                o = CHANGE_NONE
        elif self.representation == CHANGE_ANGLE:
            if prev_angle > self.phi:
                o = TURN_TOWARDS
            elif prev_angle < self.phi:
                o = TURN_AWAY
            else:
                o = TURN_NONE
        elif self.representation == CHANGE_BOTH:
            if prev_dis > cur_dis and prev_angle > self.phi:
                o = DIS_CLOSER_TURN_TOWARDS
            elif prev_dis > cur_dis and prev_angle < self.phi:
                o = DIS_CLOSER_TURN_AWAY
            elif prev_dis > cur_dis and prev_angle == self.phi:
                o = DIS_CLOSER_TURN_NONE
            elif prev_dis < cur_dis and prev_angle > self.phi:
                o = DIS_FARTHER_TURN_TOWARDS
            elif prev_dis < cur_dis and prev_angle < self.phi:
                o = DIS_FARTHER_TURN_AWAY
            elif prev_dis < cur_dis and prev_angle == self.phi:
                o = DIS_CLOSER_TURN_NONE
            elif prev_dis == cur_dis and prev_angle > self.phi:
                o = DIS_NONE_TURN_TOWARDS
            elif prev_dis == cur_dis and prev_angle < self.phi:
                o = DIS_NONE_TURN_AWAY
            elif prev_dis == cur_dis and prev_angle == self.phi:
                o = DIS_NONE_TURN_NONE
        return o

    def act(self, a):
        ''' Go different directions then observe (calculate change of distance) '''
        prev_pos = np.copy(self.pos)
        prev_angle = np.copy(self.phi)
        if self.distance() > self.source_size:
            if a == GO_LEFT:
                self.theta -= self.granularity
            elif a == GO_RIGHT:
                self.theta += self.granularity
            self.pos[0] += self.vel * np.cos(self.theta)
            self.pos[1] += self.vel * np.sin(self.theta)
            self.check_bounds()
            self.step_count += 1
        else:
        	if CONTINUAL_LEARNING:
        		self.steps_episode.append(self.step_count)
        		self.reset()
        		return self.observe(self.pos, self.phi)
        return self.observe(prev_pos, prev_angle)

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

    @staticmethod
    def vec_norm(vec):
        return vec / LA.norm(vec)