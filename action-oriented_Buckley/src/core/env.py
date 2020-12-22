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

        self.pos = [self.env_size/2, self.env_size/2]
        self.s_pos = None  # target position
        self.obst_pos = None  # obstacle position
        self.theta = None  # agent's orientation
        self.phi = None  # approach angle
        self.step_count = None  # time steps used so far
        self.episode_count = 0
        self.steps_episode = []  # total time steps used per episode
        self.s_pos_list = [[self.env_size/2, self.env_size-2*self.source_size], 
                            [self.env_size-2*self.source_size, self.env_size/2],
                            [self.env_size/2, 2*self.source_size],
                            [2*self.source_size, self.env_size/2]]
        self.obstacle_list = [[0.5*(self.s_pos_list[0][0]+self.s_pos_list[1][0]),
                              0.5*(self.s_pos_list[0][1]+self.s_pos_list[1][1])],
                              [0.5*(self.s_pos_list[1][0]+self.s_pos_list[2][0]),
                              0.5*(self.s_pos_list[1][1]+self.s_pos_list[2][1])],
                              [0.5*(self.s_pos_list[2][0]+self.s_pos_list[3][0]),
                              0.5*(self.s_pos_list[2][1]+self.s_pos_list[3][1])],
                              [0.5*(self.s_pos_list[0][0]+self.s_pos_list[3][0]),
                              0.5*(self.s_pos_list[0][1]+self.s_pos_list[3][1])]]
        self.reset()

    def reset(self):
        ''' Reset agent position and orientation, source position '''
        # rand_loc = np.random.rand() * (2 * np.pi)
        # fx = self.env_size / 2 + (self.init_distance * np.cos(rand_loc))
        # fy = self.env_size / 2 + (self.init_distance * np.sin(rand_loc))
        # self.pos = [fx, fy]  # positon of the back end of the chemotaxis/agent
        self.s_pos = self.s_pos_list[self.episode_count % len(self.s_pos_list)] # source position
        self.obst_pos = self.obstacle_list[(self.episode_count-1) % len(self.obstacle_list)]
        if self.episode_count == 0:
            self.theta = np.random.rand() * (2 * np.pi)  # orientation of the agent
            # self.obst_pos = None
        vec_agent_to_source = self.vec_norm(np.asarray(self.s_pos) - np.asarray(self.pos))
        vec_agent_heading = np.asarray([np.cos(self.theta), np.sin(self.theta)])
        self.phi = np.arccos(np.dot(vec_agent_to_source, vec_agent_heading))
        self.step_count = 0
        
        self.observe(self.pos, self.phi)

    def observe(self, prev_pos=None, prev_angle=None):
        ''' Calcuate the change of distance / approach angle '''
        if self.representation != CHANGE_ANGLE:
            prev_dis = self.dis(prev_pos[0], prev_pos[1], self.s_pos[0], self.s_pos[1])
            cur_dis = self.dis(self.pos[0], self.pos[1], self.s_pos[0], self.s_pos[1])
        if self.representation != CHANGE_DISTANCE:
            vec_agent_to_source = self.vec_norm(np.asarray(self.s_pos) - np.asarray(self.pos))
            vec_agent_heading = np.asarray([np.cos(self.theta), np.sin(self.theta)])
            self.phi = np.arccos(np.dot(vec_agent_to_source, vec_agent_heading))
        # get the observation based on representation of choice
        if self.representation == CHANGE_DISTANCE:
            if prev_dis > cur_dis:
                o = DIS_CLOSER
            elif prev_dis < cur_dis:
                o = DIS_FARTHER
            else:
                o = DIS_NONE
        
        elif self.representation == CHANGE_ANGLE:
            if prev_angle > self.phi:
                o = TURN_TOWARDS
            elif prev_angle < self.phi:
                o = TURN_AWAY
            else:
                o = TURN_NONE
                
        elif self.representation == CHANGE_BOTH:
            if prev_angle == self.phi:
                o_angle = TURN_NONE
            else:
                if prev_angle > self.phi:
                    o_angle = TURN_TOWARDS
                elif prev_angle < self.phi:
                    o_angle = TURN_AWAY

            if prev_dis > cur_dis and o_angle == TURN_TOWARDS:
                o = DIS_CLOSER_TURN_TOWARDS
            elif prev_dis > cur_dis and o_angle == TURN_AWAY:
                o = DIS_CLOSER_TURN_AWAY
            elif prev_dis > cur_dis and o_angle == TURN_NONE:
                o = DIS_CLOSER_TURN_NONE
            elif prev_dis < cur_dis and o_angle == TURN_TOWARDS:
                o = DIS_FARTHER_TURN_TOWARDS
            elif prev_dis < cur_dis and o_angle == TURN_AWAY:
                o = DIS_FARTHER_TURN_AWAY
            elif prev_dis < cur_dis and o_angle == TURN_NONE:
                o = DIS_FARTHER_TURN_NONE
            elif prev_dis == cur_dis and o_angle == TURN_TOWARDS:
                o = DIS_NONE_TURN_TOWARDS
            elif prev_dis == cur_dis and o_angle == TURN_AWAY:
                o = DIS_NONE_TURN_AWAY
            elif prev_dis == cur_dis and o_angle == TURN_NONE:
                o = DIS_NONE_TURN_NONE

        elif self.representation == DISCRETE_BOTH:
            angle_change = self.phi - prev_angle
            if prev_dis > cur_dis:
                if angle_change == 0:
                    o = DCTN
                elif angle_change>0 and angle_change<=11.25/180*np.pi:
                    o = DCTL11
                elif angle_change>11.25/180*np.pi and angle_change<=22.5/180*np.pi:
                    o = DCTL22
                elif angle_change>22.5/180*np.pi and angle_change<=45/180*np.pi:
                    o = DCTL45
                elif angle_change>45/180*np.pi and angle_change<=90/180*np.pi:
                    o = DCTL90
                elif angle_change>90/180*np.pi:
                    o = DCTL180
                elif angle_change<0 and angle_change>= -11.25/180*np.pi:
                    o = DCTR11
                elif angle_change<-11.25/180*np.pi and angle_change>=-22.5/180*np.pi:
                    o = DCTR22
                elif angle_change<-22.5/180*np.pi and angle_change>=-45/180*np.pi:
                    o = DCTR45
                elif angle_change<-45/180*np.pi and angle_change>=-90/180*np.pi:
                    o = DCTR90
                elif angle_change<-90/180*np.pi:
                    o = DCTR180
            elif prev_dis < cur_dis:
                if angle_change == 0:
                    o = DFTN
                elif angle_change>0 and angle_change<=11.25/180*np.pi:
                    o = DFTL11
                elif angle_change>11.25/180*np.pi and angle_change<=22.5/180*np.pi:
                    o = DFTL22
                elif angle_change>22.5/180*np.pi and angle_change<=45/180*np.pi:
                    o = DFTL45
                elif angle_change>45/180*np.pi and angle_change<=90/180*np.pi:
                    o = DFTL90
                elif angle_change>90/180*np.pi:
                    o = DFTL180
                elif angle_change<0 and angle_change>= -11.25/180*np.pi:
                    o = DFTR11
                elif angle_change<-11.25/180*np.pi and angle_change>=-22.5/180*np.pi:
                    o = DFTR22
                elif angle_change<-22.5/180*np.pi and angle_change>=-45/180*np.pi:
                    o = DFTR45
                elif angle_change<-45/180*np.pi and angle_change>=-90/180*np.pi:
                    o = DFTR90
                elif angle_change<-90/180*np.pi:
                    o = DFTR180
            else:
                if angle_change == 0:
                    o = DNTN
                elif angle_change>0 and angle_change<=11.25/180*np.pi:
                    o = DNTL11
                elif angle_change>11.25/180*np.pi and angle_change<=22.5/180*np.pi:
                    o = DNTL22
                elif angle_change>22.5/180*np.pi and angle_change<=45/180*np.pi:
                    o = DNTL45
                elif angle_change>45/180*np.pi and angle_change<=90/180*np.pi:
                    o = DNTL90
                elif angle_change>90/180*np.pi:
                    o = DNTL180
                elif angle_change<0 and angle_change>= -11.25/180*np.pi:
                    o = DNTR11
                elif angle_change<-11.25/180*np.pi and angle_change>=-22.5/180*np.pi:
                    o = DNTR22
                elif angle_change<-22.5/180*np.pi and angle_change>=-45/180*np.pi:
                    o = DNTR45
                elif angle_change<-45/180*np.pi and angle_change>=-90/180*np.pi:
                    o = DNTR90
                elif angle_change<-90/180*np.pi:
                    o = DNTR180

        return o

    def act(self, a):
        ''' Go different directions then observe (calculate change of distance) '''
        prev_pos = np.copy(self.pos)
        prev_angle = np.copy(self.phi)
        if self.distance() > self.source_size:
            if self.representation == DISCRETE_BOTH:
                if a == GO_LEFT_1125:
                    self.theta += 11.25/180*np.pi
                elif a == GO_LEFT_225:
                    self.theta += 22.5/180*np.pi
                elif a == GO_LEFT_45:
                    self.theta += 45/180*np.pi
                elif a == GO_LEFT_90:
                    self.theta += 90/180*np.pi
                elif a == GO_LEFT_180:
                    self.theta += np.pi
                elif a == GO_RIGHT_90:
                    self.theta -= 90/180*np.pi
                elif a == GO_RIGHT_45:
                    self.theta -= 45/180*np.pi
                elif a == GO_RIGHT_225:
                    self.theta -= 22.5/180*np.pi
                elif a == GO_RIGHT_1125:
                    self.theta -= 11.25/180*np.pi
            else:
                if a == GO_LEFT:
                    self.theta += self.granularity
                elif a == GO_RIGHT:
                    self.theta -= self.granularity
            self.pos[0] += self.vel * np.cos(self.theta)
            self.pos[1] += self.vel * np.sin(self.theta)
            self.check_bounds(prev_pos)
            self.step_count += 1
        else:
            if CONTINUAL_LEARNING:
                self.steps_episode.append(self.step_count)
                self.episode_count += 1
                self.reset()
                return self.observe(self.pos, self.phi)
        return self.observe(prev_pos, prev_angle)

    def distance(self):
        ''' Distance between the agent and the source '''
        return self.dis(self.pos[0], self.pos[1], self.s_pos[0], self.s_pos[1])

    def check_bounds(self, prev_pos):
        if self.pos[0] > self.env_size:
            self.pos[0] = self.env_size
        if self.pos[0] < 0:
            self.pos[0] = 0
        if self.pos[1] > self.env_size:
            self.pos[1] = self.env_size
        if self.pos[1] < 0:
            self.pos[1] = 0
        if self.dis(self.pos[0], self.pos[1], self.obst_pos[0], self.obst_pos[1]) <= OBSTACLE_SIZE:
            self.pos[:] = prev_pos[:]

    @staticmethod
    def dis(x1, y1, x2, y2):
        ''' Euclidean distance '''
        return np.sqrt(((x1 - x2) * (x1 - x2)) + ((y1 - y2) * (y1 - y2)))

    @staticmethod
    def vec_norm(vec):
        return vec / LA.norm(vec)