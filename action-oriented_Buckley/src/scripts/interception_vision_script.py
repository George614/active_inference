# -*- coding: utf-8 -*-
"""
Simple trial to record and visualize the behaviour of active inference 
agents in trials of an interception task. Single agent (without model averaging)
is learned and mutilple agents learned separately are saved.

@author: George (Zhizhuo) Yang
"""
import numpy as np
import sys
import os

sys.path.append(os.getcwd() + '/..')

import core
from core.config import *
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.animation import FFMpegWriter

matplotlib.use("Agg")

TRAIN_STEPS = 1500
N_AGENTS = 1


if __name__ == "__main__":
    for n in range(N_AGENTS):
        # initialize agents
        full_agent = core.get_mdp(FULL_ID)
        inst_agent = core.get_mdp(INST_ID)
        epis_agent = core.get_mdp(EPIS_ID)
        rand_agent = core.get_mdp(RAND_ID)
        
        # learn each type of single agent individually 
        full_agent, full_record = core.learn_record_trial(full_agent, TRAIN_STEPS)
        inst_agent, inst_record = core.learn_record_trial(inst_agent, TRAIN_STEPS)
        epis_agent, epis_record = core.learn_record_trial(epis_agent, TRAIN_STEPS)
        rand_agent, rand_record = core.learn_record_trial(rand_agent, TRAIN_STEPS)
        
        # save record of trial information for all agents
        np.save("../data/full_single_vision_{}.npy".format(n), full_record)
        np.save("../data/inst_single_vision_{}.npy".format(n), inst_record)
        np.save("../data/epis_single_vision_{}.npy".format(n), epis_record)
        np.save("../data/rand_single_vision_{}.npy".format(n), rand_record)
        print("> Data saved for agent #{}.".format(n))
        
        # access the free energy values for 2 control states respectively
        efe_straight = full_record["EFE"][:, GO_STRAIGHT]
        efe_left = full_record["EFE"][:, GO_LEFT] 
        efe_right = full_record["EFE"][:, GO_RIGHT]
        inst_straight = full_record["instrumental"][:, GO_STRAIGHT]
        inst_left = full_record["instrumental"][:, GO_LEFT]
        inst_right = full_record["instrumental"][:, GO_RIGHT]
        epis_straight = full_record["epistemic"][:, GO_STRAIGHT]
        epis_left = full_record["epistemic"][:, GO_LEFT]
        epis_right = full_record["epistemic"][:, GO_RIGHT]
        
        steps = np.arange(full_record["steps"])
        visual_angle = np.squeeze(full_record["visual_angle"])
        orientation = np.squeeze(full_record["orientation"])
        
        ### plot the free energy under 2 control states for 3 agents respectively ###
    
        fig, ax = plt.subplots()
        ax.plot(steps, efe_straight, lw=1, linestyle="--", label="EFE go straight")
        ax.plot(steps, efe_left, lw=1, linestyle="--", label="EFE go left")
        ax.plot(steps, efe_right, lw=1, linestyle="--", label="EFE go right")
        ax.legend()
        plt.xlabel('Time steps')
        plt.ylabel('Free energy in bits')
        plt.title("Full (E.F.E) agent")
        fig.savefig('../figs/Full_control_energy_{}.pdf'.format(n), dpi=600, bbox_inches="tight")
        # plt.show()
        
        fig, ax = plt.subplots()
        ax.plot(steps, inst_straight, lw=1, linestyle="--", label="Instrumental go straight")
        ax.plot(steps, inst_left, lw=1, linestyle="--", label="Instrumental go left")
        ax.plot(steps, inst_right, lw=1, linestyle="--", label="Instrumental go right")
        ax.legend()
        plt.xlabel('Time steps')
        plt.ylabel('Free energy in bits')
        plt.title("Instrumental component")
        fig.savefig('../figs/Instrumental_control_energy_{}.pdf'.format(n), dpi=600, bbox_inches="tight")
        # plt.show()
        
        fig, ax = plt.subplots()
        ax.plot(steps, epis_straight, lw=1, linestyle="--", label="Epistemic go straight")
        ax.plot(steps, epis_left, lw=1, linestyle="--", label="Epistemic go left")
        ax.plot(steps, epis_right, lw=1, linestyle="--", label="Epistemic go right")
        ax.legend()
        plt.xlabel('Time steps')
        plt.ylabel('Free energy in bits')
        plt.title("Epistemic component")
        fig.savefig('../figs/Epistemic_control_energy_{}.pdf'.format(n), dpi=600, bbox_inches="tight")
        # plt.show()
        
        ## plot the visual angle ##
        
        fig, ax = plt.subplots()
        ax.plot(steps, visual_angle, lw=1, linestyle="-", label="visual angle")
        ax.plot(steps, orientation, lw=1, linestyle="-", label="orientation")
        ax.legend()
        plt.xlabel("Time steps")
        plt.ylabel("Angles in radius")
        plt.title("Visual angle of the target and orientation of agent")
        fig.savefig("../figs/vision_angles_{}.pdf".format(n), dpi=600, bbox_inches="tight")
        # plt.show()
        
        ### generate an animation file for the full (E.F.E) agent's trajectory during the trial ###
        
        metadata = dict(title='Full agent trajectory', artist='George Yang',
                    comment='Animation for a single full (E.F.E) agent during 1 trial')
        writer = FFMpegWriter(fps=15, metadata=metadata)
        
        pos_full = full_record['position']
        theta_full = full_record['orientation'].squeeze()
        
        fig = plt.figure()
        line1, = plt.plot([], [], 'b-o', markersize=2)
        line2, = plt.plot([], [], 'k-D', markersize=2)
        scat = plt.scatter(250, 250, s=25**2, alpha=0.2)
    
        agent_size = AGENT_SIZE
    
        plt.xlim(0, ENVIRONMENT_SIZE)
        plt.ylim(0, ENVIRONMENT_SIZE)
    
        with writer.saving(fig, "../videos/intercept_vision_full_agent_{}.mp4".format(n), 200):
            for i in range(len(pos_full)):
                # back position
                x = pos_full[i, 0]
                y = pos_full[i, 1]
                # calculate front position
                fx = x + agent_size * np.cos(theta_full[i])
                fy = y + agent_size * np.sin(theta_full[i])
                line1.set_data(x, y)
                line2.set_data(fx, fy)
                writer.grab_frame()
        
        print("> Video saved for agent #{}.".format(n))