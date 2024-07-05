# -*- coding: utf-8 -*-
"""
Created on Wed Nov  4 09:15:22 2020

@author: George
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

TRAIN_STEPS = 1000

if __name__ == "__main__":

    # initialize agents
    full_agent1 = core.get_mdp(FULL_ID)
    full_agent2 = core.get_mdp(FULL_ID)
       
    # learn each type of single agent individually 
    full_agent1, full_agent2, full_record1, full_record2 = core.compare_agents(full_agent1, full_agent2, TRAIN_STEPS)
       
    # save record of trial information for all agents
    np.save("../data/full_single_record_1.npy", full_record1)
    np.save("../data/full_single_record_2.npy", full_record2)

    print("> Data saved")
    
    steps = np.arange(full_record1["steps"])
    
    ### generate an animation file for the full (E.F.E) agent's trajectory during the trial ###
    
    metadata = dict(title='Full agent trajectory', artist='George Yang',
                comment='Animation for a single full (E.F.E) agent during 1 trial')
    writer = FFMpegWriter(fps=15, metadata=metadata)
    
    pos_full1 = full_record1['position']
    pos_full2 = full_record2['position']
    theta_full1 = full_record1['orientation'].squeeze()
    theta_full2 = full_record2['orientation'].squeeze()
    
    fig = plt.figure()
    line1, = plt.plot([], [], 'b-o', markersize=2)
    line2, = plt.plot([], [], 'k-D', markersize=2)
    line3, = plt.plot([], [], 'g-o', markersize=2)
    line4, = plt.plot([], [], 'y-D', markersize=2)
    scat = plt.scatter(250, 250, s=25**2, alpha=0.2)

    agent_size = AGENT_SIZE

    plt.xlim(0, ENVIRONMENT_SIZE)
    plt.ylim(0, ENVIRONMENT_SIZE)

    with writer.saving(fig, "../videos/full_agent_compare.mp4", 500):
        for i in range(len(pos_full1)):
            # back position
            x1 = pos_full1[i, 0]
            y1 = pos_full1[i, 1]
            x2 = pos_full2[i, 0]
            y2 = pos_full2[i, 1]
            # calculate front position
            fx1 = x1 + agent_size * np.cos(theta_full1[i])
            fy1 = y1 + agent_size * np.sin(theta_full1[i])
            fx2 = x2 + agent_size * np.cos(theta_full2[i])
            fy2 = y2 + agent_size * np.sin(theta_full2[i])
            line1.set_data(x1, y1)
            line2.set_data(fx1, fy1)
            line3.set_data(x2, y2)
            line4.set_data(fx2, fy2)
            writer.grab_frame()