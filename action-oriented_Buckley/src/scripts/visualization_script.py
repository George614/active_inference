# -*- coding: utf-8 -*-
"""
Simple trial to record and visualize the behaviour of active inference 
agents in a trial. Single agent (without model averaging) is supported
for now.

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

if __name__ == "__main__":

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
    np.save("../data/full_single_record.npy", full_record)
    np.save("../data/inst_single_record.npy", inst_record)
    np.save("../data/epis_single_record.npy", epis_record)
    np.save("../data/rand_single_record.npy", rand_record)
    print("> Data saved")
    
    # access the free energy values for 2 control states respectively
    efe_run = full_record["EFE"][:, RUN]
    efe_tumble = full_record["EFE"][:, TUMBLE] 
    inst_run = inst_record["EFE"][:, RUN]
    inst_tumble = inst_record["EFE"][:, TUMBLE]
    epis_run = epis_record["EFE"][:, RUN]
    epis_tumble = epis_record["EFE"][:, TUMBLE]
    
    steps = np.arange(full_record["steps"])
    
    ### plot the free energy under 2 control states for 3 agents respectively ###

    fig, ax = plt.subplots()
    ax.plot(steps, efe_run, lw=1, linestyle="--", label="EFE Run")
    ax.plot(steps, efe_tumble, lw=1, linestyle="--", label="EFE Tumble")
    ax.legend()
    plt.xlabel('Time steps')
    plt.ylabel('Free energy in bits')
    plt.title("Full (E.F.E) agent")
    fig.savefig('../figs/Full_control_energy.pdf', dpi=600, bbox_inches="tight")
    plt.show()
    
    fig, ax = plt.subplots()
    ax.plot(steps, inst_run, lw=1, linestyle="--", label="Instrumental Run")
    ax.plot(steps, inst_tumble, lw=1, linestyle="--", label="Instrumental Tumble")
    ax.legend()
    plt.xlabel('Time steps')
    plt.ylabel('Free energy in bits')
    plt.title("Instrumental agent")
    fig.savefig('../figs/Instrumental_control_energy.pdf', dpi=600, bbox_inches="tight")
    plt.show()
    
    fig, ax = plt.subplots()
    ax.plot(steps, epis_run, lw=1, linestyle="--", label="Epistemic Run")
    ax.plot(steps, epis_tumble, lw=1, linestyle="--", label="Epistemic Tumble")
    ax.legend()
    plt.xlabel('Time steps')
    plt.ylabel('Free energy in bits')
    plt.title("Epistemic agent")
    fig.savefig('../figs/Epistemic_control_energy.pdf', dpi=600, bbox_inches="tight")
    plt.show()

    ### generate an animation file for the full (E.F.E) agent's trajectory during the trial ###
    
    metadata = dict(title='Full agent trajectory', artist='George Yang',
                comment='Animation for a single full (E.F.E) agent during 1 trial')
    writer = FFMpegWriter(fps=15, metadata=metadata)
    
    pos_full = full_record['position']
    theta_full = full_record['orientation'].squeeze()
    
    fig = plt.figure()
    line1, = plt.plot([], [], 'b-o', markersize=2)
    # line2, = plt.plot([], [], 'b-', linewidth=2)
    line3, = plt.plot([], [], 'k-D', markersize=2)
    scat = plt.scatter(250, 250, s=25**2, alpha=0.2)

    agent_size = AGENT_SIZE

    plt.xlim(0, ENVIRONMENT_SIZE)
    plt.ylim(0, ENVIRONMENT_SIZE)

    with writer.saving(fig, "../videos/full_agent.mp4", 500):
        for i in range(len(pos_full)):
            # back position
            x = pos_full[i, 0]
            y = pos_full[i, 1]
            # calculate front position
            fx = x + agent_size * np.cos(theta_full[i])
            fy = y + agent_size * np.sin(theta_full[i])
            line1.set_data(x, y)
            line3.set_data(fx, fy)
            writer.grab_frame()