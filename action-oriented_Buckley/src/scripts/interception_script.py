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
import animate_energy_plots as apt

sys.path.append(os.getcwd() + '/..')
import core
from core.config import *

TRAIN_STEPS = 4500
N_AGENTS = 3

run_time = []  # runtime for full agents across trials

if __name__ == "__main__":
    print("Starting experiments with {} as observation, training {} agents...".format(CHANGE_DICT[OBV_OPTION], N_AGENTS))
    pathlist = os.getcwd().split('\\')
    path = os.path.join(pathlist[0], os.sep, *pathlist[1:-1], "data", CHANGE_DICT[OBV_OPTION])
    if CONTINUAL_LEARNING:
        path = path + "_continual_learning"
    if not os.path.isdir(path):
        os.makedirs(path)
        print("Created folder: ", path)
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
        np.save(os.path.join(path, "full_single_record_{}.npy".format(n)), full_record)
        np.save(os.path.join(path, "inst_single_record_{}.npy".format(n)), inst_record)
        np.save(os.path.join(path, "epis_single_record_{}.npy".format(n)), epis_record)
        np.save(os.path.join(path, "rand_single_record_{}.npy".format(n)), rand_record)
        print("> Data saved for agent #{}.".format(n))
          
        # generate visualizations and save as pdf and mp4
        apt.plot_energy(full_record, n, path)
        apt.animate_energy_plots(full_record, n, path)
        apt.animate_trajectory(full_record, n, path)
        
        run_time.append(full_record["runtime"])
    
    runtime_avg_scaled = np.mean(run_time) / TRAIN_STEPS * 1000
    print("> Finished all experiments with scaled-average runtime ", runtime_avg_scaled)