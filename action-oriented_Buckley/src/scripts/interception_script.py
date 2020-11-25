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
import matplotlib.pyplot as plt

sys.path.append(os.getcwd() + '/..')
import core
from core.config import *

TRAIN_STEPS = 5000
TEST_STEPS = 0
N_AGENTS = 100

run_time = []  # runtime for full agents across trials
episode_steps_full = []  # steps used per episode for all full agents
episode_steps_inst = []  # steps used per episode for all full agents
episode_steps_epis = []  # steps used per episode for all full agents
full_records = []
inst_records = []
epis_records = []
rand_records = []

if __name__ == "__main__":
    print("Starting experiments with {} as observation, training {} agents...".format(CHANGE_DICT[OBV_OPTION], N_AGENTS))
    pathlist = os.getcwd().split(os.sep)
    path = os.path.join(pathlist[0], os.sep, *pathlist[1:-1], "data", CHANGE_DICT[OBV_OPTION])
    if CONTINUAL_LEARNING and TEST_STEPS<=0:
        path = path + "_continual_learning_circle"
    elif CONTINUAL_LEARNING and TEST_STEPS>0:
        path = path + "_shut_learning"
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
        full_agent, full_record = core.learn_record_trial(full_agent, TRAIN_STEPS, TEST_STEPS)
        inst_agent, inst_record = core.learn_record_trial(inst_agent, TRAIN_STEPS, TEST_STEPS)
        epis_agent, epis_record = core.learn_record_trial(epis_agent, TRAIN_STEPS, TEST_STEPS)
        rand_agent, rand_record = core.learn_record_trial(rand_agent, TRAIN_STEPS, TEST_STEPS)
        
        # save record of trial information for all agents
        full_records.append(full_record)
        inst_records.append(inst_record)
        epis_records.append(epis_record)
        rand_records.append(rand_record)
        # np.save(os.path.join(path, "full_single_record_{}.npy".format(n)), full_record)
        # np.save(os.path.join(path, "inst_single_record_{}.npy".format(n)), inst_record)
        # np.save(os.path.join(path, "epis_single_record_{}.npy".format(n)), epis_record)
        # np.save(os.path.join(path, "rand_single_record_{}.npy".format(n)), rand_record)
        # print("> Data saved for agent #{}.".format(n))
          
        # generate visualizations and save as pdf and mp4
        # apt.plot_energy(full_record, n, path)
        # apt.animate_energy_plots(full_record, n, path)
        # apt.animate_trajectory(full_record, n, path)
        
        run_time.append(full_record["runtime"])
        episode_steps_full.append(full_record["steps_episode"])
        episode_steps_inst.append(inst_record["steps_episode"])
        episode_steps_epis.append(epis_record["steps_episode"])
        print("> Steps used per episode for EFE agent #{}".format(n), full_record["steps_episode"])
    
    len_full = [len(steps) for steps in episode_steps_full]
    steps_arr_full = [steps[:min(len_full)] for steps in episode_steps_full]
    steps_arr_full = np.asarray(steps_arr_full)
    mean_full = np.mean(steps_arr_full, axis=0)
    std_full = np.std(steps_arr_full, axis=0)
    plt.errorbar(np.arange(1, min(len_full)+1), mean_full, std_full, linestyle='None', marker='^')
    plt.xlabel("Current episode")
    plt.ylabel("Steps")
    plt.show()

    np.save(os.path.join(path, "full_records.npy"), np.stack(full_records))
    np.save(os.path.join(path, "inst_records.npy"), np.stack(inst_records))
    np.save(os.path.join(path, "epis_records.npy"), np.stack(epis_records))
    np.save(os.path.join(path, "rand_records.npy"), np.stack(rand_records))
    runtime_avg_scaled = np.mean(run_time) / TRAIN_STEPS * 1000
    print("> Finished all experiments with scaled-average runtime ", runtime_avg_scaled)