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

import matplotlib.pyplot as plt
from multiprocessing import Pool

sys.path.append(os.getcwd() + '/..')
import core
import animate_energy_plots as apt
from core.config import *

TRAIN_STEPS = 5000
TEST_STEPS = 0
N_AGENTS = 1


## handle path and create folder if necessary
pathlist = os.getcwd().split(os.sep)
path = os.path.join(pathlist[0], os.sep, *pathlist[1:-1], "data", CHANGE_DICT[OBV_OPTION])
if CONTINUAL_LEARNING and TEST_STEPS<=0:
    # path = path + "_continual_learning_circle"
    path = path + "_continual_learning_rhombus_cone"
elif CONTINUAL_LEARNING and TEST_STEPS>0:
    path = path + "_shut_learning"
if not os.path.isdir(path):
    os.makedirs(path)
    print("Created folder: ", path)


def run_exp_parallel(n):
    ''' wrapper function to train multiple agents of various types in parallel '''
    print("> Processing agent #{}.".format(n))
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
    
    ## generate visualizations and save as pdf and mp4 ##
    apt.plot_energy(full_record, n, path)
    # apt.animate_energy_plots(full_record, n, path)
    apt.animate_trajectory(full_record, n, path)
    
    return full_record, inst_record, epis_record, rand_record
    

if __name__ == "__main__":
    print("Starting experiments with {} as observation, training {} agents...".format(CHANGE_DICT[OBV_OPTION], N_AGENTS))

    ## run experiments using multiprocessing package for parallel computation
    n_processes = min(N_AGENTS, os.cpu_count())
    with Pool(processes=n_processes) as pool:
        results = pool.map(run_exp_parallel, range(N_AGENTS))
        
    full_records = [agent[0] for agent in results]
    inst_records = [agent[1] for agent in results]
    epis_records = [agent[2] for agent in results]
    rand_records = [agent[3] for agent in results]
    run_time = [full_record["runtime"] for full_record in full_records] # runtime for full agents across trials
    ## create a heapmap from ensembled states distributions of all agents  ##
    states_dists = [full_record["states_dist"] for full_record in full_records]
    states_sum = np.sum(np.stack(states_dists, axis=0), axis=0)
    states_dist = states_sum / np.sum(states_sum)
    states_dist = np.round(states_dist, 3)
    apt.create_heatmap(states_dist, path)
    
    ## Calculate and visualize statistics of steps needed per episode for ##
    ## agents to reach the goal ##
    # episode_steps_full = [full_record["steps_episode"] for full_record in full_records]
    # episode_steps_inst = [inst_record["steps_episode"] for inst_record in inst_records]
    # episode_steps_epis = [epis_record["steps_episode"] for epis_record in epis_records]
    # len_full = [len(steps) for steps in episode_steps_full]
    # steps_arr_full = [steps[:min(len_full)] for steps in episode_steps_full]
    # steps_arr_full = np.asarray(steps_arr_full)
    # mean_full = np.mean(steps_arr_full, axis=0)
    # std_full = np.std(steps_arr_full, axis=0)
    # plt.errorbar(np.arange(1, min(len_full)+1), mean_full, std_full, linestyle='None', marker='^')
    # plt.xlabel("Current episode")
    # plt.ylabel("Steps")
    # plt.show()

    ## save results for all agents of all types ##
    np.save(os.path.join(path, "full_records.npy"), np.stack(full_records))
    np.save(os.path.join(path, "inst_records.npy"), np.stack(inst_records))
    np.save(os.path.join(path, "epis_records.npy"), np.stack(epis_records))
    np.save(os.path.join(path, "rand_records.npy"), np.stack(rand_records))
    print("> All data saved in " + path)
    runtime_avg_scaled = np.mean(run_time) / TRAIN_STEPS * 1000
    print("> Finished all experiments with scaled-average runtime ", runtime_avg_scaled)