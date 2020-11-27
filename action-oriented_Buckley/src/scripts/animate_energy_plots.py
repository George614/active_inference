# -*- coding: utf-8 -*-
"""
Created on Sun Nov 15 19:22:17 2020

@author: George
"""
import matplotlib.pyplot as plt 
from matplotlib.animation import FuncAnimation, FFMpegWriter
import numpy as np
import time
import sys
import os

sys.path.append(os.getcwd() + '/..')
from core.config import *

ani_fps = 15

def plot_energy(record, n, path):
    '''
    Plot E.F.E and its components under 3 control states respectively
    and save them as PDF format for later use in paper.
    record: numpy list which stores all information of a trial
    n: index of the trial
    '''
    efe = record["EFE"]
    epistemic = record["epistemic"]
    instrumental = record["instrumental"]
    fully_trained = record["fully_trained"]
    train_steps = record["steps"]
    if fully_trained:
        efe = efe[train_steps:]
        epistemic = epistemic[train_steps:]
        instrumental = instrumental[train_steps:]
    steps = np.arange(len(efe))
    efe_straight = efe[:, GO_STRAIGHT]
    efe_left = efe[:, GO_LEFT] 
    efe_right = efe[:, GO_RIGHT]
    inst_straight = instrumental[:, GO_STRAIGHT]
    inst_left = instrumental[:, GO_LEFT]
    inst_right = instrumental[:, GO_RIGHT]
    epis_straight = epistemic[:, GO_STRAIGHT]
    epis_left = epistemic[:, GO_LEFT]
    epis_right = epistemic[:, GO_RIGHT]
    fig, ax = plt.subplots()
    ax.plot(steps, efe_straight, lw=1, linestyle="--", label="EFE go straight")
    ax.plot(steps, efe_left, lw=1, linestyle="--", label="EFE go left")
    ax.plot(steps, efe_right, lw=1, linestyle="--", label="EFE go right")
    ax.legend()
    plt.xlabel('Time steps')
    plt.ylabel('Free energy in bits')
    plt.title("Full (E.F.E) agent")
    fig.savefig(os.path.join(path, 'Full_control_energy_{}.pdf'.format(n)), dpi=600, bbox_inches="tight")
    
    fig, ax = plt.subplots()
    ax.plot(steps, inst_straight, lw=1, linestyle="--", label="Instrumental go straight")
    ax.plot(steps, inst_left, lw=1, linestyle="--", label="Instrumental go left")
    ax.plot(steps, inst_right, lw=1, linestyle="--", label="Instrumental go right")
    ax.legend()
    plt.xlabel('Time steps')
    plt.ylabel('Free energy in bits')
    plt.title("Instrumental component")
    fig.savefig(os.path.join(path, 'Instrumental_control_energy_{}.pdf'.format(n)), dpi=600, bbox_inches="tight")
    
    fig, ax = plt.subplots()
    ax.plot(steps, epis_straight, lw=1, linestyle="--", label="Epistemic go straight")
    ax.plot(steps, epis_left, lw=1, linestyle="--", label="Epistemic go left")
    ax.plot(steps, epis_right, lw=1, linestyle="--", label="Epistemic go right")
    ax.legend()
    plt.xlabel('Time steps')
    plt.ylabel('Free energy in bits')
    plt.title("Epistemic component")
    fig.savefig(os.path.join(path, 'Epistemic_control_energy_{}.pdf'.format(n)), dpi=600, bbox_inches="tight")
    print("> Plots saved for agent #{}.".format(n))

def animate_energy_plots(record, n, path):
    '''
    Generate animated plots for EFE and its components for all control
    states and save as mp4 format.
    record: numpy list which stores all information of a trial
    n: index of the trial
    '''
    efe = record["EFE"]
    epistemic = record["epistemic"]
    instrumental = record["instrumental"]
    fully_trained = record["fully_trained"]
    train_steps = record["steps"]
    if fully_trained:
        efe = efe[train_steps:]
        epistemic = epistemic[train_steps:]
        instrumental = instrumental[train_steps:]
    steps = len(efe)
    start_time = time.perf_counter()
    fig = plt.figure(constrained_layout=True)
    ymax = max(efe.max(), epistemic.max(), instrumental.max())
    ymin = min(efe.min(), epistemic.min(), instrumental.min())
    ax = plt.axes(xlim=(0, steps), ylim=(ymin, ymax)) 
    line1, = ax.plot([], [], lw=1, linestyle="-", label="EFE go straight")
    line2, = ax.plot([], [], lw=1, linestyle="-", label="EFE go left")
    line3, = ax.plot([], [], lw=1, linestyle="-", label="EFE go right")
    line4, = ax.plot([], [], lw=1, linestyle="-.", label="epistemic go straight")
    line5, = ax.plot([], [], lw=1, linestyle="-.", label="epistemic go left")
    line6, = ax.plot([], [], lw=1, linestyle="-.", label="epistemic go right")
    line7, = ax.plot([], [], lw=1, linestyle="--", label="instrumental go straight")
    line8, = ax.plot([], [], lw=1, linestyle="--", label="instrumental go left")
    line9, = ax.plot([], [], lw=1, linestyle="--", label="instrumental go right")
    ax.legend(bbox_to_anchor=(0., 1.12, 1., .102), loc='lower left',
               ncol=3, mode="expand", fontsize='small', borderaxespad=0.)
    plt.xlabel('Time steps')
    plt.ylabel('Free energy in bits')
    plt.title("Full agent's EFE and its decomposition")
    
    # initialization function 
    def init(): 
        # creating an empty plot/frame 
        line1.set_data([], [])
        line2.set_data([], [])
        line3.set_data([], [])
        line4.set_data([], [])
        line5.set_data([], [])
        line6.set_data([], [])
        line7.set_data([], [])
        line8.set_data([], [])
        line9.set_data([], [])
        return line1, line2, line3, line4, line5, line6, line7, line8, line9 
    
    # lists to store x and y axis points 
    efe_straight_trial = []
    efe_left_trial = []
    efe_right_trial = []
    epis_straight_trial = []
    epis_left_trial = []
    epis_right_trial = []
    inst_straight_trial = []
    inst_left_trial = []
    inst_right_trial = []
    xdata = []
    
    # animation function 
    def animate(i): 
        # x, y values to be plotted 
        x = i 
        efe_straight = efe[i, GO_STRAIGHT]
        efe_left = efe[i, GO_LEFT]
        efe_right = efe[i, GO_RIGHT]
        epis_straight = epistemic[i, GO_STRAIGHT]
        epis_left = epistemic[i, GO_LEFT]
        epis_right = epistemic[i, GO_RIGHT]
        inst_straight = instrumental[i, GO_STRAIGHT]
        inst_left = instrumental[i, GO_LEFT]
        inst_right = instrumental[i, GO_RIGHT]
        
        # appending new points to x, y axes points list 
        xdata.append(x)
        efe_straight_trial.append(efe_straight)
        efe_left_trial.append(efe_left)
        efe_right_trial.append(efe_right)
        epis_straight_trial.append(epis_straight)
        epis_left_trial.append(epis_left)
        epis_right_trial.append(epis_right)
        inst_straight_trial.append(inst_straight)
        inst_left_trial.append(inst_left)
        inst_right_trial.append(inst_right)
        line1.set_data(xdata, efe_straight_trial)
        line2.set_data(xdata, efe_left_trial)
        line3.set_data(xdata, efe_right_trial)
        line4.set_data(xdata, epis_straight_trial)
        line5.set_data(xdata, epis_left_trial)
        line6.set_data(xdata, epis_right_trial)
        line7.set_data(xdata, inst_straight_trial)
        line8.set_data(xdata, inst_left_trial)
        line9.set_data(xdata, inst_right_trial)
        
        return line1, line2, line3, line4, line5, line6, line7, line8, line9
    
    ani = FuncAnimation(fig, animate, init_func=init, frames=steps, blit=True)
    ani.save(os.path.join(path, "energy_animation_{}.mp4".format(n)), fps=ani_fps, dpi=200)
    print("> Runtime for animating energy plots: {} s".format(time.perf_counter()-start_time))
    print("> Animation of energy saved for agent #{}.".format(n))

def animate_trajectory(record, n, path):
    '''
    Generate an animation file for the full (E.F.E) agent's trajectory 
    during 1 trial.
    record: numpy list which stores all information of a trial
    n: index of the trial
    '''
    metadata = dict(title='Full agent trajectory', artist='George Yang',
                    comment='Animation for a single full (E.F.E) agent during 1 trial')
    writer = FFMpegWriter(fps=ani_fps, metadata=metadata)
    
    pos_full = record['position']
    s_pos_full = record['s_pos']
    theta_full = record['orientation'].squeeze()
    fully_trained = record["fully_trained"]
    train_steps = record["steps"]
    if fully_trained:
        pos_full = pos_full[train_steps:]
        theta_full = theta_full[train_steps:]
        s_pos_full = s_pos_full[train_steps:]
    
    fig = plt.figure()
    line1, = plt.plot([], [], 'b-o', markersize=2)
    line2, = plt.plot([], [], 'k-D', markersize=2)
    line_s, = plt.plot([], [], 'yo', markersize=SOURCE_SIZE, alpha=0.2)

    plt.xlim(0, ENVIRONMENT_SIZE)
    plt.ylim(0, ENVIRONMENT_SIZE)

    with writer.saving(fig, os.path.join(path, "intercept_full_agent_{}.mp4".format(n)), 200):
        for i in range(len(pos_full)):
            # back position
            x = pos_full[i, 0]
            y = pos_full[i, 1]
            # calculate front position
            fx = x + AGENT_SIZE * np.cos(theta_full[i])
            fy = y + AGENT_SIZE * np.sin(theta_full[i])
            line1.set_data(x, y)
            line2.set_data(fx, fy)
            # source position
            s_pos = s_pos_full[i, :]
            line_s.set_data(s_pos[0], s_pos[1])
            writer.grab_frame()
    
    print("> Animation for agent trajectory saved for agent #{}.".format(n))