# -*- coding: utf-8 -*-
"""
Created on Sun Nov 15 19:22:17 2020

@author: George
"""
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
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
    print("> Creating animation for agent trajectory...")
    metadata = dict(title='Full agent trajectory', artist='George Yang',
                    comment='Animation for a single full (E.F.E) agent during 1 trial')
    writer = FFMpegWriter(fps=ani_fps, metadata=metadata)
    
    pos_trial = record['position']
    s_pos_trial = record['s_pos']
    obst_pos_trial = record['obst_pos']
    theta_trial = record['orientation'].squeeze()
    uQ_trial = record['uQ']
    phi_trial = record['approach_angle']
    prev_obv_trial = record['prev_obv']
    cur_obv_trial = record['cur_obv']
    fully_trained = record["fully_trained"]
    train_steps = record["steps"]
    if fully_trained:
        pos_trial = pos_trial[train_steps:]
        theta_trial = theta_trial[train_steps:]
        s_pos_trial = s_pos_trial[train_steps:]
        obst_pos_trial = obst_pos_trial[train_steps:]
    
    fig = plt.figure()
    line1, = plt.plot([], [], 'b-o', markersize=2)
    line2, = plt.plot([], [], 'k-D', markersize=2)
    line_s, = plt.plot([], [], 'yo', markersize=SOURCE_SIZE, alpha=0.2)
    line_o, = plt.plot([], [], 'go', markersize=OBSTACLE_SIZE, alpha=0.2)
    # text1 = plt.text(0, 510, '', fontsize='small')
    # text2 = plt.text(90, 510, '', fontsize='small')
    # text3 = plt.text(200, 510, '', fontsize='small')
    text4 = plt.text(300, 510, '', fontsize='small')
    text5 = plt.text(0, 530, '', fontsize='small')

    plt.xlim(0, ENVIRONMENT_SIZE)
    plt.ylim(0, ENVIRONMENT_SIZE)

    with writer.saving(fig, os.path.join(path, "intercept_full_agent_{}.mp4".format(n)), 200):
        for i in tqdm(range(len(pos_trial))):
            # back position
            x = pos_trial[i, 0]
            y = pos_trial[i, 1]
            # calculate front position
            fx = x + AGENT_SIZE * np.cos(theta_trial[i])
            fy = y + AGENT_SIZE * np.sin(theta_trial[i])
            line1.set_data(x, y)
            line2.set_data(fx, fy)
            # source position
            s_pos = s_pos_trial[i, :]
            obst_pos = obst_pos_trial[i, :]
            line_s.set_data(s_pos[0], s_pos[1])
            line_o.set_data(obst_pos[0], obst_pos[1])
            # maxU = np.argmax(uQ_trial[i])
            prev_obv = np.squeeze(prev_obv_trial[i])
            cur_obv = np.squeeze(cur_obv_trial[i])
            # text2.set_text('Go straight: %.2f' % uQ_trial[i, 0])
            # text1.set_text('Go left: %.2f' % uQ_trial[i, 1])
            # text3.set_text('Go right: %.2f' % uQ_trial[i, 2])
            text4.set_text('Approach angle: %.2f' % (phi_trial[i]/np.pi*180))
            # text5.set_text("Prev_obv/Cur_obv: " + OBV_BOTH_DICT[int(prev_obv)] + " / " + OBV_BOTH_DICT[int(cur_obv)])
            text5.set_text("Prev_obv/Cur_obv: " + OBV_DISCRETE_ARRAY[int(prev_obv)] + " / " + OBV_DISCRETE_ARRAY[int(cur_obv)])
            # text2.set_color('red' if maxU==0 else 'black')
            # text1.set_color('red' if maxU==1 else 'black')
            # text3.set_color('red' if maxU==2 else 'black')

            writer.grab_frame()
    
    print("> Animation for agent trajectory saved for agent #{}.".format(n))


def create_heatmap(matrix, path, color_bar=True):
    print("> Creating states distribution plot...")
    plt.rc("text", usetex=True)
    f, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(4, 7))
    ax1.set_title('Go Straight', fontsize=7, loc='left')
    ax2.set_title('Go Left', fontsize=7, loc='left')
    ax3.set_title('Go Right', fontsize=7, loc='left')

    if np.shape(matrix)[1] == 9:
        x_labels = [r"$s_{t-1}^{dntn}$", r"$s_{t-1}^{dntt}$", r"$s_{t-1}^{dnta}$",
                    r"$s_{t-1}^{dctn}$", r"$s_{t-1}^{dctt}$", r"$s_{t-1}^{dcta}$",
                    r"$s_{t-1}^{dftn}$", r"$s_{t-1}^{dftt}$", r"$s_{t-1}^{dfta}$"]
        y_labels = [r"$s_{t}^{dntn}$", r"$s_{t}^{dntt}$", r"$s_{t}^{dnta}$",
                    r"$s_{t}^{dctn}$", r"$s_{t}^{dctt}$", r"$s_{t}^{dcta}$",
                    r"$s_{t}^{dftn}$", r"$s_{t}^{dftt}$", r"$s_{t}^{dfta}$"]
    elif np.shape(matrix)[1] == 3:
        x_labels = [r"$s_{t-1}^{none}$", r"$s_{t-1}^{closer}$", r"$s_{t-1}^{farther}$"]
        y_labels = [r"$s_{t}^{none}$", r"$s_{t}^{closer}$", r"$s_{t}^{farther}$"]
    
    g1 = sns.heatmap(
        matrix[0, :, :] * 100,
        cmap="OrRd",
        ax=ax1,
        vmin=0.0,
        vmax=30.0,
        linewidth=2,
        annot=True,
        fmt='.1f',
        xticklabels=x_labels,
        yticklabels=y_labels,
        cbar=color_bar,
    )
    g2 = sns.heatmap(
        matrix[1, :, :] * 100,
        cmap="OrRd",
        ax=ax2,
        vmin=0.0,
        vmax=30.0,
        linewidth=2,
        annot=True,
        fmt='.1f',
        xticklabels=x_labels,
        yticklabels=y_labels,
        cbar=color_bar,
    )
    g3 = sns.heatmap(
        matrix[2, :, :] * 100,
        cmap="OrRd",
        ax=ax3,
        vmin=0.0,
        vmax=30.0,
        linewidth=2,
        annot=True,
        fmt='.1f',
        xticklabels=x_labels,
        yticklabels=y_labels,
        cbar=color_bar,
    )
    g1.set_yticklabels(g1.get_yticklabels(), rotation=0, fontsize=6)
    g1.set_xticklabels(g1.get_xticklabels(), rotation=0, fontsize=6)
    g2.set_yticklabels(g2.get_yticklabels(), rotation=0, fontsize=6)
    g2.set_xticklabels(g2.get_xticklabels(), rotation=0, fontsize=6)
    g3.set_yticklabels(g3.get_yticklabels(), rotation=0, fontsize=6)
    g3.set_xticklabels(g3.get_xticklabels(), rotation=0, fontsize=6)
    
    f.tight_layout()
    f.savefig(os.path.join(path, "states_dist.pdf"), dpi=400, bbox_inches="tight")
    print("> Saved states distribution plot.")