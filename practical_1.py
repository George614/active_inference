# -*- coding: utf-8 -*-
"""
T-Maze task as a demo for active inference.
First script illustrates trade-off between information gain and maximising
reward.

@author: George (Zhizhuo) Yang
"""
import numpy as np
from spm_MDP_VB_X import spm_MDP_VB_X

## Outcome probabilities (state-outcome mapping): A
# Columns reflect hidden states, while rows reflect observations. Entries
# reflect probabilities for making an observation, given a hidden state
# (o|s). Will be normalised in the routine, but useful to define as 
# probabilities right away.
a = 0.75
b = 1 - a

A = np.array([[1, 1, 0, 0, 0, 0, 0, 0],  # ambiguous starting position (center)
             [0, 0, 1, 1, 0, 0, 0, 0],   # safe arm selected and rewarded
             [0, 0, 0, 0, a, b, 0, 0],   # risky arm selected and rewarded
             [0, 0, 0, 0, b, a, 0, 0],   # risky arm selected and not rewarded
             [0, 0, 0, 0, 0, 0, 1, 0],   # informative cue - high reward probability
             [0, 0, 0, 0, 0, 0, 0, 1]])  # informative cue - low reward probability

## Controlled transitions (state-state mappings): B
# Next, we have to specify the probabilistic transitions of hidden states
# under each action or control state. Here, there are four actions taking the
# agent directly to each of the four locations.
# 
# This is where the Markov property comes in. Transition probabilities only
# depend on the current state and action, not the history of previous
# states or actions.
Ns = 8
Nu = 4
B = np.zeros((Ns, Ns, Nu))
# move to/stay in the middle
B[:, :, 0] = np.array([[1, 0, 0, 0, 0, 0, 1, 0],  # starting point, high reward context
                       [0, 1, 0, 0, 0, 0, 0, 1],  # starting point, low reward context
                       [0, 0, 1, 0, 0, 0, 0, 0],  # safe option, high reward context
                       [0, 0, 0, 1, 0, 0, 0, 0],  # safe option, low reward context
                       [0, 0, 0, 0, 1, 0, 0, 0],  # risky option, high reward context
                       [0, 0, 0, 0, 0, 1, 0, 0],  # risky option, low reward context
                       [0, 0, 0, 0, 0, 0, 0, 0],  # cue location, high reward context
                       [0, 0, 0, 0, 0, 0, 0, 0]]) # cue location, low reward context

# move up left to safe (and check for reward)
B[:, :, 1] = np.array([[0, 0, 0, 0, 0, 0, 0, 0],  # starting point, high reward context
                       [0, 0, 0, 0, 0, 0, 0, 0],  # starting point, low reward context
                       [1, 0, 1, 0, 0, 0, 1, 0],  # safe option, high reward context
                       [0, 1, 0, 1, 0, 0, 0, 1],  # safe option, low reward context
                       [0, 0, 0, 0, 1, 0, 0, 0],  # risky option, high reward context
                       [0, 0, 0, 0, 0, 1, 0, 0],  # risky option, low reward context
                       [0, 0, 0, 0, 0, 0, 0, 0],  # cue location, high reward context
                       [0, 0, 0, 0, 0, 0, 0, 0]]) # cue location, low reward context

# move up right to risky (and check for reward)
B[:, :, 2] = np.array([[0, 0, 0, 0, 0, 0, 0, 0],  # starting point, high reward context
                       [0, 0, 0, 0, 0, 0, 0, 0],  # starting point, low reward context
                       [0, 0, 1, 0, 0, 0, 0, 0],  # safe option, high reward context
                       [0, 0, 0, 1, 0, 0, 0, 0],  # safe option, low reward context
                       [1, 0, 0, 0, 1, 0, 1, 0],  # risky option, high reward context
                       [0, 1, 0, 0, 0, 1, 0, 1],  # risky option, low reward context
                       [0, 0, 0, 0, 0, 0, 0, 0],  # cue location, high reward context
                       [0, 0, 0, 0, 0, 0, 0, 0]]) # cue location, low reward context

# move down and check cue
B[:, :, 3] = np.array([[0, 0, 0, 0, 0, 0, 0, 0],  # starting point, high reward context
                       [0, 0, 0, 0, 0, 0, 0, 0],  # starting point, low reward context
                       [0, 0, 1, 0, 0, 0, 0, 0],  # safe option, high reward context
                       [0, 0, 0, 1, 0, 0, 0, 0],  # safe option, low reward context
                       [0, 0, 0, 0, 1, 0, 0, 0],  # risky option, high reward context
                       [0, 0, 0, 0, 0, 1, 0, 0],  # risky option, low reward context
                       [1, 0, 0, 0, 0, 0, 1, 0],  # cue location, high reward context
                       [0, 1, 0, 0, 0, 0, 0, 1]]) # cue location, low reward context

## Priors
# Now, we have to specify the prior preferences in terms of log
# probabilities. Here, the agent prefers rewarding outcomes
# 
# This is a vector that has the same length as the number of observable
# outcomes (rows of A-matrix). Entries reflect preferences for these
# observations (higher number means higher preferences). These numbers go
# through a softmax in the routine.
cs = 2**1     # preference for safe option
cr = 2**2     # preference for risky option
# prior preference for [staying at starting point | safe | risky + reward | 
# risky + no reward | cue context 1 | cue context 2]
C = np.array([[0, cs, cr, -cs, 0, 0]])
C = C.T

# prior beliefs about the initial state
D = np.kron(np.array([1/2, 0, 0, 0]), np.array([1, 1]))
D = np.expand_dims(D, axis=-1)

## Allowable policies (sequences of actions)
# number of rows    = number of time steps
# number of columns = number of allowed policies
# 
# numbers == actions
# 0 == go to starting point
# 1 == go to safe option
# 2 == go to risky option
# 3 == go to cue location
V = np.array([[0, 0, 0, 0, 1, 2, 3, 3, 3, 3],
              [0, 1, 2, 3, 1, 2, 0, 1, 2, 3]])

## Define MDP Structure
mdp = {}
mdp['V'] = V      # allowable polices
mdp['A'] = A      # observation model (state-outcome mapping)
mdp['B'] = B      # transition probabilities 
mdp['C'] = C      # preferred states 
mdp['D'] = D      # prior over initial states
mdp['s'] = 0      # initial state (0 == high reward context, 1 == low reward context)
mdp['eta'] = 0.5  # learning rate

## Tasks with a random context (hidden state)
# Fairly precise behaviour WITH information-gain
# number of simulated trials
n = 32                       # number of trials
idx = np.where(np.random.rand(n) > 1/2)[0]  # randomise hidden states over trials
MDP = [mdp.copy() for _ in range(n)]        # replicate mdp structure over trials
for i in idx:
    MDP[i]['s'] = 1          # randomise initial state
for MDP_ in MDP:
    MDP_['alpha'] = 16       # precision of action selection

MDP  = spm_MDP_VB_X(MDP)


