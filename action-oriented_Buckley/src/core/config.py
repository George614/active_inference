import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

##############################
#     Simulation config      #
##############################

N_AVERAGES = 200
TEST_TRIAL_LEN = 2000
MODEL_REDUCTION_TRIAL_LEN = 500

##############################
#     Environment config    #
##############################

ENVIRONMENT_SIZE = 500
INIT_DISTANCE = ENVIRONMENT_SIZE/2
SOURCE_SIZE = 25
OBSTACLE_SIZE = 20
AGENT_SIZE = 5
VELOCITY = 1
GRANULARITY = 30/180*np.pi

# TUMBLE = 0
# RUN = 1
# NEG_GRADIENT = 0
# POS_GRADIENT = 1

CHANGE_DISTANCE = 0
CHANGE_ANGLE = 1
CHANGE_BOTH = 2

CHANGE_DICT = {0 : "change_of_distance", 1 : "change_of_apporach_angle", 2 : "change_of_both"}
# select the representation of observation / hidden state before running any experiments
OBV_OPTION = CHANGE_BOTH

# action/control ID
GO_STRAIGHT = 0
GO_LEFT = 1
GO_RIGHT = 2

# Obsercation/state ID
CHANGE_NONE = 0
CHANGE_CLOSER = 1
CHANGE_FARTHER = 2

TURN_NONE = 0
TURN_TOWARDS = 1
TURN_AWAY = 2

DIS_NONE_TURN_NONE = 0
DIS_NONE_TURN_TOWARDS = 1
DIS_NONE_TURN_AWAY = 2
DIS_CLOSER_TURN_NONE = 3
DIS_CLOSER_TURN_TOWARDS = 4
DIS_CLOSER_TURN_AWAY = 5
DIS_FARTHER_TURN_NONE = 6
DIS_FARTHER_TURN_TOWARDS = 7
DIS_FARTHER_TURN_AWAY = 8

OBV_BOTH_DICT = {0 : "DIS_NONE_TURN_NONE", 1 : "DIS_NONE_TURN_TOWARDS", 2 : "DIS_NONE_TURN_AWAY",
				3 : "DIS_CLOSER_TURN_NONE", 4 : "DIS_CLOSER_TURN_TOWARDS", 5 : "DIS_CLOSER_TURN_AWAY",
				6 : "DIS_FARTHER_TURN_NONE", 7 : "DIS_FARTHER_TURN_TOWARDS", 8 : "DIS_FARTHER_TURN_AWAY"}
##############################
#       Agent config        #
##############################

FULL_ID = 0
INST_ID = 1
EPIS_ID = 2
RAND_ID = 3
AGENT_NAMES = ["E.F.E", "Instrumental", "Epistemic", "Random"]
N_AGENTS = 4

CONTINUAL_LEARNING = True
##############################
#        MDP config          #
##############################

if OBV_OPTION == CHANGE_DISTANCE or OBV_OPTION == CHANGE_ANGLE:
	N_OBS = 3
	N_CONTROL = 3
	N_STATES = 3
	N_DISTRIBUTIONS = 9
	PRIOR_ID = 1
elif OBV_OPTION == CHANGE_BOTH:
	N_OBS = 9
	N_CONTROL = 3
	N_STATES = 9
	N_DISTRIBUTIONS = 27
	PRIOR_ID = 3
else:
	raise ValueError("Incorrect choise of observation/hidden state representation.")

ALPHA = 1 / 10
LR = 0.005


# TUMBLE_NEG_ID = 0
# TUMBLE_POS_ID = 1
# RUN_NEG_ID = 2
# RUN_POS_ID = 3

##############################
#       File config          #
##############################

DISTANCE_PATH = "../data/distance.npy"
ACCURACY_PATH = "../data/accuracy.npy"
STEPS_PATH = "../data/steps.npy"

STATES_PATH = "../data/states.npy"
REVERSED_STATES_PATH = "../data/reversed_states.npy"
COMPLEXITY_PATH = "../data/complexity.npy"
REVERSED_COMPLEXITY_PATH = "../data/reversed_complexity.npy"

ACTIVE_ACCURACY_PATH = "../data/active_accuracy.npy"
PASSIVE_ACCURACY_PATH = "../data/passive_accuracy.npy"

PRUNED_PATH = "../data/pruned.npy"

FAILED_PATH = "../data/failed.npy"

##############################
#       Figures config       #
##############################

DISTANCE = "../figs/distance.pdf"
ACCURACY = "../figs/accuracy.pdf"

FULL_STATES = "../figs/full_states.pdf"
INST_STATES = "../figs/inst_states.pdf"
EPIS_STATES = "../figs/epis_states.pdf"
RAND_STATES = "../figs/rand_states.pdf"
COLOR_BAR = "../figs/color_bar.pdf"

FULL_STATES_REVERSED = "../figs/full_states_reversed.pdf"
INST_STATES_REVERSED = "../figs/inst_states_reversed.pdf"
EPIS_STATES_REVERSED = "../figs/epis_states_reversed.pdf"
RAND_STATES_REVERSED = "../figs/rand_states_reversed.pdf"

FULL_PRUNED = "../figs/full_pruned.pdf"
INST_PRUNED = "../figs/inst_pruned.pdf"
EPIS_PRUNED = "../figs/epis_pruned.pdf"
RAND_PRUNED = "../figs/rand_pruned.pdf"
COLOR_BAR_PRUNED = "../figs/color_bar_pruned.pdf"
TOTAL_PRUNED = "../figs/total_pruned.pdf"

COMPLEXITY = "../figs/complexity.pdf"
REVERSED_COMPLEXITY = "../figs/reversed_complexity.pdf"

ACTIVE_ACCURACY = "../figs/active_accuracy.pdf"

FAILED_MODELS = "../figs/failed_models.pdf"


##############################
#        Plot config         #
##############################


def get_color_palette():
    palette = sns.color_palette("Paired", 12)
    _colors = [palette[5], palette[7], palette[0], palette[2]]
    return _colors
