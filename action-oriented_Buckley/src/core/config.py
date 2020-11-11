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
INIT_DISTANCE = 400
SOURCE_SIZE = 25
AGENT_SIZE = 5
VELOCITY = 1
GRANULARITY = 30/180*np.pi

# action/control ID
GO_STRAIGHT = 0
GO_LEFT = 1
GO_RIGHT = 2
# Obsercation/state ID
VISUAL_R0 = 0
VISUAL_R1 = 1
VISUAL_R2 = 2
VISUAL_R3 = 3
VISUAL_R4 = 4 
VISUAL_R5 = 5

PREFERENCE_R0 = 0
PREFERENCE_R1 = 2
PREFERENCE_R2 = 4
PREFERENCE_R3 = 2
PREFERENCE_R4 = 0
PREFERENCE_R5 = -2

##############################
#       Agent config        #
##############################

FULL_ID = 0
INST_ID = 1
EPIS_ID = 2
RAND_ID = 3
AGENT_NAMES = ["E.F.E", "Instrumental", "Epistemic", "Random"]
N_AGENTS = 4

# vision parameters
FOV = 120
VISUAL_BOUNDS = [-FOV/2, -20, -0.5, 0.5, 20, FOV/2]
VISUAL_BOUNDS = [bound/180*np.pi for bound in VISUAL_BOUNDS]
VISUAL_RANGES = [[VISUAL_BOUNDS[i], VISUAL_BOUNDS[i+1]] for i in range(len(VISUAL_BOUNDS)-1)]

##############################
#        MDP config          #
##############################

N_OBS = 6
N_CONTROL = 3
N_STATES = 6

N_DISTRIBUTIONS = 18

ALPHA = 1 / 10
LR = 0.005

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
