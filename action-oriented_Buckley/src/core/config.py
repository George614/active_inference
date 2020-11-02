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

TUMBLE = 0
RUN = 1
NEG_GRADIENT = 0
POS_GRADIENT = 1

##############################
#       Agent config        #
##############################

FULL_ID = 0
INST_ID = 1
EPIS_ID = 2
RAND_ID = 3
AGENT_NAMES = ["E.F.E", "Instrumental", "Epistemic", "Random"]
N_AGENTS = 4

##############################
#        MDP config          #
##############################

N_OBS = 2
N_CONTROL = 2
N_STATES = 2

N_DISTRIBUTIONS = 4
TUMBLE_NEG_ID = 0
TUMBLE_POS_ID = 1
RUN_NEG_ID = 2
RUN_POS_ID = 3

PRIOR_ID = 1
ALPHA = 1 / 10
LR = 0.005

##############################
#       File config          #
##############################

DISTANCE_PATH = "src/data/distance.npy"
ACCURACY_PATH = "src/data/accuracy.npy"
STEPS_PATH = "src/data/steps.npy"

STATES_PATH = "src/data/states.npy"
REVERSED_STATES_PATH = "src/data/reversed_states.npy"
COMPLEXITY_PATH = "src/data/complexity.npy"
REVERSED_COMPLEXITY_PATH = "src/data/reversed_complexity.npy"

ACTIVE_ACCURACY_PATH = "src/data/active_accuracy.npy"
PASSIVE_ACCURACY_PATH = "src/data/passive_accuracy.npy"

PRUNED_PATH = "src/data/pruned.npy"

FAILED_PATH = "src/data/failed.npy"

##############################
#       Figures config       #
##############################

DISTANCE = "src/figs/distance.pdf"
ACCURACY = "src/figs/accuracy.pdf"

FULL_STATES = "src/figs/full_states.pdf"
INST_STATES = "src/figs/inst_states.pdf"
EPIS_STATES = "src/figs/epis_states.pdf"
RAND_STATES = "src/figs/rand_states.pdf"
COLOR_BAR = "src/figs/color_bar.pdf"

FULL_STATES_REVERSED = "src/figs/full_states_reversed.pdf"
INST_STATES_REVERSED = "src/figs/inst_states_reversed.pdf"
EPIS_STATES_REVERSED = "src/figs/epis_states_reversed.pdf"
RAND_STATES_REVERSED = "src/figs/rand_states_reversed.pdf"

FULL_PRUNED = "src/figs/full_pruned.pdf"
INST_PRUNED = "src/figs/inst_pruned.pdf"
EPIS_PRUNED = "src/figs/epis_pruned.pdf"
RAND_PRUNED = "src/figs/rand_pruned.pdf"
COLOR_BAR_PRUNED = "src/figs/color_bar_pruned.pdf"
TOTAL_PRUNED = "src/figs/total_pruned.pdf"

COMPLEXITY = "src/figs/complexity.pdf"
REVERSED_COMPLEXITY = "src/figs/reversed_complexity.pdf"

ACTIVE_ACCURACY = "src/figs/active_accuracy.pdf"

FAILED_MODELS = "src/figs/failed_models.pdf"


##############################
#        Plot config         #
##############################


def get_color_palette():
    palette = sns.color_palette("Paired", 12)
    _colors = [palette[5], palette[7], palette[0], palette[2]]
    return _colors
