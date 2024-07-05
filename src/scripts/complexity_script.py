import numpy as np
import sys
import os

sys.path.append(os.getcwd() + '/..')
import core
from core.config import *


# calculate KL-divergence between 2 transition matrices
def get_factor_complexity(a, b): 
    kls = np.zeros(4)
    kls[TUMBLE_NEG_ID] = np.sum(a[0, :, 0] * np.log(a[0, :, 0] / b[0, :, 0]), axis=0)
    kls[TUMBLE_POS_ID] = np.sum(a[0, :, 1] * np.log(a[0, :, 1] / b[0, :, 1]), axis=0)
    kls[RUN_NEG_ID] = np.sum(a[1, :, 0] * np.log(a[1, :, 0] / b[1, :, 0]), axis=0)
    kls[RUN_POS_ID] = np.sum(a[1, :, 1] * np.log(a[1, :, 1] / b[1, :, 1]), axis=0)
    return kls


# KL-divergence between initialized model and fully-learned model
def process_agent(agent, reverse_prior=False):
    mdp = core.get_mdp(agent, reverse_prior=reverse_prior)
    original_model = np.copy(mdp.B)  # model with basic initialization
    mdp = core.learn_trial(mdp, TEST_TRIAL_LEN)  # learn a model for extended period of time
    return get_factor_complexity(original_model, mdp.B)


def get_complexity(reverse_prior):
    _complexity = np.zeros([N_AGENTS, N_DISTRIBUTIONS, N_AVERAGES])

    for agent_id in range(N_AGENTS):
        print("> Processing agent: {}".format(AGENT_NAMES[agent_id]))

        for n in range(N_AVERAGES):
            if n % 50 == 0:
                print("> Processing average [{}/{}]".format(n, N_AVERAGES))

            kl_divs = process_agent(agent_id, reverse_prior=reverse_prior)
            _complexity[agent_id, :, n] = kl_divs

    return _complexity


if __name__ == "__main__":
    print("\n> Processing complexity")
    complexity = get_complexity(False)
    print("\n> Processing complexity (reverse prior)")
    reversed_complexity = get_complexity(True)

    np.save(COMPLEXITY_PATH, complexity)
    np.save(REVERSED_COMPLEXITY_PATH, reversed_complexity)
    print("> Data saved")
