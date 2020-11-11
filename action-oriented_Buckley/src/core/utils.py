import numpy as np
from .config import *
from .mdp import MDP


def get_mdp(agent_id, reverse_prior=False):
    a = np.eye(N_OBS)
    b = np.random.rand(N_CONTROL, N_STATES, N_STATES)
    c = np.asarray([[PREFERENCE_R0, PREFERENCE_R1, PREFERENCE_R2, PREFERENCE_R3, PREFERENCE_R4, PREFERENCE_R5]])
    c = c.T
    c = MDP.softmax(c)

    kwargs = {}
    if agent_id == FULL_ID:
        kwargs = {"alpha": ALPHA, "beta": 1, "lr": LR}
    elif agent_id == INST_ID:
        kwargs = {"alpha": ALPHA, "beta": 0, "lr": LR}
    elif agent_id == EPIS_ID:
        kwargs = {"alpha": 0, "beta": 1, "lr": LR}
    elif agent_id == RAND_ID:
        kwargs = {"alpha": 0, "beta": 0, "lr": LR}

    mdp = MDP(a, b, c, **kwargs)
    return mdp

# deprecated for now
def get_true_model():
    b = np.zeros([N_CONTROL, N_STATES, N_STATES])
    b[TUMBLE, :, :] = np.array([[0.5, 0.5], [0.5, 0.5]])
    b[RUN, :, :] = np.array([[1, 0], [0, 1]])
    b += np.exp(-16)
    return b
