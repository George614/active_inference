import numpy as np
from .env import Environment
from .config import *


def learn_trial(mdp, n_steps, record_states=False):
    env = Environment()
    obv = env.observe()
    mdp.reset(obv)
    states = np.zeros([N_CONTROL, N_STATES, N_STATES])

    for step in range(n_steps):
        prev_obv = obv  # save the previous observation
        action = mdp.step(obv)  # calculate the action to be performed using the old observation
        obv = env.act(action)  # execute the action and generate new observation
        mdp.update(action, obv, prev_obv)  # learning of the agent in MDP
        if record_states:
            states[action, obv, prev_obv] += 1 # count the numbers in transition matrix

    if record_states:
        return mdp, states
    return mdp


def learn_record_trial(mdp, n_steps, record_states=True):
    env = Environment()
    obv = env.observe()
    mdp.reset(obv)
    # record the distribution of states
    states_dist = np.zeros((N_CONTROL, N_STATES, N_STATES))
    # values of EFE for all control states throughout a trial (to be recorded)
    EFE_trial = np.zeros((n_steps, N_CONTROL))
    epistemic_trial = np.zeros((n_steps, N_CONTROL))
    instrumental_trial = np.zeros((n_steps, N_CONTROL))
    pos_trial = np.zeros((n_steps, 2))
    theta_trial = np.zeros((n_steps, 1))

    for step in range(n_steps):
        # execute routine in a step
        prev_obv = obv
        action = mdp.step(obv)
        obv = env.act(action)
        mdp.update(action, obv, prev_obv)
        # record the values
        EFE_trial[step, :] = np.squeeze(mdp.EFE[:])
        epistemic_trial[step, :] = np.squeeze(mdp.surprise[:])
        instrumental_trial[step, :] = np.squeeze(mdp.utility[:])
        pos_trial[step, :] = env.pos[:]
        theta_trial[step, :] = env.theta
        if record_states:
            states_dist[action, obv, prev_obv] += 1
    
    record_dict = {"steps" : n_steps,
                    "EFE" : EFE_trial,
                    "epistemic" : epistemic_trial,
                    "instrumental" : instrumental_trial,
                    "position" : pos_trial,
                    "orientation" : theta_trial}
    if record_states:
        record_dict["states_dist"] : states_dist

    return mdp, record_dict


def test_distance(mdp, steps):
    env = Environment()
    obv = env.observe()
    mdp.reset(obv)

    for _ in range(steps):
        action = mdp.step(obv)
        obv = env.act(action)

    return (env.distance() - env.source_size) + 1


def test_passive_accuracy(mdp, n_steps):
    env = Environment()
    obv = env.observe()
    mdp.reset(obv)
    acc = 0

    for _ in range(n_steps):
        random_action = np.random.choice([0, 1]) # passive means take random action
        pred, t_pred = mdp.predict_obv(random_action, obv)
        _ = mdp.step(obv)
        obv = env.act(random_action)
        acc += diff(t_pred, pred)

    return acc


def test_active_accuracy(mdp, n_steps):
    env = Environment()
    obv = env.observe()
    mdp.reset(obv)
    acc = 0

    for _ in range(n_steps):
        action = mdp.step(obv)  # active means take action based on E.F.E
        pred, t_pred = mdp.predict_obv(action, obv)
        acc += diff(t_pred, pred)
        obv = env.act(action)

    return acc


def diff(p, q):
    return np.mean(np.square(p - q))
