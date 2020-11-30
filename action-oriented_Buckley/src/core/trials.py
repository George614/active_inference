import numpy as np
from .env import Environment
from .config import *
import copy
import time

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


def learn_record_trial(mdp, n_steps, test_steps=None, env=None, record_states=True):
    if env is None:
        env = Environment()
    if test_steps is not None and not CONTINUAL_LEARNING:
        raise ValueError("Cannot run fully-trained agent without continual learning!")
    if test_steps is None:
        test_steps = 0
    cur_pos = env.pos
    cur_phi = env.phi
    obv = env.observe(cur_pos, cur_phi)
    mdp.reset(obv)
    # record the distribution of states
    states_dist = np.zeros((N_CONTROL, N_STATES, N_STATES))
    # values of EFE for all control states throughout a trial (to be recorded)
    EFE_trial = np.zeros((n_steps + test_steps, N_CONTROL))
    epistemic_trial = np.zeros((n_steps + test_steps, N_CONTROL))
    instrumental_trial = np.zeros((n_steps + test_steps, N_CONTROL))
    uQ_trial = np.zeros((n_steps + test_steps, N_CONTROL))  # probabilities for all control states
    pos_trial = np.zeros((n_steps + test_steps, 2))  # agent's position
    s_pos_trial = np.zeros((n_steps + test_steps, 2))  # source position
    theta_trial = np.zeros((n_steps + test_steps, 1))  # agent's orientation in the environment
    phi_trial = np.zeros((n_steps + test_steps, 1))   # agent's approach angle to the target
    prev_obv_trial = np.zeros((n_steps + test_steps, 1))  # previous observation

    time_start = time.perf_counter()

    for step in range(n_steps + test_steps):
        # execute routine in a step
        prev_obv = obv
        action = mdp.step(obv)
        obv = env.act(action)
        # train an agent fully then turn off learning and run it
        if step < n_steps:
            mdp.update(action, obv, prev_obv)
        if step == n_steps:
            env.reset()
            mdp.reset(env.observe(env.pos, env.phi))
        # record the values
        EFE_trial[step, :] = np.squeeze(mdp.EFE[:])
        epistemic_trial[step, :] = np.squeeze(mdp.surprise[:])
        instrumental_trial[step, :] = np.squeeze(mdp.utility[:])
        uQ_trial[step, :] = np.squeeze(mdp.uQ[:])
        pos_trial[step, :] = env.pos[:]
        s_pos_trial[step, :] = env.s_pos[:]
        theta_trial[step, :] = env.theta
        phi_trial[step, :] = env.phi
        prev_obv_trial[step, :] = prev_obv
        # if np.argmax(mdp.uQ[:]) != mdp.action:
        # print("action {}, max uQ {}".format(mdp.action, np.argmax(mdp.uQ)))
        if record_states:
            states_dist[action, obv, prev_obv] += 1
    
    time_trial = time.perf_counter() - time_start
    record_dict = {"steps" : n_steps,
                    "EFE" : EFE_trial,
                    "epistemic" : epistemic_trial,
                    "instrumental" : instrumental_trial,
                    "uQ" : uQ_trial,
                    "position" : pos_trial,
                    "s_pos" : s_pos_trial,
                    "orientation" : theta_trial,
                    "approach_angle" : phi_trial,
                    "prev_obv" : prev_obv_trial,
                    "runtime" : time_trial,
                    "fully_trained" : False,
                    "steps_episode" : env.steps_episode,
                    "B" : mdp.B,
                    "Ba" : mdp.Ba,
                    "wB" : mdp.wB}
    if record_states:
        record_dict["states_dist"] = states_dist
    if test_steps > 0 and CONTINUAL_LEARNING:
        record_dict["fully_trained"] = True

    return mdp, record_dict


def compare_agents(mdp1, mdp2, n_steps, record_states=False):
    env1 = Environment()
    env2 = copy.deepcopy(env1)
    mdp1, record_dict1 = learn_record_trial(mdp1, n_steps, env=env1, record_states=record_states)
    mdp2, record_dict2 = learn_record_trial(mdp2, n_steps, env=env2, record_states=record_states)
    
    return mdp1, mdp2, record_dict1, record_dict2

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
