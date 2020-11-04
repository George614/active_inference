
import sys
import os

sys.path.append(os.getcwd() + '/..')
sys.path.append(os.getcwd() + '/src')

print("\nCWD:" + os.getcwd() + "\n")

import numpy as np

import core as core
from core.config import *
from core.env import Environment

if __name__ == "__main__":

    mdp = core.get_mdp(FULL_ID)
    record_states=False
    
    # full,states = learn_trial(full, TRAIN_STEPS,record_states=True)
    # full = learn_trial(full, TRAIN_STEPS,record_states=False)
    #passive_accuracy[FULL_ID, n] = core.test_passive_accuracy(full, TEST_STEPS)

    env = Environment()
    obv = env.observe()
    mdp.reset(obv)
    # states = np.zeros([N_CONTROL, N_STATES, N_STATES])

    n_steps = 300;

    for step in range(n_steps):

        if step % 10 == 0:
            print("> Processing step {}".format(step))

        prev_obv = obv
        
        ##################################################################
        #### mdp.step(mdp,obv)
            
        mdp.obv = obv
        
        ##################################################################
        #### mdp.infer_sQ(obv)

        likelihood = mdp.lnA[obv, :]  # lnP(o_t|s_t, lambda)
        likelihood = likelihood[:, np.newaxis]
        prior = np.dot(mdp.B[mdp.action], mdp.sQ) # lnP(s_t|s_t-1, u_t-1) in equation 11
        
        prior = np.log(prior)
        mdp.sQ = mdp.softmax(likelihood + prior)

        ##################################################################
        #### mdp.evaluate_efe()
        mdp.EFE = np.zeros([mdp.Nu, 1])

        for u in range(mdp.Nu):
            fs = np.dot(mdp.B[u], mdp.sQ)  # phy_s_tau in equation 18
            fo = np.dot(mdp.A, fs)          # phi_o in equation 16
            fo = mdp.normdist(fo + mdp.p0)

            # instrumental value E_Q(o)[lnP(o)], equation 16?
            utility = (np.sum(fo * np.log(fo / mdp.C), axis=0)) * mdp.alpha
            utility = utility[0]
            
            ##################################################################
            #### surprise = mdp.bayesian_surprise(u, fs) * mdp.betas
            # parameter epistemic value, equation 15 and 18
            surprise = 0
            wb = mdp.wB[u, :, :]

            for st in range(mdp.Ns):  # s_tau
                for s in range(mdp.Ns):  # s_t
                    surprise += fs[st] * wb[st, s] * mdp.sQ[s]  # equation 18 first half
                
            surprise = -surprise * mdp.beta

            # equation 15 and 18
            mdp.EFE[u] -= utility
            mdp.EFE[u] += surprise
        

        ##################################################################
        ##### mdp.infer_uq()
        mdp.uQ = mdp.softmax(mdp.EFE)

        ##################################################################
        #### mdp.act()
        hu = max(mdp.uQ)
        options = np.where(mdp.uQ == hu)[0]
        mdp.action = int(np.random.choice(options))


        obv = env.act(mdp.action) # Run or tumble, and return an observation of 0 or 1

        ##################################################################
        #### mdp.update(mdp, obv, prev_obv)

        new = obv;
        previous = prev_obv;

        mdp.Ba[mdp.action, new, previous] += mdp.lr  # equation 13
        b = np.copy(mdp.Ba[mdp.action])
        mdp.B[mdp.action] = mdp.normdist(b)
        
        ##################################################################
        ### mdp.calc_wb()

        wb_norm = np.copy(mdp.Ba)
        wb_avg = np.copy(mdp.Ba)

        for u in range(mdp.Nu):
            for s in range(mdp.Ns):
                wb_norm[u, :, s] = np.divide(1.0, np.sum(wb_norm[u, :, s]))
                wb_avg[u, :, s] = np.divide(1.0, (wb_avg[u, :, s]))
        
        # self.wB is An array encoding uncertainty about the trainsition matrix
        # for hidden states (based on paper and its reference)
        mdp.wB = wb_norm - wb_avg

        if record_states:
            states[mdp.action, obv, prev_obv] += 1 # count the numbers in transition matrix
        

        


