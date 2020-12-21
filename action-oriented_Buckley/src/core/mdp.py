import numpy as np


class MDP(object):
    def __init__(self, a, b, c, lr=0.1, alpha=1, beta=1):

        self.A = a  # likelihood matrix (state-observation mapping) (N_OBS, N_OBS)
        self.B = b  # transition matrix (control, state-state transitions) (N_CONTROL, N_STATES, N_STATES)
        self.C = c  # prior preference / goal  (N_OBS, 1)

        self.alpha = alpha  # instrumental weight 
        self.beta = beta    # epoliciestemic weight
        self.lr = lr        # learning rate
        self.p0 = np.exp(-16)  # epsilon to avoid numberical overflow

        if np.size(self.C, 1) > np.size(self.C, 0):
            self.C = self.C.T

        self.Ns = self.A.shape[1]  # number of hidden states
        self.No = self.A.shape[0]  # number of observations
        self.Nu = self.B.shape[0]  # number of control states
        self.Npi = 3  # number of policies
        self.H = 3  # horizon length

        self.A = self.A + self.p0
        self.A = self.normdist(self.A)
        self.lnA = np.log(self.A)

        self.B = self.B + self.p0
        for u in range(self.Nu):
            self.B[u] = self.normdist(self.B[u])
        self.Ba = np.copy(self.B)
        self.wB = 0   # should be an array of shape (N_CONTROL, N_STATES, N_STATES)
        self.calc_wb()

        self.true_B = self.get_true_model()

        self.C = self.C + self.p0
        self.C = self.normdist(self.C)

        self.sQ = np.zeros([self.Ns, 1])
        self.uQ = np.zeros([self.Nu, 1])
        self.EFE = np.zeros([self.Nu, 1])
        self.all_EFE = np.zeros([self.Npi, self.H, 1])
        self.utility = np.zeros([self.Nu, 1])
        self.surprise = np.zeros([self.Nu, 1])

        self.all_sQ = np.zeros([self.Npi, self.H, self.Ns])
        self.all_obv = np.zeros([self.Npi, self.H], dtype=int)
        self.policies = np.zeros([self.Npi, self.H])

        self.action_range = np.arange(0, self.Nu)
        self.obv = 0
        self.action = 0

    def reset(self, obv):
        self.obv = obv
        likelihood = self.lnA[obv, :]
        likelihood = likelihood[:, np.newaxis]
        self.sQ = self.softmax(likelihood)
        self.action = int(np.random.choice(self.action_range))

    def step(self, obv):
        self.obv = obv
        self.infer_sQ(obv)
        self.evaluate_efe()
        self.infer_uq()
        return self.act()

    # equation 11 nad 12 in paper
    def infer_sQ(self, obv):
        likelihood = self.lnA[obv, :]  # lnP(o_t|s_t, lambda)
        likelihood = likelihood[:, np.newaxis]
        prior = np.dot(self.B[self.action], self.sQ) # lnP(s_t|s_t-1, u_t-1) in equation 11
        prior = np.log(prior)
        self.sQ = self.softmax(likelihood + prior)

    def evaluate_efe(self):
        self.EFE = np.zeros([self.Nu, 1])

        for u in range(self.Nu):
            fs = np.dot(self.B[u], self.sQ)  # phi_s_tau in equation 18
            fo = np.dot(self.A, fs)          # phi_o in equation 16
            fo = self.normdist(fo + self.p0)

            # instrumental value E_Q(o)[lnP(o)], equation 16?
            utility = (np.sum(fo * np.log(fo / self.C), axis=0)) * self.alpha
            utility = utility[0]
            
            # parameter epoliciestemic value, equation 15 and 18
            surprise = self.bayesian_surprise(u, fs) * self.beta

            self.utility[u] = -utility
            self.surprise[u] = surprise

            # equation 15 and 18
            self.EFE[u] -= utility
            self.EFE[u] += surprise

    # equation 14
    def infer_uq(self):
        self.uQ = self.softmax(self.EFE)

    # equation 12? learning of the agent
    def update(self, action, new, previous):
        self.Ba[action, new, previous] += self.lr  # equation 13
        b = np.copy(self.Ba[action])
        self.B[action] = self.normdist(b)
        self.calc_wb()

    ### MDP for a horizon of future steps ###

    def step_horizon(self, obv):
        self.obv = obv
        self.policies = np.random.choice(self.Nu, size=(self.Npi, self.H))
        self.policies[:, 0] = np.arange(self.Nu, dtype=int)
        for t in range(self.H):
            if t==0:
                vec_obv = np.repeat(obv, self.Npi)
            self.infer_sQ_horizon(vec_obv, t)
            vec_obv = self.evaluate_efe_horizon(t)
            self.all_obv[:, t] = np.squeeze(vec_obv)
        self.infer_uQ_horizon()
        return self.act()

    def infer_sQ_horizon(self, vec_obv, t):
        '''
        vec_obv: observation of current time step for all policies
        t: current time step, least value should be 1 (start with the second step)
        '''
        if t==0:  # use the actual values from previous time step
            mat_sQ = np.repeat(np.swapaxes(self.sQ, 0, 1), self.Npi, axis=0)
            vec_u_prev = np.repeat(self.action, self.Npi)
        else:
            # mat_sQ: hidden state parameter phi_s from last time step for all policies
            mat_sQ = self.all_sQ[:, t-1, :]
            # previous control state for all policies
            vec_u_prev = self.policies[:, t-1]
        for pi_i, u in enumerate(vec_u_prev):
            likelihood = self.lnA[int(vec_obv[pi_i]), :]  # lnP(o_t|s_t, lambda)
            likelihood = likelihood[:, np.newaxis]
            prior = np.dot(self.B[u], np.expand_dims(mat_sQ[pi_i, :], axis=1)) # lnP(s_t|s_t-1, u_t-1, theta) in equation 11
            prior = np.log(prior)
            self.all_sQ[pi_i, t, :] = np.squeeze(self.softmax(likelihood + prior))

    def evaluate_efe_horizon(self, t):
        ''' evaluate EFE value for all policies at one step '''
        EFE = np.zeros([self.Npi, 1])
        o_hat = np.zeros([self.Npi,])
        # action (control state) to be evaluated at current time step
        vec_u_t = self.policies[:, t]

        for pi_i, u in enumerate(vec_u_t):
            fs = np.dot(self.B[u], np.expand_dims(self.all_sQ[pi_i, t, :], axis=1))  # phi_s_tau in equation 18
            fo = np.dot(self.A, fs)          # phi_o in equation 16
            fo = self.normdist(fo + self.p0)
            o_hat[pi_i] = np.argmax(fo)

            # instrumental value E_Q(o)[lnP(o)], equation 16?
            utility = (np.sum(fo * np.log(fo / self.C), axis=0)) * self.alpha
            utility = utility[0]
            
            # parameter epoliciestemic value, equation 15 and 18
            surprise = self.bayesian_surprise(u, fs) * self.beta

            # equation 15 and 18
            EFE[pi_i] = surprise - utility
            
        self.all_EFE[:, t, :] = EFE[:, :]
        return o_hat

    def infer_uQ_horizon(self):
        self.uQ = self.softmax(np.sum(self.all_EFE, axis=1))

    def update_horizon(self, new, previous):
        policy_ID = self.action  # works only if setting #policies equals to #actions
        for t in range(self.H):
            action = self.policies[policy_ID, t]
            if t==0:
                self.Ba[action, self.all_obv[policy_ID, t], previous] += self.lr  # equation 13
            elif t==self.H-1:
                self.Ba[action, new, self.all_obv[policy_ID, t-1]] += self.lr
            else:
                self.Ba[action, self.all_obv[policy_ID, t], self.all_obv[policy_ID, t-1]] += self.lr
            b = np.copy(self.Ba[action])
            self.B[action] = self.normdist(b)
        self.calc_wb()
    
    # not used
    def calc_expectation(self):
        for u in range(self.Nu):
            b = np.copy(self.Ba[u])
            self.B[u] = self.normdist(b)
        self.calc_wb()

    # equation 17?
    def calc_wb(self):
        wb_norm = np.copy(self.Ba)
        wb_avg = np.copy(self.Ba)

        for u in range(self.Nu):
            for s in range(self.Ns):
                wb_norm[u, :, s] = np.divide(1.0, np.sum(wb_norm[u, :, s]))
                wb_avg[u, :, s] = np.divide(1.0, (wb_avg[u, :, s]))
        
        # self.wB is An array encoding uncertainty about the trainsition matrix
        # for hidden states (based on paper and its reference)
        self.wB = wb_norm - wb_avg

    # sample action from uQ
    def act(self):
        hu = max(self.uQ)
        options = np.where(self.uQ == hu)[0]
        self.action = int(np.random.choice(options))
        return self.action

    def bayesian_surprise(self, u, fs):
        surprise = 0
        wb = self.wB[u, :, :]
        # for st in range(self.Ns):  # s_tau
        #     for s in range(self.Ns):  # s_t
        #         surprise += fs[st] * wb[st, s] * self.sQ[s]  # equation 18 first half
        # this is a faster implementation
        surprise = np.dot(np.swapaxes(fs, 0, 1), np.dot(wb, self.sQ))
        return np.squeeze(-surprise)

    def predict_obv(self, action, obv):
        _obv = np.zeros([2, 1]) + self.p0
        _obv[obv] = 1
        # phi_s and phi_o calculated by current model
        fs = np.dot(self.B[action], _obv)
        fo = np.dot(self.A, fs)
        fo = self.normdist(fo + self.p0)
        # phi_s and phi_o calculated by perfect model
        tfs = np.dot(self.true_B[action], _obv)
        tfo = np.dot(self.A, tfs)
        tfo = self.normdist(tfo + self.p0)
        return fo, tfo

    @staticmethod
    def entropy(fs):
        fs = fs[:, 0]
        return -np.sum(fs * np.log(fs), axis=0)

    @staticmethod
    def softmax(x):
        x = x - x.max()
        x = np.exp(x)
        x = x / np.sum(x)
        return x

    @staticmethod
    def normdist(x):  # normalize by the sum of rows
        return np.dot(x, np.diag(1 / np.sum(x, 0)))

    @staticmethod
    def get_true_model():
        b = np.zeros([2, 2, 2])
        b[0, :, :] = np.array([[0.5, 0.5], [0.5, 0.5]])
        b[1, :, :] = np.array([[1, 0], [0, 1]])
        b += np.exp(-16)
        return b
