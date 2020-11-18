import numpy as np


class MDP(object):
    def __init__(self, a, b, c, lr=0.1, alpha=1, beta=1):

        self.A = a  # likelihood matrix (state-observation mapping) (N_OBS, N_OBS)
        self.B = b  # transition matrix (control, state-state transitions) (N_CONTROL, N_STATES, N_STATES)
        self.C = c  # prior preference / goal  (N_OBS, 1)

        self.alpha = alpha  # instrumental weight 
        self.beta = beta    # epistemic weight
        self.lr = lr        # learning rate
        self.p0 = np.exp(-16)  # epsilon to avoid numberical overflow

        if np.size(self.C, 1) > np.size(self.C, 0):
            self.C = self.C.T

        self.Ns = self.A.shape[1]  # number of hidden states
        self.No = self.A.shape[0]  # number of observations
        self.Nu = self.B.shape[0]  # number of control states

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
        self.utility = np.zeros([self.Nu, 1])
        self.surprise = np.zeros([self.Nu, 1])

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
            fs = np.dot(self.B[u], self.sQ)  # phy_s_tau in equation 18
            fo = np.dot(self.A, fs)          # phi_o in equation 16
            fo = self.normdist(fo + self.p0)

            # instrumental value E_Q(o)[lnP(o)], equation 16?
            utility = (np.sum(fo * np.log(fo / self.C), axis=0)) * self.alpha
            utility = utility[0]
            
            # parameter epistemic value, equation 15 and 18
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
        for st in range(self.Ns):  # s_tau
            for s in range(self.Ns):  # s_t
                surprise += fs[st] * wb[st, s] * self.sQ[s]  # equation 18 first half
        return -surprise

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
