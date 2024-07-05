# -*- coding: utf-8 -*-
"""
Created on Sat Sep 19 17:42:26 2020

Python implementation of functions used in active inference with POMDP and 
Variational Bayesian methods in the SPM software.

@author: George (Zhizhuo) Yang
"""

import numpy as np
from numpy import linalg as LA
from scipy.special import digamma


def spm_log(A):
    'log of numeric array plus a small constant (to avoid numerical error)'
    return np.log(A + 1e-16)


def spm_norm(A):
    'normalization of a probability transition matrix by column'
    if len(np.shape(A)) == 2:
        sum_ = np.sum(A, axis=0)
        A[:, sum_>0] = A[:, sum_>0] / sum_[sum_>0]
        A[:, sum_==0] = 1 / np.size(A, axis=0)
    #TODO add support for more dimensions later
    return A


def spm_cum(A):
    'summation of a probability transition matrix by column'
    if len(np.shape(A)) == 2:
        return np.sum(A, axis=0, keepdims=True)
    #TODO add support for more dimensions later


def spm_psi(A):
    'normalization of a probability transition rate matrix by columns'
    return digamma(A) - digamma(np.sum(A, axis=0, keepdims=True))
    

def spm_cross(A, B):
    'multidimensional outer product'
    if len(np.shape(A)) <= 2 and len(np.shape(B)) <= 2:
        return np.outer(A, B)
    #TODO add support for more dimensions later


def spm_dot(A, B, dims=None):
    'multidimensional dot (inner) product'
    # need to add code to handle different dims/range
    # hack dims here
    if (np.size(A, 1)!=np.size(B, 0)) and (np.size(A, 0)==np.size(B, 0)):
        return np.transpose(np.dot(B.T, A))
    if len(np.shape(A))<=2 and len(np.shape(B))<=2 and np.size(A, 1)==np.size(B, 0):
        return np.dot(A, B)

def spm_softmax(x, k=1):
    '''
    softmax function over columns
    x - numeric array
    k - precision, sensitivity or inverse temperature / variance    
    '''
    x = np.exp(x - np.max(x, axis=0))
    return x/np.sum(x, axis=0)


def spm_MDP_G(A, x):
    ''' 
    auxiliary function for Bayesian superise or mutual information
    A - likelyhood array (probability of outcome given causes/states)
    x - probability density of causes (states)
    '''
    # probability distribution over the hidden causes: i.e., Q(x)
    qx = np.asarray(x)
    # accumulate expexctation of uncertainty / entropy. i.e., E[lnP(o|x)]
    # i.e. Expected Ambiguity
    G = 0
    qo = 0
    for i in np.where(np.squeeze(qx) > np.exp(-16))[0]:
        # probability over outcomes for this combination of causes
        po = 1
        po = spm_cross(po, A[:, i])
        qo = qo + qx[i] * po.T
        G  = G + qx[i] * np.dot(po, spm_log(po.T))
    # subtract entropy of expectations. i.e., E[lnQ(o)]
    G = G - np.dot(qo.T, spm_log(qo))
    return G

def spm_MDP_VB_X(MDP, options=None):
    '''
    active inference and learning using variational Bayes (factorised)
    
    '''
    ### deal with a sequence of trials ###
    # options
    if options is None:
        options = {}
        options['plot'] = 0
        options['gamma'] = 0
    # if there are multiple trials ensure that parameters are updated
    if type(MDP) is list:
        opts = options
        opts['plot'] = 0
        OUT = []
        for i in range(len(MDP)):
            # update concentration parameters
            if i>0:
                try:
                    MDP[i]['a'] = OUT[i-1]['a']
                except:
                    pass
                try:
                    MDP[i]['b'] = OUT[i-1]['b']
                except:
                    pass
                try:
                    MDP[i]['c'] = OUT[i-1]['c']
                except:
                    pass
            # solve this trial
            OUT.append(spm_MDP_VB_X(MDP[i], opts))
            # Bayesian model reduction (to be added in the future)
            
        # plot summary statistics over trials
        #TODO add plotting code later
        return OUT
    ### set up and preliminaries ###
    try:
        T = MDP['T']  # number of updates (time steps)
        V = MDP['U']  # allowable actions (1,Np)
    except:
        V = MDP['V']  # allowable policies (T-1, Np)
        T = np.size(V, axis=0) + 1  # number of transitions (time steps)
    # ensure policy length is less than the number of updates
    if np.size(V, axis=0) > T-1:
        V = V[:T-1, :]
    # number of transitions, policies and states
    Np = np.size(V, axis=1)         # number of allowable policies
    Ns = np.size(MDP['B'], axis=0)  # number of hidden states
    Nu = np.size(MDP['B'], axis=2)  # number of hidden controls (actions)
    No = np.size(MDP['A'], axis=0)  # number of outcomes
    
    ### parameters of generative model and polices ###
    # likelihood model (for a POMDP implicit in G)
    p0 = np.exp(-16)    # epsilon value to avoid numerical error
    MDP['A'] = spm_norm(MDP['A'])
    # parameters (concentration parameters): A (later development for learning it)
    A = MDP['A']
    # transition probabilities (priors)
    sB = np.zeros_like(MDP['B'])
    rB = np.zeros_like(MDP['B'])
    for j in range(Nu):
        # controlable transition probabilities
        MDP['B'][:,:,j] = spm_norm(MDP['B'][:,:,j])
        # parameters (concentration parameters): B (later development for learning it)
        sB[:,:,j] = spm_norm(MDP['B'][:,:,j] + p0)      # s_t   -> s_t+1
        rB[:,:,j] = spm_norm(MDP['B'][:,:,j].T + p0)    # s_t+1 -> s_t 
    # priors over initial hidden states - concentration parameters
    if 'd' in MDP:
        D = spm_norm(MDP['d'])
    elif 'D' in MDP:
        D = spm_norm(MDP['D'])
    else:
        D = spm_norm(np.ones((Ns, 1)))
        MDP['D'] = D
    # prior prefrences (log probabilities): C
    if 'C' in MDP:
        Vo = MDP['C']
    else:
        Vo = np.zeros((No, 1))
    # assume constant prefrences, if only final states are specified
    if np.size(Vo, axis=1)==1:
        Vo = np.repeat(Vo, T, axis=1)
    Vo = spm_log(spm_softmax(Vo))
    try:
        ambiguity = MDP['ambiguity']
    except:
        ambiguity = True
    try:
        curiosity = MDP['curiosity']
    except:
        curiosity = True
    try:
        alpha = MDP['alpha']
    except:
        alpha = 16
    try:
        beta = MDP['beta']
    except:
        beta = 1
    try:
        eta = MDP['eta']
    except:
        eta = 1
    try:
        tau = MDP['tau']
    except:
        tau = 4
    try:
        chi = MDP['chi']
    except:
        chi = 1/64
    # initialize posteriors over states
    Ni = 16  # number of VB iterations
    xn = np.zeros((Ni, Ns, T, T, Np)) + 1/Ns
    vn = np.zeros((Ni, Ns, T, T, Np))
    x  = np.zeros((Ns, T, Np)) + 1/Ns
    X  = np.repeat(D, T, axis=1)
    for k in range(Np):
        x[:, 0, k] = D.squeeze()
    # initialize posteriors over policies and action
    P  = np.zeros((Nu, T-1))                 # probability of action at time 1,...,T - 1
    wn = np.zeros((T*Ni,))                   # simulated neuronal encoding of precision
    un = np.zeros((Np, T*Ni))                # simulated neuronal encoding of policies
    u  = np.zeros((Np, T))                   # conditional expectations over policies
    a  = np.zeros((T-1,), dtype=np.int)      # action at 1,...,T-1
    # expected rate parameter (precisions? need to figure out)
    p = np.arange(Np, dtype=np.int)          # allowable policies
    qbeta = beta                             # initialize rate parameters
    gu = [1/qbeta] * T                       # posterior precision (policy)
    
    #### solve ####
    s = np.zeros((T,), dtype=np.int) - 1     # states (initialized to be invalid)
    o = np.zeros((T,), dtype=np.int) - 1     # outcomes (initialized to be invalid)
    O = [None] * T                           # vector representation of outcomes
    for t in range(T):
        ## generate true states and outcomes
        # sampled state - based on previous action
        if t==0:
            s[t] = MDP['s']                  # read initial state from MDP strcture
        else:
            if t > 0:
                ps = MDP['B'][:, s[t-1], a[t-1]] # probability of sampled states
            else:
                ps = spm_norm(MDP['D'])
            s[t] = np.where(np.random.rand() < np.cumsum(ps))[0][0] # sampled state
        # sample outcome from true state if not specified
        try:
            o[t] = MDP['o'][t]
        except:
            po = MDP['A'][:, s[t]]
            o[t] = np.where(np.random.rand() < np.cumsum(po))[0][0]
        # posterior predictive density over states (prior for subordinate level)
        if t>0:
            xq = sB[:, :, a[t-1]] @ X[:, t-1] #TODO debug this line!
        else:
            xq = X[:, t]
        o_temp = np.zeros((No, 1))
        o_temp[o[t], 0] = 1
        O[t] = o_temp
        
        ## variational updates
        # reset
        x = spm_softmax(spm_log(x) / 4)  # why /4 ?
        
        ## variational updates (hidden states) under sequential policies
        S = np.size(V, 0) + 1    # time steps
        F = np.zeros((Np, 1))    # free energy
        G = np.zeros((Np, 1))    # expected free energy
        for k in p:
            dF = 1
            for i in range(Ni):
                F[k] = 0
                for j in range(S):
                    # marginal likelihood over outcome factors
                    if j<=t:
                        xq = np.expand_dims(x[:, j, k], axis=-1)
                        Ao = spm_dot(A, O[j])    # outcome map to state
                    # hidden states for this time and policy
                    sx = np.expand_dims(x[:, j, k], axis=-1)
                    v  = np.zeros_like(sx) #TODO  spm_zeros()
                    # evaluate free energy and gradients (v = dFdx)
                    if dF>0:
                        # marginal likelihood over outcome factors
                        if j<=t:
                            v = v + spm_log(Ao)
                        # entropy
                        qx = spm_log(sx)
                        # emperical priors
                        if j==0:
                            v = v - qx + spm_log(D)
                        if j>0:
                            v = v - qx + spm_log(sB[:, :, V[j-1, k]] @ x[:, j-1, k:k+1]) # Bayesian filtering
                        if j<S-1:
                            v = v - qx + spm_log(rB[:, :, V[j, k]] @ x[:, j+1, k:k+1])   # Bayesian smoothing
                        # (negative) expected free energy
                        F[k] = F[k] + sx.T @ v
                        # update
                        sx = spm_softmax(qx + v/tau) # what is tau?
                    else:
                        F[k] = G[k]
                    # store update neuronal activity
                    x[:, j, k] = sx.squeeze()
                    xn[i, :, j, t, k] = sx.squeeze()
                    vn[i, :, j, t, k] = np.squeeze(v - np.mean(v))
                # convergence 
                if i>0:
                    dF = F[k] - G[k]
                G[:] = F[:]
        
        ## accumulate expected free energy of policies (Q)
        Q = np.zeros((Np, 1))
        for k in p:  # number of policies
            for j in range(S):  # number of time steps 
                # get expected states for this policy and time step
                xq = x[:, j, k:k+1]
                ## (negative) expected free energy
                # Bayesian suprise about states
                if ambiguity:
                    Q[k] = Q[k] + spm_MDP_G(A, xq) # epistemic value: hidden state exploration
                # prior preference about outcomes
                qo = spm_dot(A, xq)
                Q[k] = Q[k] + qo.T @ Vo[:, j:j+1] # extrinsic value - reward
        # eliminate unlikely policies
        if 'U' not in MDP:
            p = p[np.squeeze((F[p] - np.max(F[p])) > -3)]
        else:
            options['gamma'] = 1
            
        ## variational updates - policies and precision
        # previous expected precision
        if t>0:
            gu[t] = gu[t-1]
        for i in range(Ni):
            # posterior and prior beliefs about policies
            qu = spm_softmax(gu[t] * Q[p] + F[p])
            pu = spm_softmax(gu[t] * Q[p])
            # precision with free energy gradients (v = -dF/dw)
            if options['gamma']:
                gu[t] = 1/beta
            else:
                eg = (qu - pu).T @ Q[p]  # beliefs about enacted policy times their value
                dFdg = qbeta - beta + eg
                qbeta = qbeta - dFdg/2
                gu[t] = 1/qbeta
            # simulated dopamine responses (precision at each iteration)
            n = t * Ni + i
            wn[n] = gu[t]
            un[p, n] = qu.squeeze()
            u[p, t] = qu.squeeze()
        
        # Bayesian model averaging of hidden states (over policies)
        for i in range(S):
            X[:, i] = np.squeeze(x[:, i, :] @ u[:, t:t+1])
        # record (negative) free energies
        if 'F' not in MDP or 'G' not in MDP:
            MDP['F'] = []
            MDP['G'] = []
        MDP['F'].append(F)
        MDP['G'].append(Q)
        
        # action selection and sampling of next state (outcome)
        if t<T-1:
            # marginal posterior probability of action (for each modality)
            Pu = np.zeros((Nu, 1))
            for i in range(Np):
                sub = V[t, i]
                Pu[sub] = Pu[sub] + u[i, t]
            # action selection - a softmax function of action potential
            Pu = spm_softmax(alpha * np.log(Pu))
            P[:, t] = Pu.squeeze()
            # next action - sampled from marginal posterior
            try:
                a[t] = MDP['u'][t]
            except:
                idx = np.where(np.random.rand() < np.cumsum(Pu))[0][0]
                a[t] = idx
        elif t==T-1:
            break
    
    ### learning ###
    #TODO add learning for state-outcome mapping (A) and state-state mapping (B)
    
    # initial hidden states
    if 'd' in MDP:
        i = MDP['d'] > 0
        MDP['d'][i] = MDP['d'][i] + X[i, 0] * eta
    # simulate dopamine (or cholinergic) responses
    dn = 8 * np.gradient(wn) + wn/8 # why?
    # Bayesian model averaging of expected hidden states over policies
    Xn = np.zeros((Ni, Ns, T, T))
    Vn = np.zeros((Ni, Ns, T, T))
    for i in range(T):
        for k in p:
            Xn[:, :, :, i] = Xn[:, :, :, i] + xn[:, :, :, i, k] * u[k, i]
            Vn[:, :, :, i] = Xn[:, :, :, i] + vn[:, :, :, i, k] * u[k, i]
    # assemble results and place in MDP structure
    MDP['T']   = T              # number of belief updates
    MDP['P']   = P              # probability of action at time 1,...,T - 1
    MDP['Q']   = x              # conditional expectations over N hidden states
    MDP['X']   = X              # Bayesian model averages over T outcomes
    MDP['R']   = u              # conditional expectations over policies
    MDP['V']   = V              # policies
    MDP['o']   = o              # outcomes at 1,...,T
    MDP['s']   = s              # states   at 1,...,T
    MDP['u']   = a              # action   at 1,...,T - 1
    MDP['w']   = gu             # posterior expectations of precision (policy)
    MDP['C']   = Vo             # utility
    
    MDP['un']  = un             # simulated neuronal encoding of policies
    MDP['vn']  = Vn             # simulated neuronal prediction error
    MDP['xn']  = Xn             # simulated neuronal encoding of hidden states
    MDP['wn']  = wn             # simulated neuronal encoding of precision
    MDP['dn']  = dn             # simulated dopamine responses (deconvolved)
    return MDP

