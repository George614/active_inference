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
    if len(np.shape(A)) <= 2 and len(np.shape(B)) <= 2:
        return np.dot(A, B)
    # need to add code to handle different dims/range
    # hack dims here
    if (np.size(A, 1)!=np.size(B, 0)) and (np.size(A, 0)==np.size(B, 0)):
        return np.dot(B.T, A)


def spm_softmax(x, k=1):
    '''
    softmax function over columns
    x - numeric array
    k - precision, sensitivity or inverse temperature / variance    
    '''
    x = np.exp(x - np.max(x))
    return x/np.sum(x)


def spm_MDP_G(A, x):
    ''' 
    auxiliary function for Bayesian superise or mutual information
    A - likelyhood array (probability of outcome given causes/states)
    x - probability density of causes
    '''
    # probability distribution over the hidden causes: i.e., Q(x)
    qx = np.asarray(x)
    # accumulate expexctation of entropy. i.e., E[lnP(o|x)]
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

def spm_MDP_VB_X(MDP, options):
    if options is None:
        options = {}
        options['plot'] = 0
        options['gamma'] = 0
    if type(MDP) is list:
        opts = options
        opts['plot'] = 0
        