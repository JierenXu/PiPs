#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 20 15:55:51 2018

@author: jieren
"""

import numpy as np


class Config(object):
    # parameters for GP regression
    def __init__(self, Sig):
        sparams = {}
        eps   = 10**(-3)
        sigma = 10**(-0.8)
        sparams['eps'] = np.float32(eps)
        sparams['beta'] = np.float32(1.) 
        sparams['alpha'] = np.float32(sigma)
        self.GPparams= sparams
        
        # parameters for RDBR
        rparams = {}
        rparams['maxiter'] = 200.
        rparams['eps_error'] = 1e-6
        rparams['show'] = 0.
        rparams['nknots'] = 5.
        rparams['knotremoval_factor'] = 1.001
        rparams['order'] = 3.
        rparams['eps_diff'] = rparams['eps_error']
        rparams['variance'] = 0.2**2;
        self.RDBRparams = rparams
        
        # PIP parameters
        self.l = 2**6 # sample density of fixed non-oscillatory pattern
        self.b = 2**3 # number of periods of fixed non-oscillator
        self.m = self.l*self.b

class Trig_ConstantFreq(Config):
    def __init__(self, Sig):
        super(Trig_ConstantFreq, self).__init__(Sig)
        # main model parameters 
        signal = Sig['signal']
        time = Sig['time']
        self.T = np.max(time)
        self.N = len(signal)
        self.K = Sig['K']
        self.D = Sig['D']
        self.Dc = 0
        # computing parameters
        opt = {}
        opt['Onechip'] = 1
        opt['Update Amp'] = False 
        opt['Update Pattern'] = False 
        self.opt = opt

        
def get_config(name, Sig):
    try:
        return globals()[name](Sig)
    except KeyError:
        raise KeyError("Equation for the required problem not found.")

