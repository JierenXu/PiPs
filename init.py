#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 20 17:07:53 2018

@author: jieren
"""



import numpy as np


class init(object):
    # parameters for GP regression
    def __init__(self, Sig,config,init_Freq):
        self.b = config.b
        self.l = config.l
        
        # initialize patterns as sin
        self.sh1 = lambda x: np.cos(2*np.pi*x)
        self.sh2 = lambda x: np.sin(2*np.pi*x)
        
        # design PIPs
        self.Z = np.arange(0,self.b,1./self.l)
        self.z = np.arange(0,1.,1./self.l)
        U = np.zeros([2,self.b*self.l])
        U[0,:] = self.sh1(self.Z)
        U[1,:] = self.sh2(self.Z)
        self.U = U
        self.u = U[:,0:self.l]
        
        self.init_Freq = init_Freq
        


class Trig_ConstantFreq(init):
    def __init__(self, Sig, config,init_Freq):
        super(Trig_ConstantFreq, self).__init__(Sig,config,init_Freq)
        # main model parameters       
        int_F1 = self.init_Freq   ## raw estimated frequency for mode 1
        int_F2 = self.init_Freq   ## raw estimated frequency for mode 2
        
        # initialize key variables 
        B0 = np.ones([config.K,config.D+1])
        B0[0,1] = int_F1
        B0[1,1] = int_F2
        B0 = np.float32(B0)
        C0 = np.ones([config.K,config.Dc+1])
        C0 = np.float32(C0)
        print(B0)
        self.B0 = B0
        self.C0 = C0
        self.c = np.max([int_F1,int_F2])
        self.n = int(config.N/config.T/self.c*(config.b/2) )
        
        
def get_init(name, Sig,config,init_Freq):
    try:
        return globals()[name](Sig,config,init_Freq)
    except KeyError:
        raise KeyError("Equation for the required problem not found.")

