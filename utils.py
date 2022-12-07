#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 20 00:27:31 2018

@author: jieren
"""
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from astropy.stats import sigma_clip, mad_std

def del_all_flags(FLAGS):
    flags_dict = FLAGS._flags()    
    keys_list = [keys for keys in flags_dict]    
    for keys in keys_list:
        FLAGS.__delattr__(keys)
        
def rbf_kernel(_x,_z,sparams,nx,nz):
    _rxx = tf.tile(_x, [nz])
    _rxx = tf.transpose(tf.reshape(_rxx, [nz,nx]), perm = [1,0])
    _rzz = tf.tile(_z, [nx])
    _rzz = tf.reshape(_rzz, [nx,nz])
    _temp = (_rxx - _rzz) #  <<<<<<<<<<<<<<<<<<<<<<
    _dxz = _temp * _temp
    _Ktt = sparams['beta'] * tf.exp(-_dxz/(sparams['alpha']))     
    return _Ktt

def inv_rbf_kernel(Z,sparams):
    m = Z.shape[0]
    rzz = np.tile(Z,[m])
    rzz = np.reshape(rzz,[m,m])
    rxx = rzz.T
    temp = rxx - rzz
    dzz = temp*temp
    Kzz = sparams['beta'] * np.exp(-dzz/(sparams['alpha']))
    iK = np.linalg.inv(Kzz+sparams['eps']*np.eye(m))
    return iK

def phi0_2_phi(new_phi0,C0,t):
    order = new_phi0.shape[1]-1-1
    n = len(t)
    phi_chip   = np.tile(np.transpose(np.asmatrix(new_phi0[:,0])), [1,n])
    omega_chip = np.zeros(phi_chip.shape)
    for ord in np.arange(order+1)+1:
        phi_chip   = phi_chip + np.transpose(np.asmatrix(new_phi0[:,ord])) * t**ord
        omega_chip = omega_chip + np.transpose(np.asmatrix(new_phi0[:,ord])) * t**(ord-1) * ord
    Dc = C0.shape[1]-1-1
    amp_chip   = np.tile(np.transpose(np.asmatrix(C0[:,0])), [1,n])
    for ord in np.arange(Dc+1)+1:
       amp_chip   = amp_chip + np.transpose(np.asmatrix(C0[:,ord])) * t**ord
    return phi_chip,omega_chip,amp_chip

def coef2poly(time,B):
    N = len(time)
    K = B.shape[0]
    D = B.shape[1]
    poly = np.ones([K,N]) * np.matlib.repmat(np.reshape(B[:,0],[K,1]),1,N)
    for ord in np.arange(D-1)+1:
        poly = poly + np.matlib.repmat(time**ord , K,1) * np.matlib.repmat(np.reshape(B[:,ord],[K,1]),1,N)
    return poly

def gen_mode(amp,phase,z,u):
    N = phase.shape[1]
    K = phase.shape[0]
    modes = np.zeros([K,N])
    for k in np.arange(K):
        modes[k] = amp[k] * np.interp(np.mod(phase[k],1.), z, u[k])
    return modes
    


def compute_const_inst_info(est_phi_points,est_omega_points,est_amp_points,est_accuracy):
    K = est_phi_points.shape[1] - 1
    est_p = est_phi_points[:,1:K+1]
    est_o = est_omega_points[:,1:K+1]
    est_a = est_amp_points[:,1:K+1]
    est_t = est_phi_points[:,0]
    T = np.sort(np.unique(est_t))
    NN = len(T)

    ## delete outlier according to the accuracy
    acc = est_accuracy
    y = acc
    filtered_data = sigma_clip(y, sigma=3, maxiters=3, stdfunc=mad_std)
    outlier = filtered_data.mask

    est_p = est_p[~outlier]
    est_o = est_o[~outlier]
    est_a = est_a[~outlier]
    est_t = est_t[~outlier]
    acc = acc[~outlier]
    N = est_p.shape[0]
    #print N

    ## identify modes
    sort_modes = np.argsort(est_o)
    for cc in range(N):
        est_p[cc,:] = est_p[cc,sort_modes[cc,:]]
        est_o[cc,:] = est_o[cc,sort_modes[cc,:]]
        est_a[cc,:] = est_a[cc,sort_modes[cc,:]]
    # deal with cross-over
    cross_diff = -est_o[:,0] + est_o[:,1]
    is_Cross = np.min(cross_diff)
    if is_Cross< 1e-3:
        peaks, _ = find_peaks(-cross_diff, distance=100) 
        cross_inspect = np.ones(N)
        cur = 1
        for cc in np.arange(len(peaks)):
            cur = -cur
            if cc == len(peaks)-1: cross_inspect[peaks[cc]:N] = cur
            else: cross_inspect[peaks[cc]:peaks[cc+1]] = cur
    #    plt.plot(est_o)
        for cc in range(N):
            if cross_inspect[cc] == 1: sort_modes = [0,1]
            else: sort_modes = [1,0]
            est_p[cc,:] = est_p[cc,sort_modes]
            est_o[cc,:] = est_o[cc,sort_modes]
            est_a[cc,:] = est_a[cc,sort_modes]
    #    plt.plot(est_o)

    ## compute robust mean of frequency
    omega = np.zeros(K)
    for kk in range(K):
        est_o_mode = est_o[:,kk]
        y = est_o_mode
        filtered_data = sigma_clip(y, sigma=3, maxiters=3, stdfunc=mad_std)
        outlier = filtered_data.mask
        est_o_mode = est_o_mode[~outlier]
        omega[kk] = np.mean(est_o_mode)
    #print omega

    ## compute robust mean of phase
    t = np.sort(np.unique(est_t))
    n = len(t)
    #print n
    phase = np.zeros([K,NN])
    for kk in range(K):
        est_p_mode = est_p[:,kk]
        y = est_p_mode
        filtered_data = sigma_clip(y, sigma=3, maxiters=3, stdfunc=mad_std)
        outlier = filtered_data.mask
        est_p_mode = est_p_mode[~outlier]

        est_p_mode = est_p_mode - np.floor(est_p_mode)
       # plt.plot(est_t,est_p_mode,'*')
        phi_max = np.zeros(n)
        for nn in np.arange(n):
            ind = np.where( est_t == t[nn])
            ind0  = ind[np.argmin(acc[ind])]
            ind0 = ind0[0]  
            phi_max[nn] = est_p_mode[ind0]
       # plt.plot(phi_max[800:1000],'*') 

        lift_max = phi_max - np.floor(phi_max)
        for nn in np.arange(n-1)+1:
            former = lift_max[nn-1]
            former_int = np.floor(former)
            former_deci = former - former_int
            current_deci = phi_max[nn]
            if former_deci > current_deci + 0.3: former_int = former_int + 1
            lift_max[nn] = current_deci + former_int       
        p = np.poly1d(np.polyfit(t, lift_max, 1))
        phase[kk,:] = p(T)
    
    amp = np.zeros([K,NN])
    for kk in range(K):
        est_a_mode = est_a[:,kk]
        y = est_a_mode
        filtered_data = sigma_clip(y, sigma=3, maxiters=3, stdfunc=mad_std)
        outlier = filtered_data.mask
        est_a_mode = est_a_mode[~outlier]
       # plt.plot(est_t,est_p_mode,'*')
        a_max = np.zeros(n)
        for nn in np.arange(n):
            ind = np.where( est_t == t[nn])
            ind0  = ind[np.argmin(acc[ind])]
            ind0 = ind0[0]  
            a_max[nn] = est_a_mode[ind0]
       # plt.plot(phi_max[800:1000],'*') 
        lift_max = a_max      
        p = np.poly1d(np.polyfit(t, lift_max, 0))
        amp[kk,:] = p(T)
    return phase,omega,p,amp
            