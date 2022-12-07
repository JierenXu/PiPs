#!/usr/bin/env python
# coding: utf-8
import logging
import numpy as np
import tensorflow as tf
import os
import scipy.io
import matplotlib.pyplot as plt
from NOPsolver import NOP_Update
from config import get_config
from init import get_init
from utils import del_all_flags

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.1) 

del_all_flags(tf.flags.FLAGS)
#FLAGS = tf.app.flags.FLAGS
flags = tf.compat.v1.flags
flags.DEFINE_string('problem_name', 'Trig_ConstantFreq',
                           """The name of targeted signal.""")
flags.DEFINE_integer('num_run', 1,
                            """The number of experiments to repeatedly run for the same problem.""")


def main():
    problem_name = flags.FLAGS['problem_name'].value

    ## Mimic a signal of two modes with very similar frequency F1 and F2
    init_Freq = 40
    F1 = 39.8  # frequency of first mode --------- to be retrieved
    F2 = 39.99    # frequency of second mode --------- to be retrieved

    # generate signal
    short = 1
    N = 2**10 //short
    K = 2
    time = np.arange(0,1./short,1./(N*short)) # try downsampling

    true_ins_Freq = np.zeros([2,N])
    true_ins_Freq[0,:] = np.ones(N) * F1
    true_ins_Freq[1,:] = np.ones(N) * F2

    true_ins_Phase = np.zeros([2,N])
    true_ins_Phase[0,:] = time * F1
    true_ins_Phase[1,:] = time * F2

    true_Mode = np.zeros([2,N])
    true_Mode[0,:] = np.cos(2*np.pi*true_ins_Phase[0,:])
    true_Mode[1,:] = np.sin(2*np.pi*true_ins_Phase[1,:])

    true_f = np.sum(true_Mode,axis = 0)

    # try adding noise
    var = 0.1
    noise = np.random.randn(N)*var
    signal = true_f + var*noise
    #plt.plot(true_f)
    #only two periods are used to estimate F1 and F2


    # initialize algorithm
    Sig = {}
    Sig['signal'] = signal # input signal to be estimated
    Sig['time'] = time
    Sig['K'] = K # input number of modes
    Sig['D'] = 1 # input order of phase function

    logging.basicConfig(level=logging.INFO,format='%(levelname)-6s %(message)s')
    config = get_config(problem_name,Sig)

    #init_Freq = 40
    init = get_init(problem_name,Sig,config,init_Freq)
    trainer = NOP_Update(Sig,config, init)

    # update variables
    for idx_run in range(1, FLAGS['num_run'].value +1):
        logging.info('Begin to solve %s with run %d' % (problem_name, idx_run))
        for i in range(1):
            trainer.update_phase()
            trainer.update_shape()

    pattern = trainer.u
    component = trainer.modes
    phase = trainer.phase
    amp = trainer.amp
    plt.plot(np.squeeze(np.asarray(pattern[1,:])),'-*')

if __name__ == '__main__':
    main()
