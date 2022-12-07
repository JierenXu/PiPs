import logging
import time
import numpy as np
import numpy.matlib
import tensorflow as tf
import os
import scipy.io as sio
import matplotlib.pyplot as plt
#import matlab.engine
from utils import rbf_kernel,inv_rbf_kernel,phi0_2_phi,compute_const_inst_info, coef2poly,gen_mode

class NOP_Update():
    def __init__(self, Sig,config,init):
        # compute the fixed matrix
        self.signal = Sig['signal']
        self.time = Sig['time']
        self.U = init.U
        self.Z = init.Z 
        self.z = init.z
        self.u = init.u
        # main parameters
        self.N = config.N
        self.K = config.K
        self.T = config.T
        # optimization parameters
        self.sparams = config.GPparams
        self.rparams = config.RDBRparams
        self.opt = config.opt
        # initialize patterns as sin
        self.b = config.b
        self.l = config.l
        self.m = config.m
        self.n = init.n            
        self.Step = self.n/2     # stride of between two consequetive chips
        self.LS = (self.N - self.n)/self.Step + 2 # total number of chips
        logging.info('Signal has %d samples, divided into %d chips each with %d samples'%(self.N,self.LS,self.n))
        if self.opt['Onechip'] == 1:  
            self.LS = 1
            self.N = self.n
            self.signal = self.signal[0:self.n]
            self.time = self.time[0:self.n]
            logging.info('But we only compute the first chip')
        # initialize phase, amp and modes
        self.D = config.D
        self.Dc = config.Dc  
        self.B0 = init.B0                     
        self.C0 = init.C0  
        self.phase = coef2poly(self.time,self.B0)
        self.amp = coef2poly(self.time,self.C0)
        self.modes = gen_mode(self.amp,self.phase,self.z,self.u)
        # initialize matlab engine if update shape
        if self.opt['Update Pattern'] == True:  self.initialize_shape()  
        logging.info('Update Pattern:    %s'%(self.opt['Update Pattern']))
        logging.info('Update Amplitude:  %s'%(self.opt['Update Amp']))
        
    def update_phase(self, max_epochs = 5000):
        est_phi_points,est_omega_points,est_amp_points,final_cost = self.phase_solver(max_epochs)
        self.phase, self.omega, _ ,self.amp= compute_const_inst_info(est_phi_points,est_omega_points,est_amp_points,final_cost)
        self.modes = gen_mode(self.amp,self.phase,self.z,self.u)
        
    def update_shape(self):
        if self.opt['Update Pattern'] == True:
            mat_phase = matlab.double(self.phase.tolist())
            mat_amp = matlab.double(self.amp.tolist())
            self.eng.workspace['mat_phase'] = mat_phase
            self.eng.workspace['mat_amp'] = mat_amp
            print( 'running RDBR...')
            #self.eng.edit('srcIterRegNOP',nargout=0)
            #sio.savemat('rdbr.mat', {'phase':self.phase,'signal':self.signal})
            self.eng.eval("[mat_shapes,mat_modes] = srcIterRegNOP(mat_signal,mat_N,mat_K,mat_amp,mat_phase,mat_opt,mat_Z);",nargout = 0)
            print ('RDBR done!')
            # translate back the new PIP U and new modes
            self.u = np.asmatrix(self.eng.workspace['mat_shapes'])
            self.U = np.matlib.repmat(self.u, 1, self.b)
            self.modes = np.asmatrix(self.eng.workspace['mat_modes'])
   
    def initialize_shape(self):
        print ('Openning Matlab engine...' )
        self.eng = matlab.engine.start_matlab()
        print ('Matlab engine openned!')
        
        self.eng.addpath(self.eng.genpath('RDBR'))
        mat_K = np.float(self.K)
        mat_N = np.float(self.N)
        mat_signal = matlab.double(self.signal.tolist())
        mat_z = matlab.double(self.z.tolist())
        mat_opt = self.eng.struct(self.rparams)
        self.eng.workspace['mat_signal'] = mat_signal
        self.eng.workspace['mat_Z'] = mat_z
        self.eng.workspace['mat_N'] = mat_N
        self.eng.workspace['mat_K'] = mat_K 
        self.eng.workspace['mat_opt'] = mat_opt
        
    
    def phase_solver(self,max_epochs):
        ########################### define parameters ###########################
        iK = np.float32(inv_rbf_kernel(self.Z,self.sparams))
        K = self.K
        D = self.D
        Dc = self.Dc
        m = self.m
        n = self.n
        t0 = self.time[np.arange(n)]
        pts = 0
        ########################## define graph #######################################
        #def define_optimizer():
        # initialize parameters
        tf.reset_default_graph() 
        _z = tf.placeholder(dtype = 'float32', shape = [m], name = 'z')
        _u = tf.placeholder(dtype = 'float32', shape = [K,m], name = 'u')
        _y = tf.placeholder(dtype = 'float32', shape = [n], name = 'y')
        
        # generate phase from B
        _B = tf.Variable(initial_value = np.float32(np.ones([K,D+1])), name = 'phi0', dtype =tf.float32)
        _phi =  tf.tile(tf.expand_dims(_B[:,0],axis=1), [1,n])  
        for ord in np.arange(D)+1:
            _phi = _phi + tf.tile(tf.expand_dims(_B[:,ord],axis=1), [1,n]) * t0 **ord 
        
        # generate amplitude from C
        _C = tf.Variable(initial_value = np.float32(np.ones([K,Dc+1])), name = 'phi0', dtype =tf.float32,trainable= self.opt['Update Amp'])
        _a =  tf.tile(tf.expand_dims(_C[:,0],axis=1), [1,n])  
        for ord in np.arange(Dc)+1:
            _a = _a + tf.tile(tf.expand_dims(_C[:,ord],axis=1), [1,n]) * t0 **ord 

         
        if 1:
            # GP REGRESSION (Pips)
            _tempKnm = rbf_kernel(_phi[0],_z,self.sparams,n,m)
            _f =   _a[0] * tf.matmul(_tempKnm,tf.matmul(iK, tf.expand_dims(_u[0,:],axis=1)))            
            for k in np.arange(K-1)+1 :
                _tempKnm = rbf_kernel(_phi[k],_z,self.sparams,n,m)      
                _f = _f + _a[k] * tf.matmul(_tempKnm,tf.matmul(iK,tf.expand_dims(_u[k,:],axis=1)))
        
        else:
            # kNN REGRESSION 
            _tempKnm_ = rbf_kernel(_phi[0],_z,self.sparams,n,m)
            _tempKnm,norm_ = tf.linalg.normalize( _tempKnm_ , ord = 1, axis = 1 )  ##############
            #print( _tempKnm_.shape )
            #print( norm_.shape )
            _f =   _a[0] * tf.matmul( _tempKnm, tf.expand_dims(_u[0,:],axis=1))           
            for k in np.arange(K-1)+1 :
                _tempKnm_ = rbf_kernel( _phi[k], _z, self.sparams,n,m)      
                _tempKnm,_ = tf.linalg.normalize( _tempKnm_ , ord = 1, axis = 1 )  #############
                _f = _f + _a[k] * tf.matmul(_tempKnm,tf.expand_dims(_u[k,:],axis=1))
        
        
        
        # construct loss function
        _dmean = tf.expand_dims(_y, axis = 1) - _f
        _cost = tf.reduce_mean(tf.square(_dmean))
        lr = 0.005
        train_ops = tf.train.AdamOptimizer(lr).minimize(_cost, var_list = tf.trainable_variables())
        
        
        # start optimization using tf
        graph = tf.get_default_graph()
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.1) 
        sess =  tf.Session(graph = graph, config=tf.ConfigProto(gpu_options=gpu_options)) 
        tf.global_variables_initializer().run(session = sess)  

        
        ########################## update each chip #######################################
        ## initialize output
        phi_points = []
        omega_points = []
        amp_points = []
        final_cost = np.ones(self.LS* self.n)
        #define_optimizer()
        
        # take the signal chip
        for ii in range(self.LS):
            logging.info('Updating Chip %d of %d: ' %(ii+1,self.LS))
            if ii != self.LS -1: a = np.arange(self.n) + ii*self.Step
            else: a = np.arange(self.n) + self.N - self.n
            t = self.time[a]
            y = self.signal[a]
            ## change initial value of phi0
            assign_op = _B.assign(self.B0)
            sess.run(assign_op)
            for eidx in range(max_epochs):
                _,loss = sess.run([train_ops,_cost], feed_dict = {_z:self.Z, _u:self.U, _y:y})
                if eidx%1000==0:
                     logging.info("step: %5u,    loss: %.4e,  elapsed time %3u" % (
                        eidx, loss, 3))
            ############# encode optimized chip phase/freq ########################
            new_B = sess.run(_B)
            new_C = sess.run(_C)
            [phi_chip,omega_chip,amp_chip] = phi0_2_phi(new_B,new_C,t0)
            for pts in np.arange(n):
                phi_points.append(np.append(t[pts],phi_chip[:,pts].reshape(-1)))
                omega_points.append(np.append(t[pts],omega_chip[:,pts].reshape(-1)))
                amp_points.append(np.append(t[pts],amp_chip[:,pts].reshape(-1)))
                final_cost[pts] = loss
                pts += 1
            ## print result for  the signal chip 
            print ('\n')
            for k in np.arange(K) : 
                print ('Component   ', k+1 )
                print ('Update B   ',  str(self.B0[k]),'  ---->    ',str(new_B[k])  )
                print ('Estimated frequency = ' , str(new_B[k][1]) )
                if  self.opt['Update Amp'] :print ( 'update C   ',  str(self.C0[k]),'  ---->    ',str(new_C[k])   )
                #print '\n'
            #plt.plot(phi,y,'-^',Z,U[0,:],'-*')

            
        ########################## global estimation #######################################
        # do global estimate
        NN = len(phi_points)
        est_phi_points = np.array(phi_points).reshape([NN,K+1])
        est_omega_points = np.array(omega_points).reshape([NN,K+1])
        est_amp_points = np.array(amp_points).reshape([NN,K+1])
        return est_phi_points,est_omega_points,est_amp_points , final_cost
            
            
