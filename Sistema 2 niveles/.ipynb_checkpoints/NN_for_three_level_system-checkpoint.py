#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 11 17:10:49 2020
@author: marios mattheakis

In this code a Hamiltonian Neural Network is designed and employed
to solve a system of four differential equations obtained by Hamilton's
equations for the the Hamiltonian of Henon-Heiles chaotic dynamical.
"""


import numpy as np
import torch
import torch.optim as optim
from torch.autograd import grad
import matplotlib.pyplot as plt
import time
import copy
from os import path
import sys


# Check to see if gpu is available. If it is, use it else use the cpu

# if torch.cuda.is_available():

#     # device = torch.device('cuda')

#     device = torch.device('cuda:0')

#     print('Using ', device, ': ', torch.cuda.get_device_name()) 

#     # torch.set_default_tensor_type(torch.cuda.FloatTensor) 

#     torch.set_default_tensor_type(torch.cuda.DoubleTensor) 

# else:

#     device = torch.device('cpu')

#     torch.set_default_tensor_type('torch.DoubleTensor')

#     print('No GPU found, us': ', torch.cuda.get_device_name()) 

#     # torch.set_default_tensor_type(torch.cuda.FloatTensor) 

#     torch.set_default_tensor_type(torch.cuda.DoubleTensor) 

# else:

#     device = torch.device('cpu')

#     torch.set_default_tensor_type('torch.DoubleTensor')

#     print('No GPU found, using cpu')
    
# Animation 
plotRealTime = True

dtype=torch.float
#%%ing cpu')
    
# Animation 
plotRealTime = True

dtype=torch.float
#%%

# %matplotlib inline
plt. close('all')


# Define the sin() activation function
class mySin(torch.nn.Module):
    @staticmethod
    def forward(input):
        return torch.sin(input)
   
   #####################################
# Hamiltonian Neural Network (HNN) class
####################################
#%%

# Calculate the derivatice with auto-differention
def dfx(x,f):
    return grad([f], [x], grad_outputs=torch.ones(x.shape, dtype=dtype), create_graph=True)[0]

def perturbPoints(grid,t0,tf,sig=0.5):
# #   stochastic perturbation of the evaluation points
# #   force t[0]=t0  & force points to be in the t-interval
    delta_t = grid[1] - grid[0]  
    noise = delta_t * torch.randn_like(grid)*sig
    t = grid + noise
    t.data[2] = torch.ones(1,1)*(-1)
    t.data[t<t0]=t0 - t.data[t<t0]
    t.data[t>tf]=2*tf - t.data[t>tf]
    t.data[0] = torch.ones(1,1)*t0
    t.requires_grad = False
    return t
#%%
    
def parametricSolutions(t, nn, X0):
    
# parametric solution intial
    t0, z10, z20, z30, z40, z50, z60, z70, z80, z90, op0, os0 = X0[0],X0[1],X0[2],X0[3],X0[4],X0[5],X0[6],X0[7],X0[8],X0[9],X0[10],X0[11]
    N1, N2, N3, N4, N5, N6, N7, N8, N9, N10, N11 = nn(t)
    dt = t-t0
#### THERE ARE TWO PARAMETRIC SOLUTIONS. Uncomment f=dt 
    f = (1-torch.exp(-dt))
    Z1 = z10 + f*N1
    Z2 = z20 + f*N2
    Z3 = z30 + f*N3
    Z4 = z40 + f*N4
    Z5 = z50 + f*N5
    Z6 = z60 + f*N6
    Z7 = z70 + f*N7
    Z8 = z80 + f*N8
    Z9 = z90 + f*N9
    OP = op0 + f*N10
    OS = os0 + f*N11
    
    
    return Z1, Z2, Z3, Z4, Z5, Z6, Z7, Z8, Z9, OP,OS
#%%

def hamEqs_Loss(t,z1,z2,z3,z4,z5,z6,z7,z8,z9,op,os,X0):

    # Equations
    f1 = op*z7 + dfx(t, z1)
    f2 = os*z9 + dfx(t, z2)
    f3 = -op*z7 - os*z9 + dfx(t, z3)
    f4 = -delta2*z4 + g1*z5 + g2*z5 + op*z8/2 - os*z6/2 + dfx(t, z5)
    f5 = delta2*z5 + g1*z4 + g2*z4 + op*z9/2 + os*z7/2 + dfx(t, z4)
    f6 = -delta1*z6 + g1*z7 + g3*z7 - op*z1/2 + op*z3/2 - os*z4/2 + dfx(t, z7)
    f7 = delta1*z7 + g1*z6 + g3*z6 + os*z5/2 + dfx(t, z6)
    f8 = -delta1*z8 + delta2*z8 + g2*z9 + g3*z9 - op*z4/2 - os*z2/2 + os*z3/2 + dfx(t, z9)
    f9 = delta1*z9 - delta2*z9 + g2*z8 + g3*z8 - op*z5/2 + dfx(t, z8)
    
    
    # Model loss terms
    L1 = (f1.pow(2)).mean();  
    L2 = (f2.pow(2)).mean(); 
    L3 = (f3.pow(2)).mean(); 
    L4 = (f4.pow(2)).mean(); 
    L5 = (f5.pow(2)).mean();  
    L6 = (f6.pow(2)).mean(); 
    L7 = (f7.pow(2)).mean(); 
    L8 = (f8.pow(2)).mean(); 
    L9 = (f9.pow(2)).mean(); 
    Lmodel = L1+L2+L3+L4+L5+L6+L7+L8+L9; 
    
    # Constraint for control
    f10 = z1
    f11 = z2 - 1    # STATE TRANSFER CONDITION
    f12 = z3
    L10 = (f10.pow(2)).mean();  
    L11 = (f11.pow(2)).mean(); 
    L12 = (f12.pow(2)).mean(); 
           
    L = Lmodel + 1e-1*L10 + 1e-1*L11 + 1e-1*L12   

    return L



# NETWORK ARCHITECTURE

# A three hidden layer NN, 1 input & seven outputs
class odeNet_2level(torch.nn.Module):
    def __init__(self, D_hid=11):
        super(odeNet_2level,self).__init__()

        # Define the Activation
        self.actF = mySin() 
        
        # define layers
        self.Lin_1   = torch.nn.Linear(1, D_hid)
        self.Lin_2   = torch.nn.Linear(D_hid, D_hid)
        self.Lin_3   = torch.nn.Linear(D_hid, D_hid)
        self.Lin_4   = torch.nn.Linear(D_hid, D_hid)
        self.Lin_5   = torch.nn.Linear(D_hid, D_hid)
        self.Lin_6   = torch.nn.Linear(D_hid, D_hid)
        self.Lin_out = torch.nn.Linear(D_hid, 11)

    def forward(self,t):
        # layer 1
        l = self.Lin_1(t);    h = self.actF(l)
        # layer 2
        l = self.Lin_2(h);    h = self.actF(l)
        # layer 3
        l = self.Lin_3(h);    h = self.actF(l)
        # layer 4
        l = self.Lin_4(h);    h = self.actF(l)        
        # layer 5
        l = self.Lin_5(h);    h = self.actF(l)
        # layer 6
        l = self.Lin_6(h);    h = self.actF(l)

        
        # output layer
        r = self.Lin_out(h)
        
        # Density matrix 
        z1n = (r[:,0]).reshape(-1,1); 
        z2n = (r[:,1]).reshape(-1,1); 
        z3n = (r[:,2]).reshape(-1,1); 
        z4n = (r[:,3]).reshape(-1,1); 
        z5n = (r[:,4]).reshape(-1,1); 
        z6n = (r[:,5]).reshape(-1,1); 
        z7n = (r[:,6]).reshape(-1,1); 
        z8n = (r[:,7]).reshape(-1,1); 
        z9n = (r[:,8]).reshape(-1,1); 
        z10n = (r[:,9]).reshape(-1,1); 
        z11n = (r[:,10]).reshape(-1,1); 
        
        
        return z1n, z2n, z3n, z4n, z5n, z6n, z7n, z8n, z9n, z10n, z11n
#%%
# Train the NN
def run_odeNet_HH_MM(X0, tf, neurons, epochs, n_train,lr, PATH= "models/model_3level", loadWeights=False,
                    minibatch_number = 1, minLoss=1e-3):
                    
    fc0 = odeNet_2level(neurons)
    fc1 =  copy.deepcopy(fc0) # fc1 is a deepcopy of the network with the lowest training loss
    # optimizer
    betas = [0.999, 0.9999]
    
    optimizer = optim.Adam(fc0.parameters(), lr=lr, betas=betas)
    Loss_history = [];     Llim =  1 
    
    # Initial time
    t0=X0[0]; 
    grid = torch.linspace(t0, tf, n_train).reshape(-1,1)
 
    
## LOADING WEIGHTS PART if PATH file exists and loadWeights=True
    if path.exists(PATH) and loadWeights==True:
        checkpoint = torch.load(PATH)
        fc0.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        tt = checkpoint['epoch']
        Ltot = checkpoint['loss']
        fc0.train(); # or model.eval
    
    
## TRAINING ITERATION    
    TeP0 = time.time()
    for tt in range(epochs):                
# Perturbing the evaluation points & forcing t[0]=t0
        # t=perturbPoints(grid,t0,tf,sig=.03*tf)
        t=perturbPoints(grid,t0,tf,sig= 0.3*tf)
            
# BATCHING
        batch_size = int(n_train/minibatch_number)
        batch_start, batch_end = 0, batch_size

        idx = np.random.permutation(n_train)
        t_b = t[idx]
        t_b = t
        t_b.requires_grad = True

        loss=0.0
        for nbatch in range(minibatch_number): 
# batch time set
            t_mb = t_b[batch_start:batch_end]
#  Network solutions 
            z1,z2,z3,z4,z5,z6,z7,z8,z9,op,os = parametricSolutions(t_mb,fc0,X0)
# LOSS
#  Loss function defined by Hamilton Eqs. (symplectic): Writing explicitely the Eqs (faster)
            Ltot = hamEqs_Loss(t_mb,z1,z2,z3,z4,z5,z6,z7,z8,z9,op,os,X0)
            

#  Loss function defined by Hamilton Eqs. (symplectic): Calculating with auto-diff the Eqs (slower)
#             Ltot = hamEqs_Loss_byH(t_mb,x,y,px,py,lam)
    # Regularization
            l2_lambda = 0.001
            l2_reg = torch.tensor(0.)
            for param in fc0.parameters():
                l2_reg += torch.norm(param)
            Ltot += l2_lambda * l2_reg
            

# OPTIMIZER
            Ltot.backward(retain_graph=False); #True
            optimizer.step(); 
            loss += Ltot.data.numpy()
       
            optimizer.zero_grad()

            batch_start +=batch_size
            batch_end +=batch_size

# keep the loss function history
        Loss_history.append(loss)       

#Keep the best model (lowest loss) by using a deep copy
        if  tt > 0.8*epochs  and Ltot < Llim:
            fc1 =  copy.deepcopy(fc0)
            Llim=Ltot 

# break the training after a thresold of accuracy
        if Ltot < minLoss :
            fc1 =  copy.deepcopy(fc0)
            print('Reach minimum requested loss')
            break



    TePf = time.time()
    runTime = TePf - TeP0     
    
    
    torch.save({
    'epoch': tt,
    'model_state_dict': fc1.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'loss': Ltot,
    }, PATH)

    return fc1, Loss_history, runTime

###
#%%
def trainModel(X0, t_max, neurons, epochs, n_train, lr,  loadWeights=False, minLoss=1e-8, showLoss=True, PATH ='models/'):
    model,loss,runTime = run_odeNet_HH_MM(X0, t_max, neurons, epochs, n_train,lr,  loadWeights=loadWeights, minLoss=minLoss, minibatch_number=1)

    np.savetxt('data/loss.txt',loss)
    
    if showLoss==True :
        print('Training time (minutes):', runTime/60)
        print('Training Loss: ',  loss[-1] )
       
        plt.figure(figsize=(10,6), tight_layout=True)
        plt.loglog(loss,'-b',alpha=0.975); 
        plt.tight_layout()
        plt.ylabel('Loss');plt.xlabel('t')
        axes = plt.gca()
        axes.xaxis.label.set_size(22)
        axes.yaxis.label.set_size(22)
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        plt.legend(prop={"size":22})
        plt.grid()

        # np.savetxt('loss_sim4.dat', loss)
        
        # plt.savefig('HHsystem/HenonHeiles_loss.png')
        plt.savefig('3level_loss.png')
#%%    

def loadModel(PATH="models/model_3level"):
    if path.exists(PATH):
        fc0 = odeNet_2level(neurons)
        checkpoint = torch.load(PATH)
        fc0.load_state_dict(checkpoint['model_state_dict'])
        fc0.train(); # or model.eval
    else:
        print('Warning: There is not any trained model. Terminate')
        sys.exit()

    return fc0    



##########################################3
##########################################3
##########################################3
#%%
# TRAIN THE NETWORK. 
# Set the initial state. lam controls the nonlinearity
# Set parameters
delta1, delta2, g1, g2, g3 = 0, 0, 0.001, 0.001, 0.004
# Set the initial state. lam controls the nonlinearity
z10, z20, z30, z40, z50, z60, z70, z80, z90,op0,os0 = 1,0,0,0,0,0,0,0,0,0,0   
 

# Set the time range and the training points N
t0, tmax, N = 0, 6, 300;     
X0 = [t0, z10, z20, z30, z40, z50, z60, z70, z80, z90, op0,os0, tmax, delta1, delta2, g1, g2, g3] 
#%%
# Here, we use one mini-batch. NO significant different in using more
n_train, neurons, epochs, lr = N, 10, int(1e4), 1e-3
trainModel(X0, tmax, neurons, epochs, n_train, lr,  loadWeights=False, minLoss=1e-6, showLoss=True)
model = loadModel()

#%%
#####################################
# TEST THE PREDICTED SOLUTIONS
#######################################

nTest = N ; t_max_test = 1.0*tmax
tTest = torch.linspace(t0,t_max_test,nTest)

tTest = tTest.reshape(-1,1);
tTest.requires_grad=True
t_net = tTest.detach().numpy()

z1,z2,z3,z4,z5,z6,z7,z8,z9,op,os= parametricSolutions(tTest,model,X0)
z1=z1.data.numpy(); 
z2=z2.data.numpy()
z3=z3.data.numpy(); 
z4=z4.data.numpy(); 
z5=z5.data.numpy(); 
z6=z6.data.numpy(); 
z7=z7.data.numpy()
z8=z8.data.numpy(); 
z9=z9.data.numpy(); 
op=op.data.numpy(); 
os=os.data.numpy()


#%%
################
# Make the plots
#################


# Plot vector components
lineW = 2 # Line thickness
plt.figure(figsize=(10,6), tight_layout=True)
plt.plot(t_net, z1,'-b', linewidth=3, label=r'$\rho_{11}$'); 
plt.plot(t_net, z2,'-r', linewidth=3, label=r'$\rho_{22}$'); 
plt.plot(t_net, z3,'-g', linewidth=3, label=r'$\rho_{33}$'); 
axes = plt.gca()
plt.xlabel('t')
axes.xaxis.label.set_size(22)
axes.yaxis.label.set_size(22)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.legend(prop={"size":22})
plt.grid()



# Plot control functions
lineW = 2 # Line thickness
plt.figure(figsize=(10,6), tight_layout=True)
plt.plot(t_net, op,'-b', linewidth=3, label=r'$\Omega_{p}(t)$'); 
plt.plot(t_net, os,'r--', linewidth=3, label=r'$\Omega_{s}(t)$'); 
axes = plt.gca()
plt.xlabel('t')
axes.xaxis.label.set_size(22)
axes.yaxis.label.set_size(22)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.legend(prop={"size":22})
plt.grid()



np.savetxt('Omega_p.dat', op)
np.savetxt('Omega_s.dat', os)

np.savetxt('times.dat', t_net)

#%%
# 14 is the new test using the correct loss function