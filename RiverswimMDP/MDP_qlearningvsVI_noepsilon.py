# -*- coding: utf-8 -*-
"""
@author: polmu
"""

import math
import os
import random
import numpy as np
import matplotlib
from numpy import linalg
import matplotlib.pyplot as plt
from numpy import nan
import statistics
import mdptoolbox.example
from functions_ import *

np.random.seed(50)

# Initialize Markov Decision Process model
actions = (0, 1)  # actions (0=left, 1=right)
states = (0, 1, 2, 3, 4)  # states
rewards = [[-0.1, 0.1],[-0.1, 0.1],[-0.1, 0.1],[-0.1, 0.1], [-0.1, 3]]
# Transition probabilities per state-action pair
probs=np.array([
      [[1,0,0,0,0],[0.4,0.6,0,0,0]],
      [[1,0,0,0,0],[0.05,0.6,0.35,0,0]],
      [[0,1,0,0,0],[0,0.05,0.6,0.35,0]],
      [[0,0,1,0,0],[0,0,0.05,0.6,0.35]],
      [[0,0,0,1,0],[0,0,0,0.4,0.6]]]
      )

# Set value iteration parameters
max_iter = 200  # Maximum number of iterations
gamma = 0.8  # discount factor
Q_vi=value_iteration(max_iter, states, actions, probs, rewards,gamma)

#Set Q-Learning parameters
episodes=10
#error_q,Q_q=q_learning(Q_vi,max_iter,episodes, states, actions, probs, rewards,gamma)

#Q-Learning combined with Value Iteration parameters

prior_percentage=[1,0.8,0.6,0.4,0.2,0]
probs_rnd=np.random.rand(5,2,5)
probs_rnd=probs_rnd/probs_rnd.sum(axis=2)[:,:,None]
Q_star=value_iteration(max_iter, states, actions, probs_rnd, rewards,gamma)
error_vi_l=np.zeros((episodes,max_iter,len(prior_percentage)))

for t in range(len(prior_percentage)):
    prior= prior_percentage[t]           
    for d in range(episodes):
        probs_prior=randomize_probs(probs_rnd, prior)         
        Q = np.random.random(size=(5,2))
        Q_vi_hat = np.zeros((5,2))
        Q_l_hat = np.zeros((5,2))
        Q_hat = np.zeros((5,2))        
        for i in range(max_iter):
            alpha=1/((i+1)**0.85)
            for s in states:
                Q_max = [0,0]
                for a in actions:
                    Q_a=0            
                    p_unknown=(1-np.nansum(probs_prior[s][a]))
                    for s_next in states:
                        if probs_prior[s][a][s_next]==probs_prior[s][a][s_next]:                        
                            Q_a += probs_prior[s][a][s_next]*(rewards[s][a]+gamma*np.max(Q[s_next]))            
                            Q_max[a] = Q_a                    
                    s_play=next_state(s,a,probs_rnd)
                    #Update Q functions and compute error
                    Q_l_hat[s][a] += alpha*(rewards[s][a] + gamma*np.max(Q[s_play])-Q[s][a])                                           
                
                Q_vi_hat[s]=Q_max                    
                Q_hat[s]=Q_vi_hat[s]+p_unknown*Q_l_hat[s]
            Q=Q_hat
            # Update Q functions and compute error    
            error_vi_l[d][i][t]=linalg.norm(Q_star-Q)

error_00=error_vi_l[:,:,5]
error_02=error_vi_l[:,:,4]
error_04=error_vi_l[:,:,3]
error_06=error_vi_l[:,:,2]
error_08=error_vi_l[:,:,1]
error_10=error_vi_l[:,:,0]
xaxis = [(i+1)for i in range(200)]
xaxis[:] = [xaxis - 1 for xaxis in xaxis]

error_50_00=np.percentile(error_00, 50,axis=0)
error_50_02=np.percentile(error_02, 50,axis=0)
error_50_04=np.percentile(error_04, 50,axis=0)
error_50_06=np.percentile(error_06, 50,axis=0)
error_50_08=np.percentile(error_08, 50,axis=0)
error_50_10=np.percentile(error_10, 50,axis=0)

fig = plt.figure(0, figsize = (6, 4))
ax = fig.add_subplot(1, 1, 1)
ax.plot(xaxis, error_50_00, color = 'red',linewidth=1.5, label='QL')
ax.plot(xaxis, error_50_02, color = 'green',linewidth=1.5, label='MI 20%')
ax.plot(xaxis, error_50_04, color = 'black',linewidth=1.5, label='MI 40%')
ax.plot(xaxis, error_50_06, color = 'orange',linewidth=1.5, label='MI 60%')
ax.plot(xaxis, error_50_08, color = 'purple',linewidth=1.5, label='MI 80%')
ax.plot(xaxis, error_50_10, color = 'blue', linewidth=1.5,label='VI')

ax.set_yscale('linear')
ax.set_ylim([0,6])
ax.set_xlim([0,75])
ax.legend(loc='upper right',prop={'size': 10})
ax.set_xlabel('Iterations',fontsize=14)
ax.set_ylabel(r'$||Q^*-Q_t||_\infty$',fontsize=14)
#ax.set_title("Random MDP with 50 randomizations of P",  loc='center')

# fig1 = plt.figure(1, figsize = (6, 4))
# ax1 = fig1.add_subplot(1, 1, 1)
# ax1.plot(xaxis, minmax_00, color = 'red',linewidth=1.5, label='Q-Learning')
# ax1.plot(xaxis, minmax_02, color = 'green',linewidth=1.5, label='Mixed Iterations 20%')
# ax1.plot(xaxis, minmax_04, color = 'black',linewidth=1.5, label='Mixed Iterations 40%')
# ax1.plot(xaxis, minmax_06, color = 'orange',linewidth=1.5, label='Mixed Iterations 60%')
# ax1.plot(xaxis, minmax_08, color = 'purple',linewidth=1.5, label='Mixed Iterations 80%')
# ax1.plot(xaxis, error_00_2575, color = 'red',linestyle='dashed',linewidth=1.5, label='Q-Learning')
# ax1.plot(xaxis, error_02_2575, color = 'green',linestyle='dashed',linewidth=1.5, label='Mixed Iterations 20%')
# ax1.plot(xaxis, error_04_2575, color = 'black',linestyle='dashed',linewidth=1.5, label='Mixed Iterations 40%')
# ax1.plot(xaxis, error_06_2575, color = 'orange',linestyle='dashed',linewidth=1.5, label='Mixed Iterations 60%')
# ax1.plot(xaxis, error_08_2575, color = 'purple',linestyle='dashed',linewidth=1.5, label='Mixed Iterations 80%')

# ax1.set_yscale('linear')
# ax1.set_xlim([0,150])
# ax1.legend(loc='upper right',prop={'size': 6})
# ax1.set_xlabel('Iterations')
# ax1.set_ylabel('Error between runs')
# ax1.set_title("Variance between 50 episodes",  loc='center')



fig.savefig('./randomPrandom_linear.png', format='png', dpi=800)
#fig1.savefig('./error_runs.png', format='png', dpi=1200)
