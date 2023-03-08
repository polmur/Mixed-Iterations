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
rewards = [[-1, -1],[-1, -1],[-1, -1],[-1, -1], [-1, 10]]
# Transition probabilities per state-action pair
probs=np.array([
      [[1,0,0,0,0],[0.4,0.6,0,0,0]],
      [[1,0,0,0,0],[0.05,0.6,0.35,0,0]],
      [[0,1,0,0,0],[0,0.05,0.6,0.35,0]],
      [[0,0,1,0,0],[0,0,0.05,0.6,0.35]],
      [[0,0,0,1,0],[0,0,0,0.4,0.6]]]
      )

# Set value iteration parameters
max_iter = 150  # Maximum number of iterations
gamma = 0.9  # discount factor
Q_vi=value_iteration(max_iter, states, actions, probs, rewards,gamma)

#Set Q-Learning parameters
episodes=50
error_q,Q_q=q_learning(Q_vi,max_iter,episodes, states, actions, probs, rewards,gamma)

#Q-Learning combined with Value Iteration parameters

prior_percentage=[1,0.8,0.6,0.4,0.2,0]
probs_rnd=np.random.rand(5,2,5)
probs_rnd=probs_rnd/probs_rnd.sum(axis=2)[:,:,None]
Q_star=value_iteration(max_iter, states, actions, probs_rnd, rewards,gamma)
pi_star=np.argmax(Q_star, axis=1)
error_vi_l=np.zeros((episodes,max_iter,len(prior_percentage)))
error_pi=np.zeros((episodes,max_iter,len(prior_percentage)))

for t in range(len(prior_percentage)):
    prior= prior_percentage[t]           
    for d in range(episodes):
        probs_prior=randomize_probs(probs_rnd, prior)         
        Q = np.zeros((5,2))
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
                            Q_a += probs_prior[s][a][s_next]*(rewards[s][a]+gamma*Q[s_next][a])            
                            Q_max[a] = Q_a                    
                    s_play=next_state(s,a,probs_rnd)
                    #Update Q functions and compute error
                    Q_l_hat[s][a] += alpha*(rewards[s][a] + gamma*Q[s_play][a]-Q[s][a])                                            
                
                Q_vi_hat[s]=Q_max                    
                Q_hat[s]=Q_vi_hat[s]+p_unknown*Q_l_hat[s]
            Q=Q_hat
            pi=np.argmax(Q, axis=1)
            # Update Q functions and compute error    
            error_vi_l[d][i][t]=linalg.norm(Q_star-Q)
            error_pi[d][i][t]=np.sum(pi == pi_star)

error_pi_percentage=np.mean(error_pi, axis=0)
error_pi_percentage=np.absolute(error_pi_percentage-np.max(error_pi_percentage))
error_pi_00=error_pi_percentage[0:21,5]
error_pi_02=error_pi_percentage[0:21,4]
error_pi_04=error_pi_percentage[0:21,3]
error_pi_06=error_pi_percentage[0:21,2]
error_pi_08=error_pi_percentage[0:21,1]
error_pi_10=error_pi_percentage[0:21,0]
xaxis0 = [(i+1)for i in range(21)]
xaxis0[:] = [xaxis0 - 1 for xaxis0 in xaxis0]

fig0 = plt.figure(0, figsize = (6, 4))
ax0 = fig0.add_subplot(1, 1, 1)
ax0.plot(xaxis0, error_pi_00, color = 'red',linewidth=1.5, label='QL')
ax0.plot(xaxis0, error_pi_02, color = 'green',linewidth=1.5, label='MI 20%')
ax0.plot(xaxis0, error_pi_04, color = 'black',linewidth=1.5, label='MI 40%')
ax0.plot(xaxis0, error_pi_06, color = 'orange',linewidth=1.5, label='MI 60%')
ax0.plot(xaxis0, error_pi_08, color = 'purple',linewidth=1.5, label='MI 80%')
ax0.plot(xaxis0, error_pi_10, color = 'blue', linewidth=1.5,label='VI')
ax0.legend(loc='upper right',prop={'size': 10})
ax0.set_xlabel('Iterations',fontsize=14)
ax0.set_xlim([0,20])
ax0.set_ylim([0,3.5])
ax0.set_ylabel(r'$\Vert \delta\:(\pi_t - \pi^*)\Vert_{\infty}$',fontsize=14)

fig0.savefig('./policy_evolution.png', format='png', dpi=800)