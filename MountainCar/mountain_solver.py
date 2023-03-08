# -*- coding: utf-8 -*-
"""
Created on Fri Jun  3 11:59:35 2022

@author: polmu
"""

import math
import os
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from functions_mountain import *

actions=(-1,0,1)
position=np.linspace(-1.2,0.5,num=141)
velocity=np.linspace(-0.07,0.07, num=141)
gamma=0.99
m1=len(actions)
n1=len(position)
n2=len(velocity)
pi_star=np.zeros((n1,n2))
states=np.argwhere((pi_star==0))
max_iter_vi=500
pr=0.8
#pi_star, Q_star =value_iteration_mountain(n1,n2,m1, max_iter_vi,states,actions,position,velocity,pr)

position_axis=np.linspace(-1.2,0.5,num=10)
velocity_axis=np.linspace(-0.07,0.07, num=6)
position_axis=np.round(position_axis, 2)
velocity_axis=np.round(velocity_axis, 3)
fig0 = plt.figure(0, figsize = (6, 4))
ax0 = fig0.add_subplot(1, 1, 1)
ax0 = sns.heatmap(Q_star, xticklabels=20, yticklabels=20, cmap="Blues")
ax0.set_xticklabels(labels = ['-1.2','-0.9','-0.6','-0.3','0.3','0.6','0.9','1.2'])
ax0.set_yticklabels(['-0.07','-0.052','-0.035','-0.017','0.017','0.035','0.052','0.07'], rotation=0)
ax0.set_xlabel('$x_t$ (m)',fontsize=14)
ax0.set_ylabel('$v_t$ (m/s)',fontsize=14) 
plt.show()

max_iter=1000

#pi_q, Q= q_learning_mountain(n1,n2,m1, max_iter,states,actions,position,velocity,gamma,pr)

#ax1 = sns.heatmap(pi_q)
#plt.show()

# episodes=1

# prior_percentage=[1,0.75,0.5,0.25,0]
# error_pi=np.zeros((episodes,max_iter,len(prior_percentage)))
# pi=np.zeros((len(prior_percentage),n1,n2))

# for t in range(len(prior_percentage)):
#     prior= prior_percentage[t]
#     p_unknown=(1-prior)          
#     for d in range(episodes):        
#         Q=np.zeros((n1,n2,m1))
#         Q_vi_hat = np.zeros((n1,n2,m1))        
#         Q_l_hat = np.zeros((n1,n2,m1))
#         idx_p=np.array([0,0,0])
#         idx_v=np.array([0,0,0])
#         Q_hat=np.zeros((n1,n2,m1))
#         p_next=np.zeros(3)
#         v_next=np.zeros(3)             
#         for i in range(max_iter):
#             alpha=1/((math.log(i+2)))            
#             np.random.shuffle(states) 
#             for s in states:
#                 Q_max = np.zeros(m1)
#                 p=position[s[0]]
#                 v=velocity[s[1]]                
#                 idx_p_q = (np.abs(position - p)).argmin()
#                 idx_v_q = (np.abs(velocity - v)).argmin()
#                 reward=cost(p,v)
#                 for a in actions:    
#                     # Compute state value
#                     p_next[a+1],v_next[a+1]=next_state_vi(p,v,a)
#                     idx_p[a+1] = (np.abs(position - p_next[a+1])).argmin()
#                     idx_v[a+1] = (np.abs(velocity - v_next[a+1])).argmin()
#                 for a in actions: 
#                     probs=np.ones(3)*0.1
#                     probs[a+1]=0.8                    
#                     Q_max[a+1] = prior*(reward+probs[0]*0.99*np.max(Q[idx_p[0]][idx_v[0]])+probs[1]*0.99*np.max(Q[idx_p[1]][idx_v[1]])+probs[2]*0.99*np.max(Q[idx_p[2]][idx_v[2]]))  
#                     #Update Q functions and compute error
#                     p_next_q,v_next_q=next_state(p,v,a,pr)                    
#                     idx_p_n = (np.abs(position -p_next_q)).argmin()
#                     idx_v_n = (np.abs(velocity - v_next_q)).argmin()
#                     Q_l_hat[idx_p_q][idx_v_q][a+1] += p_unknown*(alpha*(reward + gamma*np.max(Q[idx_p_n][idx_v_n])-Q[idx_p_q][idx_v_q][a+1]))          
                
#                 Q_vi_hat[idx_p_q][idx_v_q]=Q_max                
#                 Q_hat[idx_p_q][idx_v_q]=Q_vi_hat[idx_p_q][idx_v_q]+Q_l_hat[idx_p_q][idx_v_q]             
#                 pi[t][s[0]][s[1]]=np.argmax(Q_hat[idx_p_q][idx_v_q])-1
#                 if Q_hat[idx_p_q][idx_v_q][0]==Q_hat[idx_p_q][idx_v_q][1] and pi[t][s[0]][s[1]]==-1:
#                     pi[t][s[0]][s[1]]=0
#                 if Q_hat[idx_p_q][idx_v_q][2]==Q_hat[idx_p_q][idx_v_q][1] and pi[t][s[0]][s[1]]==1:
#                     pi[t][s[0]][s[1]]=0
#                 if Q_hat[idx_p_q][idx_v_q][2]==Q_hat[idx_p_q][idx_v_q][1] and Q_hat[idx_p_q][idx_v_q][0]==Q_hat[idx_p_q][idx_v_q][1]:
#                     pi[t][s[0]][s[1]]=0
#             Q=Q_hat            
#             error_pi[d][i][t]=np.sum(pi[t] == pi_star)
#     ax1 = sns.heatmap(pi[t])
#     #ax1 = sns.heatmap(pi_star)
#     plt.show()


# error_pi_percentage=np.mean(error_pi, axis=0)
# error_pi_02=error_pi_percentage[0:max_iter,4]
# error_pi_04=error_pi_percentage[0:max_iter,3]
# error_pi_06=error_pi_percentage[0:max_iter,2]
# error_pi_08=error_pi_percentage[0:max_iter,1]
# error_pi_10=error_pi_percentage[0:max_iter,0]
# xaxis0 = [(i+1)for i in range(max_iter)]
# xaxis0[:] = [xaxis0 - 1 for xaxis0 in xaxis0]

# fig0 = plt.figure(0, figsize = (6, 4))
# ax0 = fig0.add_subplot(1, 1, 1)
# ax0.plot(xaxis0, error_pi_02, color = 'red',linewidth=1.5, label='Q-Learning')
# ax0.plot(xaxis0, error_pi_04, color = 'black',linewidth=1.5, label='Mixed Iterations 50%')
# ax0.plot(xaxis0, error_pi_06, color = 'orange',linewidth=1.5, label='Mixed Iterations 60%')
# ax0.plot(xaxis0, error_pi_08, color = 'purple',linewidth=1.5, label='Mixed Iterations 80%')
# ax0.plot(xaxis0, error_pi_10, color = 'blue', linewidth=1.5,label='Value Iteration')
# ax0.legend(loc='lower right',prop={'size': 6})
# ax0.set_xlabel('Iterations')
# ax0.set_xlim([0,max_iter])
# ax0.set_ylabel(r'$\pi^*\equiv\pi_t$')