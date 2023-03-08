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
from numpy import linalg
from functions_maze import *
from mazelab.generators import random_maze
from scipy.spatial import distance

np.random.seed(0)
maze_x=9
maze_y=9
maze = random_maze(width=maze_x, height=maze_y, complexity=.75, density=.75)
maze_del = np.delete(maze, np.all(maze[..., :] == 1, axis=0), axis=1) # Deletes all 0-value columns
maze_del = np.delete(maze_del, np.all(maze[..., :] == 1, axis=1), axis=0) # Deletes all 0-value rows

# 0 - 'N', 1 -'S', 2 - 'W', 3 -'E', 4 - stay
actions=(0,1,2,3)
states=np.argwhere((maze_del==0))
states=states+1
walls=np.argwhere(maze_del==1)
walls=walls+1
max_iter=1000
gamma=0.99 #0.99 for 7x7 and i+1**0.5
p_max=0.8

probabilities=generate_probabilities(actions,states,p_max,maze, maze_y,maze_x)
rewards=gen_rewards(maze,maze_x,maze_y,len(actions),states)
pi_star, Q_star, J= value_iteration(max_iter,states,actions, maze,probabilities, rewards,gamma, maze_x-1, maze_y-1, len(actions))
max_iter=600

episodes=1
#pi,Q_learning,error_Q,error_pi_q= q_learning(max_iter, episodes, states,actions,p_max, maze,probabilities, rewards,gamma,Q_star,pi_star, maze_x-1, maze_y-1, len(actions))
#pi=clean_pi(walls,pi,maze)

prior_percentage=[1,0.8,0.6,0.4,0.2,0]
error_vi_l=np.zeros((episodes,max_iter,len(prior_percentage)))
error_pi=np.zeros((episodes,max_iter,len(prior_percentage)))

for t in range(len(prior_percentage)):
    prior= prior_percentage[t]           
    for d in range(episodes):
        probs_prior=randomize_probs(probabilities, prior)         
        Q = np.zeros((maze_x,maze_y,len(actions)))
        Q_vi_hat = np.zeros((maze_x,maze_y,len(actions)))
        #J = np.zeros((maze_x,maze_y))
        pi=np.zeros((maze_x,maze_y))
        Q_l_hat = np.zeros((maze_x,maze_y,len(actions)))
        Q_hat = np.zeros((maze_x,maze_y,len(actions)))        
        for i in range(max_iter):
            alpha=1/((i+1)**0.5)
            #if prior != 1:   # check if this works tomorrow
            np.random.shuffle(states) #check the effect of this 
            for s in states:
                Q_max = [0,0,0,0]
                s1=s[0]
                s2=s[1]
                for a in actions:
                    Q_a=0                  
                    # Compute state value
                    s_next=([s1-1,s2],[s1+1,s2],[s1,s2-1],[s1,s2+1],[s1,s2])
                    p_unknown=1-np.nansum([probs_prior[a][s1][s2][s1-1][s2],probs_prior[a][s1][s2][s1+1][s2],probs_prior[a][s1][s2][s1][s2-1],probs_prior[a][s1][s2][s1][s2+1],probs_prior[a][s1][s2][s1][s2]])
                    for sx in s_next :
                        if probs_prior[a][s1][s2][sx[0]][sx[1]]==probs_prior[a][s1][s2][sx[0]][sx[1]]:
                            Q_a += probabilities[a][s1][s2][sx[0]][sx[1]]*(rewards[a][s1][s2]+gamma * np.max(Q[sx[0]][sx[1]]))   
                    Q_max[a] = Q_a
                    s_y,s_x=next_state(s,a,p_max,maze)  
                    #Update Q functions and compute error
                    Q_l_hat[s1][s2][a] += p_unknown*(alpha*(rewards[a][s1][s2] + gamma*np.max(Q[s_y][s_x])-Q[s1][s2][a]))          
                Q_vi_hat[s1][s2]=Q_max                
                Q_hat[s1][s2]=Q_vi_hat[s1][s2]+Q_l_hat[s1][s2]
             
            Q=Q_hat
            #J=np.max(Q, axis=2)
            pi=np.argmax(Q, axis=2)            
            # Update Q functions and compute error    
            error_vi_l[d][i][t]=linalg.norm(Q_star-Q)
            error_pi[d][i][t]=np.sum(pi == pi_star)

error_pi_percentage=np.mean(error_pi, axis=0)
error_pi_00=error_pi_percentage[0:max_iter,5]
error_pi_02=error_pi_percentage[0:max_iter,4]
error_pi_04=error_pi_percentage[0:max_iter,3]
error_pi_06=error_pi_percentage[0:max_iter,2]
error_pi_08=error_pi_percentage[0:max_iter,1]
error_pi_10=error_pi_percentage[0:max_iter,0]
xaxis0 = [(i+1)for i in range(max_iter)]
xaxis0[:] = [xaxis0 - 1 for xaxis0 in xaxis0]

fig0 = plt.figure(0, figsize = (6, 4))
ax0 = fig0.add_subplot(1, 1, 1)
ax0.plot(xaxis0, error_pi_00, color = 'red',linewidth=1.5, label='Q-Learning')
ax0.plot(xaxis0, error_pi_02, color = 'green',linewidth=1.5, label='Mixed Iterations 20%')
ax0.plot(xaxis0, error_pi_04, color = 'black',linewidth=1.5, label='Mixed Iterations 40%')
ax0.plot(xaxis0, error_pi_06, color = 'orange',linewidth=1.5, label='Mixed Iterations 60%')
ax0.plot(xaxis0, error_pi_08, color = 'purple',linewidth=1.5, label='Mixed Iterations 80%')
ax0.plot(xaxis0, error_pi_10, color = 'blue', linewidth=1.5,label='Value Iteration')
ax0.legend(loc='lower right',prop={'size': 6})
ax0.set_xlabel('Iterations')
ax0.set_xlim([0,max_iter])
ax0.set_ylabel(r'$\pi^*\equiv\pi_t$')

#fig0.savefig('./policy_evolution_maze11x11_300iter.png', format='png', dpi=1200)
# For saving arrays in 3D :arr_reshaped = error_vi_l.reshape(error_vi_l.shape[0], -1)
#                          np.savetxt("error_Q19x19_2.0.txt", arr_reshaped)
