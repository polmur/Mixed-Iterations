# -*- coding: utf-8 -*-
"""
Created on Thu May 12 14:51:18 2022

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
import copy

def next_state(s,a,probs):
    probabilities=probs[s][a]
    s1=np.random.choice(np.arange(0, 5), p=probabilities)
    return s1

def randomize_probs(p,percentage):    
    pro = copy.copy(p)
    for i in range(len(p)):
        for j in range(len(p[0])):
            for k in range(len(p[0][0])):                
                if pro[i][j][k]>0 :
                    r=np.random.uniform(0,1)
                    if r>percentage:
                        pro[i][j][k]=np.nan
    return pro
def value_iteration(max_iter, states, actions, probs, rewards,gamma):
    Q_vi = np.zeros((5,2))
    error_vi=np.zeros(max_iter)
    # Start value iteration
    for i in range(max_iter):
        Q_new = np.zeros((5,2))
        for s in states:
            Q_max = [0,0]
            for a in actions:
                Q_a=rewards[s][a]
                # Compute state value
                for s_next in states:
                    Q_a += probs[s][a][s_next] * (gamma * np.max(Q_vi[s_next]))            
                    # Store value best action 
                    Q_max[a] = Q_a
            Q_new[s]=Q_max  # Update Q with highest value
            
        # Update Q functions and compute error
        Q_vi = Q_new
    return Q_vi

def q_learning(Q_star,max_iter,episodes, states, actions, probs, rewards,gamma):
    error_Q=np.zeros((episodes,max_iter))
    # Start Q-Learning for 100 episodes
    for d in range(episodes):
        Q = np.random.random(size=(5,2))
        Q_new_q = np.zeros((5,2))
        for i in range(max_iter):    
            for s in states:
                for a in actions: 
                    s_next=next_state(s,a,probs)            
                    # Update Q functions and compute error
                    Q_new_q[s][a] += (rewards[s][a] + gamma*Q[s_next][a]-Q[s][a])
            Q = Q_new_q # Update Q with next step Q
            error_Q[d][i]=linalg.norm(Q_star-Q_new_q)
    return error_Q, Q
