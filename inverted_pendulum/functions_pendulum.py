# -*- coding: utf-8 -*-
"""
Created on Fri Jun  3 13:02:28 2022

@author: polmu
"""

import numpy as np
from numpy import linalg
import random
import math

def next_state(p,v,a,pr):
    
    r=np.random.rand()
    g=9.8
    m=2
    M=8
    l=0.5
    alpha=0.1
    if r<pr:
        velocity_n= v + ((g*np.sin(p)-alpha*m*l*np.power(v,2)*(np.sin(2*p)/2)-alpha*np.cos(p)*150*a)/(4/3)*l -alpha*m*l*(np.power(np.cos(p),2)))*0.1  #3 is best
        position_n= p+velocity_n*0.1
        if position_n > 1.58:
            position_n= math.pi/2
            velocity_n=0
        if position_n < -1.58:
            position_n= -math.pi/2
            velocity_n=0
    else:
        i=random.randint(-1,1)
        velocity_n= v + ((g*np.sin(p)-alpha*m*l*np.power(v,2)*(np.sin(2*p)/2)-alpha*np.cos(p)*150*a)/(4/3)*l -alpha*m*l*(np.power(np.cos(p),2)))*0.1  #3 is best
        position_n= p+velocity_n*0.1
        if position_n > 1.58:
            position_n= math.pi/2
            velocity_n=0
        if position_n < -1.58:
            position_n= -math.pi/2
            velocity_n=0
    return position_n,velocity_n 

def next_state_vi(p,v,a):
    g=9.82
    m=2
    M=8
    l=0.5
    alpha=0.1     
    velocity_n= v + ((g*np.sin(p)-alpha*m*l*np.power(v,2)*(np.sin(2*p)/2)-alpha*np.cos(p)*150*a)/(4/3)*l -alpha*m*l*(np.power(np.cos(p),2)))*0.1  #3 is best
    position_n= p+velocity_n*0.1
    
    if position_n > 1.58:
        position_n= math.pi/2
        velocity_n=0
    if position_n < -1.58:
        position_n= -math.pi/2
        velocity_n=0
        
    return position_n, velocity_n 

def cost(position,velocity):
    J=100*math.cos(position)-5*velocity**2
    return J

def value_iteration_mountain(n1,n2,m1, max_iter,states,actions,position,velocity,pr):
    p_next=np.zeros(3)
    v_next=np.zeros(3)
    idx_p=np.array([0,0,0])
    idx_v=np.array([0,0,0])
    Q_vi= np.zeros((n1,n2,m1))
    J = np.random.random(size=(n1,n2))
    pi_star=np.zeros((n1,n2))
    #Start value iteration
    for i in range(max_iter):
        Q_new = np.zeros((n1,n2))
        #np.random.shuffle(states)
        for s in states:            
            Q_max = np.zeros(m1)
            p=position[s[0]]
            v=velocity[s[1]]            
            for a in actions:                
                p_next[a+1],v_next[a+1]=next_state_vi(p,v,a)
                idx_p[a+1] = (np.abs(position - p_next[a+1])).argmin()
                idx_v[a+1] = (np.abs(velocity - v_next[a+1])).argmin()
            for a in actions:
                probs=np.ones(3)*(1-pr)/2
                probs[a+1]=pr
                Q_a=cost(p,v)+probs[0]*0.95*J[idx_p[0]][idx_v[0]]+probs[1]*0.95*J[idx_p[1]][idx_v[1]]+probs[2]*0.95*J[idx_p[2]][idx_v[2]]
                Q_max[a+1] = Q_a                
            Q_vi[s[0],s[1]]=Q_max
            Q_new[s[0],s[1]]=np.max(Q_max)  # Update Q with highest value
            pi_star[s[0],s[1]]=np.argmax(Q_max)-1
            # Update Q functions and compute error
        J = Q_new
    return pi_star, J

def q_learning_mountain(n1,n2,m1, max_iter,states,actions,position,velocity,gamma,pr):    
    Q = np.random.random(size=(n1,n2,m1))
    Q_new_q = np.zeros((n1,n2,m1)) 
    for i in range(max_iter):
        alpha=1/(i+2)**0.5
        np.random.shuffle(states)
        for s in states:
            p=position[s[0]]
            v=velocity[s[1]]
            idx_p = (np.abs(position - p)).argmin()
            idx_v = (np.abs(velocity - v)).argmin()
            for a in actions:
                reward=cost(p,v)
                p_next,v_next=next_state(p,v,a,pr)
                idx_p_n = (np.abs(position - p_next)).argmin()
                idx_v_n = (np.abs(velocity - v_next)).argmin()
                # Update Q functions and compute error
                Q_new_q[idx_p][idx_v][a+1] += alpha*(reward + gamma*np.max(Q[idx_p_n][idx_v_n])-Q[idx_p][idx_v][a+1])
        Q = Q_new_q # Update Q with next step Q 
        pi=np.argmax(Q, axis=2)-1
    return pi, Q