# -*- coding: utf-8 -*-
"""
Created on Fri Jun  3 13:02:28 2022

@author: polmu
"""

import numpy as np
from numpy import linalg
import random
import copy

# array[y,x]
# 1 - 'N', 2 -'S', 3 - 'W', 4 -'E'
def next_state(s,a,p,maze):
    y=s[0]
    x=s[1]
    r=np.random.rand()
    if a==0:
        if r<p:
            next_state=[y-1,x]
        else:
            k = random.randint(0, 1)
            if k==1:
                next_state=random.choice(([y+1,x],[y,x+1],[y,x-1]))
            else:
                next_state=[y,x]
    if a==1:
        if r<p:
            next_state=[y+1,x]
        else:
            k = random.randint(0, 1)
            if k==1:
                next_state=random.choice(([y-1,x],[y,x+1],[y,x-1]))
            else:
                next_state=[y,x]
    if a==2:
        if r<p:
            next_state=[y,x-1]
        else:
            k = random.randint(0, 1)
            if k==1:
                next_state=random.choice(([y-1,x],[y,x+1],[y+1,x]))
            else:
                next_state=[y,x]
    if a==3:
        if r<p:
            next_state=[y,x+1]
        else:
            k = random.randint(0, 1)
            if k==1:
                next_state=random.choice(([y-1,x],[y,x-1],[y+1,x]))
            else:
                next_state=[y,x]
    
    return next_state[0], next_state[1]

def generate_probabilities(actions,states,p,maze, maze_y,maze_x):
    probabilities=np.zeros((len(actions),maze_y,maze_x,maze_y,maze_x))
    for s in states:
        for a in actions:
           if a==0:
               probabilities[a][s[0]][s[1]][s[0]][s[1]]=0.1
               probabilities[a][s[0]][s[1]][s[0]-1][s[1]]=p
               probabilities[a][s[0]][s[1]][s[0]+1][s[1]]=(0.9-p)/3
               probabilities[a][s[0]][s[1]][s[0]][s[1]-1]=(0.9-p)/3
               probabilities[a][s[0]][s[1]][s[0]][s[1]+1]=(0.9-p)/3
           if a==1:
               probabilities[a][s[0]][s[1]][s[0]][s[1]]=0.1
               probabilities[a][s[0]][s[1]][s[0]-1][s[1]]=(0.9-p)/3
               probabilities[a][s[0]][s[1]][s[0]+1][s[1]]=p
               probabilities[a][s[0]][s[1]][s[0]][s[1]-1]=(0.9-p)/3
               probabilities[a][s[0]][s[1]][s[0]][s[1]+1]=(0.9-p)/3
           if a==2:
               probabilities[a][s[0]][s[1]][s[0]][s[1]]=0.1
               probabilities[a][s[0]][s[1]][s[0]-1][s[1]]=(0.9-p)/3
               probabilities[a][s[0]][s[1]][s[0]+1][s[1]]=(0.9-p)/3
               probabilities[a][s[0]][s[1]][s[0]][s[1]-1]=p
               probabilities[a][s[0]][s[1]][s[0]][s[1]+1]=(0.9-p)/3
           if a==3:
               probabilities[a][s[0]][s[1]][s[0]][s[1]]=0.1
               probabilities[a][s[0]][s[1]][s[0]-1][s[1]]=(0.9-p)/3
               probabilities[a][s[0]][s[1]][s[0]+1][s[1]]=(0.9-p)/3
               probabilities[a][s[0]][s[1]][s[0]][s[1]-1]=(0.9-p)/3
               probabilities[a][s[0]][s[1]][s[0]][s[1]+1]=p
    return probabilities

def gen_rewards(maze,maze_x,maze_y,m,states):
    rewards= maze*-30
    rewards[rewards == 0] = -1
    rewards[maze_x-2,maze_y-2]=100
    rewards_dim=np.stack([rewards,rewards,rewards,rewards])
    for k in range(m-1):   
        for s in states:          
            if k==0:
                if rewards[s[0]-1,s[1]]==-30:
                    rewards_dim[k][s[0]][s[1]]=-30
                else:
                    rewards_dim[k][s[0]][s[1]]=-1
            if k==1:
                if rewards[s[0]+1,s[1]]==-30:
                    rewards_dim[k][s[0]][s[1]]=-30
                else:
                    rewards_dim[k][s[0]][s[1]]=-1
            if k==2:
                if rewards[s[0],s[1]-1]==-30:
                    rewards_dim[k][s[0]][s[1]]=-30
                else:
                    rewards_dim[k][s[0]][s[1]]=-1
            if k==3:
                if rewards[s[0],s[1]+1]==-30:
                    rewards_dim[k][s[0]][s[1]]=-30
                else:
                    rewards_dim[k][s[0]][s[1]]=-1
    rewards_dim[1][maze_x-2][maze_y-2]=100
    rewards_dim[3][maze_x-2][maze_y-2]=100
    return rewards_dim

def value_iteration(max_iter, states,actions, maze,probabilities, rewards,gamma, n1, n2, m):
    Q_vi= np.zeros((n1+1,n2+1,m))
    J = np.zeros((n1+1,n2+1))
    pi_star=np.zeros((n1+1,n2+1))
    # Start value iteration
    for i in range(max_iter):
        Q_new = np.zeros((n1+1,n2+1))
        np.random.shuffle(states)
        for s in states:            
            Q_max = np.zeros(m)
            s1=s[0]
            s2=s[1]
            for a in actions:
                Q_a=rewards[a][s1][s2]
                # Compute state value
                s_next=([s1-1,s2],[s1+1,s2],[s1,s2-1],[s1,s2+1],[s1,s2])
                for sx in s_next :  
                    Q_a += probabilities[a][s1][s2][sx[0]][sx[1]]*gamma * J[sx[0]][sx[1]]   
                Q_max[a] = Q_a
            Q_vi[s1,s2]=Q_max
            Q_new[s1,s2]=np.max(Q_max)  # Update Q with highest value
            pi_star[s1,s2]=np.argmax(Q_max)
            # Update Q functions and compute error
        J = Q_new
    return pi_star, Q_vi, J

def q_learning(max_iter, episodes, states,actions,p, maze,probabilities, rewards,gamma,Q_star, pi_star, n1, n2, m):
    # Start Q-Learning for 100 episodes
    error_Q=np.zeros((episodes,max_iter))
    error_pi=np.zeros((episodes,max_iter))
    for d in range(episodes):
        Q = np.random.random(size=(n1+1,n2+1,m))
        Q_new_q = np.zeros((n1+1,n2+1,m))        
        for i in range(max_iter):
            alpha=1/((i+1)**0.65)
            for s in states:
                s1=s[0]
                s2=s[1]
                for a in actions: 
                    s_y,s_x=next_state(s,a,p,maze)
                    # Update Q functions and compute error
                    Q_new_q[s1][s2][a] += alpha*(rewards[a][s1][s2] + gamma*np.max(Q[s_y][s_x])-Q[s1][s2][a])
            Q = Q_new_q # Update Q with next step Q 
            pi=np.argmax(Q, axis=2)
            error_Q[d][i]=linalg.norm(Q_star-Q)
            error_pi[d][i]=np.sum(pi == pi_star)-((n1+1)*(n2+1)-(n1-1)*(n2-1))+1
    return pi,Q,error_Q,error_pi

def clean_pi(walls,pi_star,maze):
    for s in walls:
        a=s[0]
        b=s[1]
        pi_star[a][b]=-10
    pi_star = np.delete(pi_star, np.all(maze[..., :] == 1, axis=0), axis=1) # Deletes all 0-value columns
    pi_star = np.delete(pi_star, np.all(maze[..., :] == 1, axis=1), axis=0) # Deletes all 0-value rows
    return pi_star

def randomize_probs(p,percentage):    
    pro = copy.copy(p)
    p1,p2,p3,p4,p5= np.where(p>0)
    size=len(p1)
    total_sum=np.sum(p)
    current_sum=total_sum
    if percentage==0:
        pro[pro != 0]=np.nan
    if ((percentage>0) and (percentage <1)):
        while percentage*total_sum < current_sum:
            index=random.randint(0, size-1)
            pro[p1[index]][p2[index]][p3[index]][p4[index]][p5[index]]=np.nan
            current_sum=np.nansum(pro)
    return pro

       