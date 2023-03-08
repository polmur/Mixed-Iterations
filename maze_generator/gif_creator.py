# -*- coding: utf-8 -*-
"""
Created on Wed Jun 15 12:54:00 2022

@author: polmu
"""
import numpy as np
import pandas as pd
import os
import imageio
import matplotlib
import random
import matplotlib.pyplot as plt
from functions_maze import *

#np.savetxt('data\pi11x11.txt',pi_star)
maze=np.loadtxt('data\maze_31x31.txt')
pi=np.loadtxt('data\pi31x31.txt')
states=np.argwhere((maze==0)|(maze==2))
maze[len(maze)-1][len(maze)-1]=2

state_0=[18,10]
i=0
filenames=[]

while (state_0[0]<len(maze)-1) or (state_0[1]<len(maze)-1):
    state=state_0
    action=pi[state[0]][state[1]]
    if action==0:
        state[0]-=1
    if action==1:
        state[0]+=1
    if action==2:
        state[1]-=1
    if action==3:
        state[1]+=1
    
    maze[state[0]][state[1]]=3
    fig2 = plt.figure(2,figsize = (4, 4))
    plt.pcolormesh(maze,cmap='Oranges')
    plt.title("31x31 Maze")
    plt.xticks([]) # remove the tick marks by setting to an empty list
    plt.yticks([]) # remove the tick marks by setting to an empty list
    plt.gca().set_aspect('equal') #set the x and y axes to the same scale
    plt.gca().invert_yaxis()
    # create file name and append it to a list
    filename = f'{i}.png'
    filenames.append(filename)
    plt.savefig(filename)
    plt.show()       
    state_0=state
    maze[state[0]][state[1]]=0
    i+=1


with imageio.get_writer('31x31maze.gif', mode='I') as writer:
    for filename in filenames:
        image = imageio.imread(filename)
        writer.append_data(image)

for filename in set(filenames):
    os.remove(filename)