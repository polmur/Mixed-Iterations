# -*- coding: utf-8 -*-
"""
Created on Thu Oct 20 15:32:25 2022

@author: polmu
"""
import math
import os
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from functions_pendulum import *

#281x281 states 


position=np.linspace(-math.pi/2,math.pi/2,num=76)
velocity=np.linspace(-6,6, num=76)
pi1 = np.loadtxt("data/pi_q_all.txt")
pi1= pi1.reshape(pi1.shape[0], pi1.shape[1]//76,76)
error_pi = np.loadtxt("error_percentage_mean3.txt")
states=np.zeros((3,5,150,50))
i_state=np.zeros(2)
state=i_state 

for d in range(50):
    for j in range(5):
        i_state=np.zeros(2)
        i_state[0]=random.randint(-800, 800)/1000
        i_state[1]=0
        pi_star=pi1[j]
        state=i_state 
        for i in range(150):
            idx_p = (np.abs(position - state[0])).argmin()
            idx_v = (np.abs(velocity - state[1])).argmin()
            r=random.uniform(0, 1)
            if r<0.85:
                u=pi_star[idx_p][idx_v]
            else:
                u=random.randint(-1, 1)
            state[0],state[1]=next_state_vi(position[idx_p],velocity[idx_v],u)
            states[0][j][i][d]=state[0]
            states[1][j][i][d]=state[1]
            states[2][j][i][d]=u

positions=states[0]
velocities=states[1]
inputs=states[2]    

positions_0=positions[4,:,:]
positions_25=positions[3,:,:]
positions_50=positions[2,:,:]
positions_75=positions[1,:,:]
positions_1=positions[0,:,:]

velocities_75=velocities[0,:,:]
inputs_75=inputs[0,:,:]

# r=random.randint(0,49)
# fig, (ax0, ax1, ax2) = plt.subplots(3, sharex=True)
# fig.suptitle('States and Input Sequence Inverted Pendulum')
# ax0.set(ylabel='$\Theta_t$')
# ax0.set_ylim([-2, 2])
# ax0.set_xlim([0, 140])
# ax1.set(ylabel='$\omega_t$')
# ax1.set_ylim([-6, 6])
# ax1.set_xlim([0, 140])
# ax2.set(xlabel='Steps', ylabel='$u_t$')
# ax2.set_ylim([-1.05, 1.05])
# ax2.set_xlim([0, 140])
# plot_pos=np.transpose(positions_1)
# plot_vel=np.transpose(velocities_75)
# plot_in=np.transpose(inputs_75)
# ax0.plot(plot_pos[r])
# ax1.plot(plot_vel[r])
# ax2.plot(plot_in[r])

positions_25_00=np.percentile(positions_0, 25,axis=1)
positions_25_25=np.percentile(positions_25, 25,axis=1)
positions_25_50=np.percentile(positions_50, 25,axis=1)
positions_25_75=np.percentile(positions_75, 25,axis=1)
positions_25_1=np.percentile(positions_1, 25,axis=1)

positions_75_00=np.percentile(positions_0, 75,axis=1)
positions_75_25=np.percentile(positions_25, 75,axis=1)
positions_75_50=np.percentile(positions_50, 75,axis=1)
positions_75_75=np.percentile(positions_75, 75,axis=1)
positions_75_1=np.percentile(positions_1, 75,axis=1)

positions_0_00=np.percentile(positions_0, 20,axis=1)
positions_0_25=np.percentile(positions_25, 20,axis=1)
positions_0_50=np.percentile(positions_50, 20,axis=1)
positions_0_75=np.percentile(positions_75, 20,axis=1)
positions_0_1=np.percentile(positions_1, 20,axis=1)

positions_100_00=np.percentile(positions_0, 90,axis=1)
positions_100_25=np.percentile(positions_25, 90,axis=1)
positions_100_50=np.percentile(positions_50, 90,axis=1)
positions_100_75=np.percentile(positions_75, 90,axis=1)
positions_100_1=np.percentile(positions_1, 90,axis=1)


positions_50_00=np.percentile(positions_0, 50,axis=1)
positions_50_25=np.percentile(positions_25, 50,axis=1)
positions_50_50=np.percentile(positions_50, 50,axis=1)
positions_50_75=np.percentile(positions_75, 50,axis=1)
positions_50_1=np.percentile(positions_1, 50,axis=1)

xaxis = [(i+1)for i in range(150)]
xaxis[:] = [xaxis - 1 for xaxis in xaxis]

fig1 = plt.figure(1, figsize = (6, 4))
ax1 = fig1.add_subplot(1, 1, 1)
ax1.set_ylim([-1, 1])
ax1.set_xlim([0, 150])
ax1.set_ylabel('$\Theta_t$', fontsize=14)
ax1.set_xlabel('Time Steps', fontsize=14)
ax1.plot(xaxis, positions_50_00, color = 'red',linewidth=1.5, label='QL')
ax1.plot(xaxis, positions_50_25, color = 'green',linewidth=1.5, label='MI 25%')
ax1.plot(xaxis, positions_50_50, color = 'black',linewidth=1.5, label='MI 50%')
ax1.plot(xaxis, positions_50_75, color = 'orange',linewidth=1.5, label='MI 75%')
ax1.plot(xaxis, positions_50_1, color = 'blue', linewidth=1.5,label='VI')
ax1.legend(loc='lower right',prop={'size': 10})
ax1.fill_between(xaxis, positions_25_00, positions_75_00, color = 'red', alpha=0.4)
ax1.fill_between(xaxis, positions_0_00, positions_100_00, color = 'red', alpha=0.4)
ax1.fill_between(xaxis, positions_25_25, positions_75_25, color = 'green', alpha=0.4)
ax1.fill_between(xaxis, positions_0_25, positions_100_25, color = 'green', alpha=0.4)
ax1.fill_between(xaxis, positions_25_50, positions_75_50, color = 'black', alpha=0.4)
ax1.fill_between(xaxis, positions_0_50, positions_100_50, color = 'black', alpha=0.4)
ax1.fill_between(xaxis, positions_25_75, positions_75_75, color = 'orange', alpha=0.4)
ax1.fill_between(xaxis, positions_0_75, positions_100_75, color = 'orange', alpha=0.4)
ax1.fill_between(xaxis, positions_25_1, positions_75_1, color = 'blue', alpha=0.4)
ax1.fill_between(xaxis, positions_0_1, positions_100_1, color = 'blue', alpha=0.4)

error_pi=np.absolute(error_pi-np.max(error_pi))
error_pi_02=error_pi[0:70,4]
error_pi_04=error_pi[0:70,3]
error_pi_06=error_pi[0:70,2]
error_pi_08=error_pi[0:70,1]
error_pi_10=error_pi[0:70,0]
xaxis0 = [(i+1)for i in range(70)]
xaxis0[:] = [xaxis0 - 1 for xaxis0 in xaxis0]

fig0 = plt.figure(0, figsize = (6, 4))
ax0 = fig0.add_subplot(1, 1, 1)
ax0.plot(xaxis0, error_pi_02, color = 'red',linewidth=1.5, label='QL')
ax0.plot(xaxis0, error_pi_04, color = 'green',linewidth=1.5, label='MI 25%')
ax0.plot(xaxis0, error_pi_06, color = 'black',linewidth=1.5, label='MI 50%')
ax0.plot(xaxis0, error_pi_08, color = 'orange',linewidth=1.5, label='MI 75%')
ax0.plot(xaxis0, error_pi_10, color = 'blue', linewidth=1.5,label='VI')
ax0.legend(loc='upper right',prop={'size': 10})
ax0.set_xlabel('Iterations',fontsize=14)
ax0.set_xlim([0,70])
ax0.set_ylim([0,1200])
ax0.set_ylabel(r'$\Vert \delta\:(\pi_t - \pi^*)\Vert_{\infty}$', fontsize=14)
