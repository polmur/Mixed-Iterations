# -*- coding: utf-8 -*-
"""
Created on Tue Jun 14 11:34:38 2022

@author: polmu
"""
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

prior_percentage=[1,0.8,0.6,0.4,0.2,0]
max_iter=100

error_pi = np.loadtxt("data\error_pi7x7.txt")
error_pi = error_pi.reshape(error_pi.shape[0], error_pi.shape[1]//len(prior_percentage),len(prior_percentage))
error_Q = np.loadtxt('data\error_Q7x7.txt')
error_Q = error_Q.reshape(error_Q.shape[0], error_Q.shape[1]//len(prior_percentage),len(prior_percentage))
maze=np.loadtxt('data\maze_7x7.txt')
maze[len(maze)-1][len(maze)-1]=2

error_pi_percentage=np.mean(error_pi, axis=0)
error_pi_percentage=np.absolute(error_pi_percentage-np.max(error_pi_percentage))
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
ax0.plot(xaxis0, error_pi_00, color = 'red',linewidth=1.5, label='QL')
ax0.plot(xaxis0, error_pi_02, color = 'green',linewidth=1.5, label='MI 20%')
ax0.plot(xaxis0, error_pi_04, color = 'black',linewidth=1.5, label='MI 40%')
ax0.plot(xaxis0, error_pi_06, color = 'orange',linewidth=1.5, label='MI 60%')
ax0.plot(xaxis0, error_pi_08, color = 'purple',linewidth=1.5, label='MI 80%')
ax0.plot(xaxis0, error_pi_10, color = 'blue', linewidth=1.5,label='VI')
ax0.legend(loc='upper right',prop={'size': 10})
ax0.set_xlabel('Iterations',fontsize=14)
ax0.set_xlim([0,max_iter])
ax0.set_ylim([0,15])
ax0.set_ylabel(r'$\Vert \delta\:(\pi_t - \pi^*)\Vert_{\infty}$',fontsize=14)
ax0.set_title("7x7 Grid World with 25 randomizations of P",  loc='center')


fig0.savefig('./error_pi_7x7.png', format='png', dpi=800)


error_00=error_Q[:,:,5]
error_02=error_Q[:,:,4]
error_04=error_Q[:,:,3]
error_06=error_Q[:,:,2]
error_08=error_Q[:,:,1]
error_10=error_Q[:,:,0]
xaxis = [(i+1)for i in range(max_iter)]
xaxis[:] = [xaxis - 1 for xaxis in xaxis]

# error_25_00=np.percentile(error_00, 25,axis=0)
# error_25_02=np.percentile(error_02, 25,axis=0)
# error_25_04=np.percentile(error_04, 25,axis=0)
# error_25_06=np.percentile(error_06, 25,axis=0)
# error_25_08=np.percentile(error_08, 25,axis=0)
# error_25_10=np.percentile(error_10, 25,axis=0)

# error_75_00=np.percentile(error_00, 75,axis=0)
# error_75_02=np.percentile(error_02, 75,axis=0)
# error_75_04=np.percentile(error_04, 75,axis=0)
# error_75_06=np.percentile(error_06, 75,axis=0)
# error_75_08=np.percentile(error_08, 75,axis=0)
# error_75_10=np.percentile(error_10, 75,axis=0)

# error_0_00=np.percentile(error_00, 0,axis=0)
# error_0_02=np.percentile(error_02, 0,axis=0)
# error_0_04=np.percentile(error_04, 0,axis=0)
# error_0_06=np.percentile(error_06, 0,axis=0)
# error_0_08=np.percentile(error_08, 0,axis=0)
# error_0_10=np.percentile(error_10, 0,axis=0)

# error_100_00=np.percentile(error_00, 90,axis=0)
# error_100_02=np.percentile(error_02, 90,axis=0)
# error_100_04=np.percentile(error_04, 90,axis=0)
# error_100_06=np.percentile(error_06, 90,axis=0)
# error_100_08=np.percentile(error_08, 90,axis=0)
# error_100_10=np.percentile(error_10, 90,axis=0)


error_50_00=np.percentile(error_00, 50,axis=0)
error_50_02=np.percentile(error_02, 50,axis=0)
error_50_04=np.percentile(error_04, 50,axis=0)
error_50_06=np.percentile(error_06, 50,axis=0)
error_50_08=np.percentile(error_08, 50,axis=0)
error_50_10=np.percentile(error_10, 50,axis=0)

# minmax_00=abs(error_100_00-error_0_00)
# minmax_02=abs(error_100_02-error_0_02)
# minmax_04=abs(error_100_04-error_0_04)
# minmax_06=abs(error_100_06-error_0_06)
# minmax_08=abs(error_100_08-error_0_08)
# minmax_10=abs(error_100_10-error_0_10)

# error_00_2575=abs(error_75_00-error_25_00)
# error_02_2575=abs(error_75_02-error_25_02)
# error_04_2575=abs(error_75_04-error_25_04)
# error_06_2575=abs(error_75_06-error_25_06)
# error_08_2575=abs(error_75_08-error_25_08)
# error_10_2575=abs(error_75_10-error_25_10)

fig1 = plt.figure(1, figsize = (6, 4))
ax1 = fig1.add_subplot(1, 1, 1)
ax1.plot(xaxis, error_50_00, color = 'red',linewidth=1.5, label='QL')
ax1.plot(xaxis, error_50_02, color = 'green',linewidth=1.5, label='MI 20%')
ax1.plot(xaxis, error_50_04, color = 'black',linewidth=1.5, label='MI 40%')
ax1.plot(xaxis, error_50_06, color = 'orange',linewidth=1.5, label='MI 60%')
ax1.plot(xaxis, error_50_08, color = 'purple',linewidth=1.5, label='MI 80%')
ax1.plot(xaxis, error_50_10, color = 'blue', linewidth=1.5,label='VI')
# ax1.fill_between(xaxis, error_25_00, error_75_00, color = 'red', alpha=0.4)
# ax1.fill_between(xaxis, error_0_00, error_100_00, color = 'red', alpha=0.4)
# ax1.fill_between(xaxis, error_25_02, error_75_02, color = 'green', alpha=0.4)
# ax1.fill_between(xaxis, error_0_02, error_100_02, color = 'green', alpha=0.4)
# ax1.fill_between(xaxis, error_25_04, error_75_04, color = 'grey', alpha=0.4)
# ax1.fill_between(xaxis, error_0_04, error_100_04, color = 'grey', alpha=0.4)
# ax1.fill_between(xaxis, error_25_06, error_75_06, color = 'orange', alpha=0.4)
# ax1.fill_between(xaxis, error_0_06, error_100_06, color = 'orange', alpha=0.4)
# ax1.fill_between(xaxis, error_25_08, error_75_08, color = 'purple', alpha=0.4)
# ax1.fill_between(xaxis, error_0_08, error_100_08, color = 'purple', alpha=0.4)
# #ax1.fill_between(xaxis, error_25_10, error_75_10, color = 'blue', alpha=0.4)
# #ax1.fill_between(xaxis, error_0_10, error_100_10, color = 'blue', alpha=0.4)

ax1.set_yscale('linear')
ax1.set_ylim([0,200])
ax1.set_xlim([0,max_iter])
ax1.legend(loc='upper right',prop={'size': 10})
ax1.set_xlabel('Iterations',fontsize=14)
ax1.set_ylabel(r'$||Q^*-Q_t||_\infty$',fontsize=14)
ax1.set_title("7x7 Grid World with 25 randomizations of P",  loc='center')

fig1.savefig('./error_Q_7x7.png', format='png', dpi=800)

# fig2 = plt.figure(2,figsize = (4, 4))
# plt.pcolormesh(maze,cmap='Oranges')
# plt.title("31x31 Maze")
# plt.xticks([]) # remove the tick marks by setting to an empty list
# plt.yticks([]) # remove the tick marks by setting to an empty list
# plt.gca().set_aspect('equal') #set the x and y axes to the same scale
# plt.gca().invert_yaxis()
# #fig2.savefig('./maze31x31.png', format='png', dpi=1200)

