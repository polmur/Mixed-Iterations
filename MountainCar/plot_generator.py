# -*- coding: utf-8 -*-
"""
Created on Tue Jun 14 11:34:38 2022

@author: polmu
"""
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

prior_percentage=[1,0.75,0.5,0.25,0]
max_iter=500

pi1 = np.loadtxt("data/pi1.txt")
p1= pi1.reshape(pi1.shape[0], pi1.shape[1]//141,141)

# error_pi1 = np.loadtxt("data\error_pi_mountain2.0.txt")
# error_pi1 = error_pi1.reshape(error_pi1.shape[0], error_pi1.shape[1]//len(prior_percentage),len(prior_percentage))
error_pi = np.loadtxt("data/error_pi.txt")
# error_pi_percentage=np.mean(error_pi, axis=0)
error_pi_percentage=error_pi
error_pi_percentage=np.absolute(error_pi_percentage-np.max(error_pi_percentage))
error_pi_02=error_pi_percentage[0:max_iter,4]
error_pi_04=error_pi_percentage[0:max_iter,3]
error_pi_06=error_pi_percentage[0:max_iter,2]
error_pi_08=error_pi_percentage[0:max_iter,1]
error_pi_10=error_pi_percentage[0:max_iter,0]
xaxis0 = [(i+1)for i in range(max_iter)]
xaxis0[:] = [xaxis0 - 1 for xaxis0 in xaxis0]

fig0 = plt.figure(0, figsize = (6, 4))
ax0 = fig0.add_subplot(1, 1, 1)
ax0.plot(xaxis0, error_pi_02, color = 'red',linewidth=1.5, label='QL')
ax0.plot(xaxis0, error_pi_04, color = 'green',linewidth=1.5, label='MI 25%')
ax0.plot(xaxis0, error_pi_06, color = 'orange',linewidth=1.5, label='MI 50%')
ax0.plot(xaxis0, error_pi_08, color = 'purple',linewidth=1.5, label='MI 75%')
ax0.plot(xaxis0, error_pi_10, color = 'blue', linewidth=1.5,label='VI')
ax0.legend(loc='upper right',prop={'size': 10})
ax0.set_xlabel('Iterations',fontsize=14)
ax0.set_xlim([0,max_iter])
ax0.set_ylim([0,7000])
ax0.set_ylabel(r'$\Vert \delta\:(\pi_t - \pi^*)\Vert_{\infty}$',fontsize=14)
