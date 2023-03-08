# -*- coding: utf-8 -*-
"""
Created on Mon Jan 23 17:15:11 2023

@author: polmu
"""

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

barWidth = 0.2

eighty = [100, 44.44, 50.45]
sixty = [50, 2.78, -32.73]
forty = [1.1, -48.57, -61.05]
twenty=[-33.33, -71.2,-74.19]

r1 = np.arange(len(sixty))
r2 = [x + barWidth for x in r1]
r3 = [x + barWidth for x in r2]
r4 = [x + barWidth for x in r3]

fig0 = plt.figure(0, figsize = (6, 4))
ax0 = fig0.add_subplot(1, 1, 1)
ax0.bar(r1, eighty, color='b', width=barWidth, edgecolor='white', label='MI 80%')
ax0.bar(r2, sixty, color='g', width=barWidth, edgecolor='white', label='MI 60%')
ax0.bar(r3, forty, color='r', width=barWidth, edgecolor='white', label='MI 40%')
ax0.bar(r4, twenty, color='y', width=barWidth, edgecolor='white', label='MI 20%')
ax0.set_ylabel('Computational Complexity Improvement (%)')
plt.xticks([r + barWidth for r in range(len(sixty))], ['RiverSwim MDP', 'GridWorld 7x7', 'GridWorld 11x11'],fontsize=12)
 
plt.legend(prop={'size': 10})
plt.show()


################################################

# barWidth = 0.2

# seventy = [-51, 76.32]
# fifty = [-61.37, 28.95]
# twentyfive = [-63.72, -29.63]

# r1 = np.arange(len(seventy))
# r2 = [x + barWidth for x in r1]
# r3 = [x + barWidth for x in r2]

# fig0 = plt.figure(0, figsize = (6, 4))
# ax0 = fig0.add_subplot(1, 1, 1)
# ax0.bar(r1, seventy, color='b', width=barWidth, edgecolor='white', label='MI 75%')
# ax0.bar(r2, fifty, color='g', width=barWidth, edgecolor='white', label='MI 50%')
# ax0.bar(r3, twentyfive, color='r', width=barWidth, edgecolor='white', label='MI 25%')
# ax0.set_ylabel('Computational Complexity Improvement (%)',fontsize=10)
# plt.xticks([r + barWidth for r in range(len(twentyfive))], ['Mountain Car', 'Inverted Pendulum'],fontsize=14)
 
# plt.legend(prop={'size': 10})
# plt.show()