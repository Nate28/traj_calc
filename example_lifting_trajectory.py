# -*- coding: utf-8 -*-
"""
Created on Fri Jun 26 12:40:03 2015

@author: scat5659
"""

import traj_calc as tc
import numpy as np
import matplotlib.pyplot as plt
import astropy.constants as ac
import heat_flux_lib as hfl
import time as tempus
import cantera as ct
import flow_calc_lib as fcl

start_time = tempus.time()

tc = reload(tc)

R_e = ac.R_earth.value
g_0 = ac.g0.value

steps = 2000
d = 4E-10

## Small sphere
#C_d = 0.47
#m = 1000
#A = 0.785
#R_n = 0.5
#L = 1.0

## Friendship 7
#C_d = 1.60 #1.15
#m = 11844.7245/9.81 #
#A = 2.812 #4.0
#R_n = 0.305
#L = 1.89
#C_l = 0#1.1
#gamma_init = np.deg2rad(12)
#V_init = 7010.4 #7.6E3
#h_init = 76.2E3#84.852E3
#h_end = 0

## Apollo 4
#C_d = 1.35 #1.15
#C_l = 0.36
#m = 5424.9
#A = 12.02
#R_n = 4.69
#L = 3.62
#gamma_init = np.deg2rad(1.5)
#V_init = 7010.4 #7.6E3
#h_init = 121920 #84.852E3
#h_end = 0

## Space shuttle
#h_init = 200E3#76.2E3
#h_end = 0
#C_d = 0.84
#C_l = 0.84
#m = 90687.495
#A = 249.909178
#R_n = 0.3048
#L = 32.766
#gamma_init = -np.deg2rad(0.01)
#V_init = 7010.4 #7.6E3

##ISS
C_d = 2.0
C_l = 0.00
m = 419455.0
A = 1800.0
R_n = 4.4
L = 100.0
gamma_init = np.deg2rad(2.9)
V_init = 7.66E3
h_init = 416E3
h_end = 0

atm_init = [1000E3, -1E3]
h = np.linspace(atm_init[0], atm_init[1], steps)
v = tc.spacecraft(C_d, m, A, R_n, L, Cl=C_l)

#atm = tc.atmosphere_us76(h)
#t = tc.trajectory(v, atm, gamma_init, V_init, g_0, R_e)

atm_2 = tc.atmosphere_nrl(h)

steps = 1E5
t = tc.trajectory_lifting(v, atm_2, gamma_init, V_init, g_0, R_e, h_init, h_end, steps)
t.initialise()
t.simulate_dopri(dt=0.25)
t.calculate_heating()
t.post_calc()

t.plot_triple()
t.show_regimes()
t.plot_trajectory()
t.show_regimes()

plt.show()

end_time = tempus.time()
print '\n=== TRAJECTORY SIMULATION COMPLETE ==='
print '%i ITERATIONS COMPUTED IN %f SECS\n' % (t.index, end_time - start_time)