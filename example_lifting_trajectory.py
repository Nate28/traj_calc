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
import time
import cantera as ct
import flow_calc_lib as fcl

# tc = reload(tc)

R_e = ac.R_earth.value
g_0 = ac.g0.value
M_e = ac.M_earth.value

atm_steps = 2000
integrator_steps = 1E5
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
h_init = 200E3#76.2E3
h_end = 0
C_d = 0.84
C_l = 0.84
m = 90687.495
A = 249.909178
R_n = 0.3048
L = 32.766
gamma_init = -np.deg2rad(0.01)
V_init = 7010.4 #7.6E3

## ISS
# C_d = 2.0
# C_l = 0.00
# m = 419455.0
# A = 1800.0
# R_n = 4.4
# L = 100.0
# gamma_init = np.deg2rad(2.9)
# V_init = 7.66E3
# h_init = 416E3
# h_end = 0

def aero_dat(Re, Ma, Kn, p_inf, q_inf, rho_inf, gamma_var, Cd_prev, Cl_prev):
	"""
	Constant aerdynamic coefficients - space shuttle example
	"""
	C_d = 0.84
	C_l = 0.84
	C_s = 0.0
	
	return [C_d, C_l, C_s]

# def aero_dat(Re, Ma, Kn, p_inf, q_inf, rho_inf, gamma_var, Cd_prev, Cl_prev):
# 	"""
# 	This is a sample function for the spacecraft_var class.  It serves as an 
# 	example of the use of an aerodynamic database for spacecraft whose 
# 	aero coefficients are to be recalculated during a simulation.  
# 	
# 	Note that the variables available for correlations are currently Reynolds,
# 	Mach, and Knudsen numbers.  These must always be passed to the aero_dat
# 	function, even if they remain unused.  Also, the order in which arguments are
# 	passed and variables returned must not be altered.
# 	"""
# 	#C_d = 2.0
# 	C_l = 0.0
# 	C_s = 0.0
# 	
# 	# Drag coefficient correlation for sphere
# 	# http://www.chem.mtu.edu/~fmorriso/DataCorrelationForSphereDrag2013.pdf
# 	C_d = (24 / Re) + \
# 		((2.6 * Re / 5.0) / (1 + (Re / 5.0)**1.52)) + \
# 		((0.411 * (Re / 263E3)**-7.94) / (1 + (Re / 263E3)**-8.0)) + \
# 		(Re**0.8 / 461E3)
# 		
# 	return [C_d, C_l, C_s]

atm_init = [1000E3, -1E3]
h = np.linspace(atm_init[0], atm_init[1], atm_steps)

p = tc.planet('Earth', M_e, R_e, g_0)

#v = tc.spacecraft(C_d, m, A, R_n, L, Cl=C_l)
v = tc.spacecraft_var(aero_dat, C_d, C_l, 0.0, m, A, R_n, L, integrator_steps)

#atm = tc.atmosphere_us76(h)
#t = tc.trajectory(v, atm, gamma_init, V_init, g_0, R_e)

atm_2 = tc.atmosphere_nrl(h)
# t = tc.trajectory_lifting(v, atm_2, gamma_init, V_init, g_0, R_e, h_init, 
# 	h_end, integrator_steps, console_output=True)
# t = tc.trajectory_lifting(v, atm_2, gamma_init, V_init, g_0, R_e, h_init, 
# 	h_end, integrator_steps)
t = tc.trajectory_lifting(v, atm_2, p, gamma_init, V_init, h_init, 
	h_end, integrator_steps, console_output=True)

t.initialise()

# Run and time
start_time = time.time()
t.simulate_dopri(dt=0.25)
end_time = time.time()

#%% Post process
t.calculate_heating()
t.post_calc()

t.plot_triple()
t.show_regimes()
t.plot_trajectory()
t.show_regimes()

#%% Plot
plt.figure()
plt.xlabel(r'$\dot{Q} \; \left( \frac{kW}{m^2} \right)$', fontsize=17)
plt.ylabel(r'$h \; \left( km \right)$', fontsize=17)
plt.plot(t.qdot.conv.bj/1000, t.h/1000, label='Brandis-Johnston')
plt.plot(t.qdot.conv.dh/1000, t.h/1000, label='Detra-Hidalgo')
plt.plot(t.qdot.conv.s/1000, t.h/1000, label='Smith')
plt.plot(t.qdot.conv.sg/1000, t.h/1000, label='Sutton-Graves')
plt.grid(True)
plt.tight_layout()
plt.xscale('log')
plt.legend(loc=0)

plt.figure()
plt.xlabel(r'$\dot{Q} \; \left( \frac{kW}{m^2} \right)$', fontsize=17)
plt.ylabel(r'$h \; \left( km \right)$', fontsize=17)
plt.plot(t.qdot.net.bj/1000, t.h/1000, label='Brandis-Johnston')
plt.plot(t.qdot.net.s/1000, t.h/1000, label='Smith')
#plt.plot(t.qdot.net.sg_ts/1000, t.h/1000, label='Sutton-Graves, Tauber-Sutton')
plt.grid(True)
plt.tight_layout()
plt.xscale('log')
plt.legend(loc=0)

plt.show()

print('\n=== TRAJECTORY SIMULATION COMPLETE ===')
print('%i ITERATIONS COMPUTED IN %f SECS\n' % (t.index, end_time - start_time))