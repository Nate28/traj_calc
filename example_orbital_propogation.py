# -*- coding: utf-8 -*-
"""
Created on Sat Sep 12 18:07:20 2015

@author: hilbert
@modified: Nathan
"""
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import traj_calc as tc
import time
tc = reload(tc)

r_earth = 6371000
r_venus = 6051800
mu_earth = 3.98574405096*(10**14)
mu_venus = 0.815*mu_earth

M_vapr = 35

C_d = 0.8
C_l = 0.2
C_s = 0.0
A = 1.0
m_craft = 500.0
R_n = 1.0
L = 1.0

h_init = 150E3
X = r_earth + h_init
X0 = [X, 0, 0]  #inital position vector
V_mag = np.sqrt(mu_earth / (r_earth + h_init))
V0 = [0, V_mag, 0]  #inital velocity vector
sim_time = 60 * 60 * 24 * 2  #time to run the simulation

atm_steps = 1000
atm_init = [h_init * 10.0, -1E3]
h = np.linspace(atm_init[0], atm_init[1], atm_steps)

sc = tc.spacecraft(C_d, m_craft, A, R_n, L, Cl=C_l, Cs=C_s)
atm = tc.atmosphere_nrl(h)

t = tc.trajectory_aerobraking(mu_earth, sc, atm, V0, X0, sim_time, r_earth, dt=5)

start_time = time.time()
t.simulate_dopri(nsteps=1E8, rtol=1E-8)
end_time = time.time()
elapsed = end_time - start_time

plt.plot(t.pos_xyz[0:t.i, 0]/1E3, t.pos_xyz[0:t.i, 1]/1E3,'-r')  #plot orbit

try:
	i_100_km = np.argwhere(t.h < 100E3)[0]
	plt.scatter(t.pos_xyz[i_100_km, 0]/1E3, t.pos_xyz[i_100_km, 1]/1E3)
except:
	pass

def plot_sphere(R):
	u = np.linspace(0, 2 * np.pi, 100)
	v = np.linspace(0, np.pi, 100)
	
	x = R * np.outer(np.cos(u), np.sin(v))
	y = R * np.outer(np.sin(u), np.sin(v))
	z = R * np.outer(np.ones(np.size(u)), np.cos(v))
	ax.plot_surface(x, y, z, rstride=4, cstride=4, color='b', alpha=0.1, 
		linewidth=0.5, shade=True)

ax = plt.gca()
circ = plt.Circle((0, 0), radius=r_earth/1E3, color='b', fill=True, alpha=0.2)
ax.add_patch(circ)
plt.show()
plt.axis('equal')
plt.tight_layout()

fig = plt.figure(figsize=(10,10))
ax = fig.add_subplot(111, projection='3d')
plot_sphere(r_earth/1000)
ax.plot(t.pos_xyz[0:t.i, 0]/1E3, t.pos_xyz[0:t.i, 1]/1E3, t.pos_xyz[0:t.i, 2]/1E3,'-r')
ax.set_aspect(1)
plt.tight_layout()
plt.show()

print 'ELAPSED TIME: %f s' % elapsed