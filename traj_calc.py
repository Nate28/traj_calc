# -*- coding: utf-8 -*-
"""
=== TRAJ CALC ===
Re-entry trajectory calculator
"""

from __future__ import print_function
import numpy as np
# import aerocalc.std_atm as atm
#import astropy.constants as ac
import flow_calc_lib as fcl
import heat_flux_lib as hcl
from scipy import integrate
import cantera as ct
#import thermopy as tp
import nrlmsise_00_header as nrl_head
import nrlmsise_00 as nrlmsise
import j77sri as j77
import matplotlib.pyplot as plt
import scipy.interpolate as spint
import rotate_lib

__author__ 		= 'Nathan Donaldson'
__contributor__ 		= 'Hilbert van Pelt'
__email__ 			= 'nathan.donaldson@eng.ox.ac.uk'
__status__ 		= 'Development'
__version__ 		= '0.51'
__license__ 		= 'MIT'

# Altitude derivatives for forward Euler solver and ODE solver initialisation
# Velocity
def dv_dh(g, p_dyn, beta, V, gamma):
	dvdh = ((p_dyn / beta) - (g * np.sin(gamma))) / (V * np.sin(gamma))
	return dvdh

# Flightpath angle
def dgamma_dh(gamma, g, V, R, h):
	dgdh = -(np.cos(gamma) * (g * ((V**2) / (R + h)))) / ((V**2) * np.sin(gamma))
	return dgdh

# Time
def dt_dh(gamma, V):
	dtdh = -1 / (V * np.sin(gamma))
	return dtdh

# Ground range
def dr_dh(R, gamma, h):
	drdh = (R * np.cos(gamma)) / ((R + h) * np.sin(gamma))
	return drdh

# Time derivatives for forward Euler solver and ODE solver initialisation
# Velocity
def dv_dt(g, p_dyn, beta, gamma):
	dvdt = (-p_dyn / beta) + (g * np.sin(gamma))
# 	dvdt = g * ((-p_dyn / beta) + (np.sin(gamma)))
	return dvdt

# Flightpath angle
def dgamma_dt(gamma, g, V, R, h, L_D_ratio, p_dyn, beta):
#	dgdt = (((p_dyn / beta) * L_D_ratio) +
#		((np.cos(gamma)) * (g - ((V**2) / (R + h))))) / V
	dgdt = ((g * np.cos(gamma)) / V) - ((p_dyn * L_D_ratio) / (V * beta)) - \
		((V * np.cos(gamma)) / (h + R))
	return dgdt

# Time
def dh_dt(gamma, V):
	dhdt = -V * np.sin(gamma)
	return dhdt

# Ground range
def dr_dt(R, gamma, h, V):
	drdt = (R * V * np.cos(gamma)) / (R + h)
	return drdt

# Gravitational acceleration variation with altitude
def grav_sphere(g_0, R, h):
	g = g_0 * ((R / (R + h))**2)
	return g

# Ballistic coefficient
def ballistic_coeff(Cd, m, A):
	beta = m / (Cd * A)
	return beta

def traj_3DOF_dh(h, y, params):
	# Function to be called by ODE solver when altitude integration of governing
	# equations is required
	V = y[0]
	gamma = y[1]
	t = y[2]
	r = y[3]

	R = params[0]
	g_0 = params[1]
	beta = params[2]
	rho = params[3]
	C_L = params[4]
	C_D = params[5]

	g = g_0 * ((R / (R + h))**2)
	#rho = atm.alt2density(h)
	p_dyn = fcl.p_dyn(rho=rho, V=V)
	dy = np.zeros(4)

	# dvdh
	dy[0] = dv_dh(g, p_dyn, beta, V, gamma)
	# dgdh
	dy[1] = dgamma_dh(gamma, g, V, R, h)
	# dtdh
	dy[2] = dt_dh(gamma, V)
	# drdh
	dy[3] = dr_dh(R, gamma, h)

	return dy

def traj_3DOF_dt(t, y, params):
	# Function to be called by ODE solver when time integration of governing
	# equations is required
	V = y[0]
	gamma = y[1]
	h = y[2]
	r = y[3]

	R = params[0]
	#g_0 = params[1]
	g = params[1]
	beta = params[2]
	rho = params[3]
	C_L = params[4]
	C_D = params[5]

	L_D_ratio = C_L / C_D
	#g = g_0 * ((R / (R + h))**2)
	#g = grav_sphere(g_0, R, h)
	#rho = atm.alt2density(h)
	p_dyn = fcl.p_dyn(rho=rho, V=V)
	dy = np.zeros(4)

	# dvdt
	dy[0] = dv_dt(g, p_dyn, beta, gamma)
	# dgdt
	dy[1] = dgamma_dt(gamma, g, V, R, h, L_D_ratio, p_dyn, beta)
	# dhdt
	dy[2] = dh_dt(gamma, V)
	# drdt
	dy[3] = dr_dt(R, gamma, h, V)

	return dy

def traj_3DOF_ascent_dt(t, y, params):
	# Function to be called by ODE solver when time integration of governing
	# equations is required
	
	### STATE
	V = y[0]
	gamma = y[1]
	h = y[2]
	r = y[3]

	### PARAMETERS
	g = params[1]
	m = params[2]
	alpha = params[3]
	F_D = params[4]
	F_L = params[5]
	F_T = params[6]
	
	dy = np.zeros(4)

	# dvdt
	dy[0] = -((g * np.sin(gamma)) / (r**2)) - \
		(F_D / m) + ((F_T * np.cos(alpha)) / m)
	# dgdt
	dy[1] = -((g * np.cos(gamma)) / (V * (r**2))) + \
		(F_L / (V * m)) + ((V * np.cos(gamma)) / r) + \
		((F_T * np.sin(alpha)) / (V * m))
	# dhdt
	dy[2] = V * np.sin(gamma)
	# drdt
	dy[3] = (V * np.cos(gamma)) / r

	return dy

def traj_3DOF_rotating_dt(t, y, params):
	"""
	Function to be called by ODE solver for simulations using a rotating
	spherical planet assumption.
	"""

	### STATE
	# r: Altitude
	# Lambda: Latitude
	# delta: Longitude
	# V: Velocity
	# gamma: Flight path angle
	# chi: Bearing
	r = y[0]
	Lambda = y[1]
	delta = y[2]
	V = y[3]
	gamma = y[4]
	chi = y[5]

	### PARAMETERS
	# R: Planet radius
	# g: Gravitational acceleration
	# F_D: Drag force
	# F_L: Lift force
	# F_D: Side force
	# F_T: Thrust force
	# m: Spacecraft mass
	# omega: Planetary rotation speed
	# alpha: pitch (thrust) angle
	# mu: yaw angle
	R = params[0]
	g = params[1]
	F_D = params[2]
	F_L = params[3]
	F_S = params[4]
	F_T = params[5]
	m = params[6]
	omega = params[7]
	alpha = params[8]
	mu = params[9]

	# Reserve space for derivatives array
	dy = np.zeros(6)

	### DERIVATIVES
	# Altitude, dr_dt
	dy[0] = V * np.sin(gamma)

	# Latitude, dLambda_dt
	dy[1] = (V * np.cos(gamma) * np.sin(chi)) / r

	# Longitude, dDelta_dt
	dy[2] = (V * np.cos(gamma) * np.cos(chi)) / (r * np.cos(Lambda))

	# Velocity, dV_dt
	dy[3] = ((F_T * np.sin(alpha)) / m) + (-g * np.sin(gamma)) + \
		(-F_D / m) + (((omega**2) * r * np.cos(Lambda)) * \
		((np.cos(Lambda) * np.sin(gamma)) - \
		(np.sin(Lambda) * np.cos(gamma) * np.sin(chi))))

	# Flight path angle, dGamma_dt
	dy[4] = (((V / r) - (g / V)) * np.cos(gamma)) + \
		((F_L * np.cos(mu)) / (m * V)) + \
		((F_T * np.sin(alpha)) / (m * V)) + \
		((F_S * np.sin(mu)) / (m * V)) + \
		(2 * omega * np.cos(chi) * np.cos(Lambda)) + \
		((((omega**2) * r * np.cos(Lambda)) / V) * \
		((np.cos(gamma) * np.cos(Lambda)) + \
		(np.sin(gamma) * np.sin(chi) * np.sin(Lambda))))

	# Bearing, dChi_dt
	dy[5] = ((F_L * np.sin(mu)) / (m * V * np.cos(gamma))) + \
		((F_S * np.cos(mu)) / (m * V * np.cos(gamma))) - \
		((V / r) * np.cos(gamma) * np.cos(chi) * np.tan(Lambda)) + \
		(2 * omega * ((np.tan(gamma) * np.sin(chi) * np.cos(Lambda)) - \
		np.sin(Lambda))) - \
		(((omega**2) * r * np.cos(chi) * np.cos(Lambda) * np.sin(Lambda)) / \
		(V * np.cos(gamma)))

	return dy

def traj_6DOF_dt(t, y, params):
	dy = None
	return dy

def orbit_xyz(t, y, params):
	"""
	Orbital dynamcis solver by Hilbert van Pelt, Australian Defense Force Academy
	Contact: HIlbert.VanPelt@student.adfa.edu.au
	"""
	Fx = params[0] #force in the x direction
	Fy = params[1] #force in the y direction
	Fz = params[2] #force in the z direction
	Ms = params[3] #mass spacecraft
	mu = params[4] #gravitational parameter  mian gravitational body

	dy = np.zeros(6)  #placeholder for derivatives

	# Acceleration in X, Y, and Z directions (respectively)
	dy[0] = Fx / Ms - (mu * y[3]) / ((y[3]**2 + y[4]**2 + y[5]**2)**(3.0 / 2.0))
	dy[1] = Fy / Ms - (mu * y[4]) / ((y[3]**2 + y[4]**2 + y[5]**2)**(3.0 / 2.0))
	dy[2] = Fz / Ms - (mu * y[5]) / ((y[3]**2 + y[4]**2 + y[5]**2)**(3.0 / 2.0))

	# Position in X, Y and Z directions (respectively)
	dy[3] = y[0]
	dy[4] = y[1]
	dy[5] = y[2]

	return dy

# Kinetic energy
def calculateKineticEnergy(m, V):
	return 0.5 * m * (V**2)

# Gravitational potential energy
def calculatePotentialEnergy(m, mu, alt, R_planet):
	r = R_planet + alt
	return -(m * (mu / r))

# Orbital energy
def calculateSpecificOrbitalEnergy(KE, PE, m, gamma):
	return ((KE * np.sin(gamma)) + PE) / m

# Gravitational parameter
def calculateGravitationalParameter(m_spacecraft, m_planet):
	return 6.67408E-11 * (m_spacecraft + m_planet)
#def calculateGravitationalParameter(h, R_planet, V):
#	return (h + R_planet) * (V**2) 

def truncate(t, i, l):
	for index, item in enumerate(l):
		t.__dict__[item] = np.delete(t.__dict__[item],
			np.arange(i, len(t.__dict__[item])), axis=0)

	if t.spacecraft.aero_coeffs_type == 'VARIABLE':
		for item in ['Cd', 'Cl', 'Cs', 'ballistic_coeff']:
			t.spacecraft.__dict__[item] = np.delete(t.spacecraft.__dict__[item],
				np.arange(i, len(t.spacecraft.__dict__[item])))

	sol_temp = t.sol
	t.sol = np.zeros([i, 4])
	t.sol = sol_temp[0:i, :]

	return None

def assign(t, t2, i, l):
	for index, item in enumerate(l):
		t.__dict__[item][i] = t2.__dict__[item]

	return None

def interpolate_event(t, h_interp, l):
	final_list = []
	for index, item in enumerate(l):
		final_list.append(spint.griddata(t.h, t.__dict__[item],
			h_interp, method='linear'))

	return final_list

def interpolate_atmosphere(t, h_interp):
	# Check for attempts to integrate atmosphere below ground level
	if h_interp < 0:
		h_interp = 0

	# Interpolate atmospheic model
	rho_interp = spint.griddata(t.atmosphere.h, t.atmosphere.rho,
			h_interp, method='linear')
	a_interp = spint.griddata(t.atmosphere.h, t.atmosphere.a,
			h_interp, method='linear')
	p_interp = spint.griddata(t.atmosphere.h, t.atmosphere.p,
			h_interp, method='linear')
	T_interp = spint.griddata(t.atmosphere.h, t.atmosphere.T,
			h_interp, method='linear')
	mu_interp = spint.griddata(t.atmosphere.h, t.atmosphere.mu,
			h_interp, method='linear')
	Cp_interp = spint.griddata(t.atmosphere.h, t.atmosphere.cp,
			h_interp, method='linear')
	Cv_interp = spint.griddata(t.atmosphere.h, t.atmosphere.cv,
			h_interp, method='linear')

	return [rho_interp, a_interp, p_interp, T_interp, mu_interp, Cp_interp,
			Cv_interp]

def error_out_of_bounds(t, i):
	# Check for atmospheric model interpolation errors
	# (OUT_OF_BOUNDS error)
	if np.isnan(t.solver_rho[i]) == True:
		t.out_of_bounds_error = True
		print('ERROR: ATMOSPHERIC INTERPOLATION OUT OF BOUNDS AT ' \
			'INDEX %i, TRY EXPANDING ALTITUDE RANGE\n=== SIMULATION' \
			'ABORTED ===' % i)

	return None

def ground_strike(t, i):
	if t.h[i] <= 0:
		t.ground_strike = True
		print('GROUND STRIKE EVENT (ALTITUDE = 0) DETECTED BETWEEN ' \
			'INDEXES %i AND %i\n=== SIMULATION ABORTED ===' % (i-1, i))
	return None

def latlon():
	return None

def atmosphere_nrl_query(h, doy=172, year=0, sec=29000, g_lat=60, g_long=120,
	lst=16, f107A=150, f107=150, ap=4, transport_model='Multi'):
	"""
	Atmosphere model using US Naval Research Laboratory Mass Spectrometer and
	Incoherent Scatter Radar atmosphere model.  Valid from 0km upwards.  Used
	primarily for satellite drag simulations.

	Order of n:
		n_He, n_O, n_N2, n_O2, n_Ar, n_H, n_N, n_total

	Order of X:
		X_He, X_O, X_N2, X_O2, X_Ar, X_H, X_N
		
	Returns:
		[rho, a, p, T, mu, mfp, n, X]
	"""

# 	gas = ct.Solution('nasa_thermo_9.xml')
	gas = ct.Solution('nasa_thermo_9.cti')
# 	gas = ct.Solution('airNASA9.cti')
	
	gas.transport_model = transport_model
	
	# Average molecular diameter of gas
	d = 4E-10

	nrl_output = nrl_head.nrlmsise_output()
	nrl_input = nrl_head.nrlmsise_input()
	flags = nrl_head.nrlmsise_flags()
	aph = nrl_head.ap_array()

	# Set magnetic values array
	for index in range(7):
		aph.a[index] = 100

	# Output in metres (as opposed to centimetres)
	flags.switches[0] = 1

	# Set other flags to TRUE (see docstring of nrlmsise_00_header.py)
	for index in range(1, 24):
		flags.switches[index] = 1

	nrl_input.doy = doy
	nrl_input.year = year
	nrl_input.sec = sec
	nrl_input.alt = h / 1000
	nrl_input.g_lat = g_lat
	nrl_input.g_long = g_long
#	nrl_input.lst = lst
	nrl_input.lst = (sec / 3600.0) + (g_long / 15.0)
	nrl_input.f107A = f107A
	nrl_input.f107 = f107
	nrl_input.ap = ap
	#nrl_input.ap_a = self.aph

	# Run NRLMSISE00 model
#	nrlmsise.gtd7(nrl_input, flags, nrl_output)
	nrlmsise.gtd7d(nrl_input, flags, nrl_output)

	# Extract density and temperature
	rho = nrl_output.d[5]
	T = nrl_output.t[1]

	# Query Cantera for gas state
	gas.TD = T, rho
	cp = gas.cp
	cv = gas.cv
	mu = gas.viscosity

	# Perfect gas constant for air (J/kgK)
	R = cp - cv

	# Ratio of specific heats (J/kgK)
	k = cp / cv

	# Pressure and speed of sound
	p = rho * T * R
	a = fcl.speed_of_sound(k, R, T)

	# Mean free path
	mfp = fcl.mean_free_path(T, p, d)

	# Number densities of gas components
	n_He = nrl_output.d[0]
	n_O = nrl_output.d[1]
	n_N2 = nrl_output.d[2]
	n_O2 = nrl_output.d[3]
	n_Ar = nrl_output.d[4]
	n_H = nrl_output.d[6]
	n_N = nrl_output.d[7]
#	n_AnomO = nrl_output.d[8]
	n_total = np.sum([n_He, n_O, n_N2, n_O2, n_Ar, n_H, n_N])

	# Generate class structure of number densities
	n = np.array([n_He, n_O, n_N2, n_O2, n_Ar, n_H, n_N, n_total])

	# Calculate mass fractions of gas components
	X_He 		= n_He / n_total
	X_O 		= n_O / n_total
	X_N2 		= n_N2 / n_total
	X_O2 		= n_O2 / n_total
	X_Ar 		= n_Ar / n_total
	X_H 		= n_H / n_total
	X_N 		= n_N / n_total

	X = np.array([X_He, X_O, X_N2, X_O2, X_Ar, X_H, X_N])

	return [rho, a, p, T, mu, mfp, n, X, k, R]

#def thermal_balance(q_in, q_out, T_init, m, Cp):
#	q_in_sum = np.cumsum(q_in)
#	T = None
#	q_net = None
#	return [T, q_net]

class placeholder:
	def __init__(self):
		pass

#class aero_coeffs(self):
#	# This function is called by the solver at every integration step and
#	# provides aerodynamic coefficients.  Constants or empirical realtions
#	# may be specified here.
#
#	self.Cd = Cd
#	self.Cl = Cl
#	self.Cs = Cs
#
#	return [Cd, Cl, Cs]

class planet:
	"""
	Class for storage of planet variables.  This class is designed for use
	with simulations where the planet variables may be assumed constant
	throughout. 

	Values for EARTH :
		Mass = 5.972E24 kg
		Mean radius = 6378136.0 m
		g_0 = 9.80665
	"""
	def __init__(self, name, mass, R, g_0):
		self.name = name
		self.m = mass
		self.R = R	
		self.g_0 = g_0
		return None

class spacecraft:
	"""
	Class for storage of spacecraft variables.  This class is designed for use
	with simulations where the spacecraft variables may be assumed constant
	throughout.
	"""
	def __init__(self, Cd, m, A, R_n, L, Cl=0, Cs=0):
		self.aero_coeffs_type = 'CONSTANT'
		self.A = A
		self.Cd = Cd
		self.Cl = Cl
		self.Cs = Cs
		self.R_n = R_n
		self.m = m
		self.L = L
		self.ballistic_coeff = (self.m) / (self.Cd * self.A)

		return None

class spacecraft_var:
	"""
	Class for storage of spacecraft variables.  This class is designed for use
	with simulations where the spacecraft aerodynamic coefficients are variable.
	A function (aero_dat) must be supplied which returns drag, lift, and lateral
	force coefficients as a function of Mach, Reynolds and Knudsen numbers.

	(Note that arguments and returns for aero_dat are given in the required
	order in the paragraph above.)

	This class must be initialised with the same number of integration steps as
	the trajectory class with which it is to be used.
	"""
	def __init__(self, aero_dat, Cd_init, Cl_init, Cs_init, m, A, R_n, L, steps):
		self.aero_coeffs_type = 'VARIABLE'

		# Store number of integration steps
		# NB: This should be the same as the trajectory instance used to run
		# calculations.
		self.steps = np.int(steps)

		# Store uder-defined function for recalculating aero coefficients
		self.aero_dat = aero_dat
		#self.Cd, self.Cl, self.Cs = self.aero_dat(self.Re, self.Ma, self.Kn)

		# Store spacecraft constants
		self.A = A
		self.R_n = R_n
		self.m = m
		self.L = L

		# Generate storage stuctures for aero coefficients
		self.Cd = np.zeros(self.steps)
		self.Cl = np.zeros(self.steps)
		self.Cs = np.zeros(self.steps)
		self.ballistic_coeff = np.zeros(self.steps)

		# Assign initial values for aero coefficients
		self.Cd_init = Cd_init
		self.Cl_init = Cl_init
		self.Cs_init = Cs_init
		self.ballistic_coeff_init = ballistic_coeff(self.Cd_init, self.m, self.A)
		self.Cd[0] = Cd_init
		self.Cl[0] = Cl_init
		self.Cs[0] = Cs_init
		self.ballistic_coeff[0] = self.ballistic_coeff_init

		return None

	def update_aero(self, index, Re, Ma, Kn, p_inf, q_inf, rho_inf, gamma_var,
		Cd_prev, Cl_prev):
		self.Cd[index], self.Cl[index], self.Cs[index] = \
			self.aero_dat(Re, Ma, Kn, p_inf, q_inf, rho_inf, gamma_var,
			Cd_prev, Cl_prev)
		self.ballistic_coeff[index] = ballistic_coeff(self.Cd[index], self.m, self.A)

		return None

class launcher_var:
	"""
	Class for storage of launcher variables.  This class is designed for use
	with simulations where the spacecraft aerodynamic coefficients are variable.
	A function (aero_dat) must be supplied which returns drag, lift, and lateral
	force coefficients as a function of Mach, Reynolds and Knudsen numbers.

	(Note that arguments and returns for aero_dat are given in the required
	order in the paragraph above.)

	This class must be initialised with the same number of integration steps as
	the trajectory class with which it is to be used.
	"""
	def __init__(self, aero_dat, thrust_dat, control_dat, Cd_init, Cl_init, 
		Cs_init, m_init, A, R_n, L, steps, Isp):
		
		self.aero_coeffs_type = 'VARIABLE'

		# Store number of integration steps
		# NB: This should be the same as the trajectory instance used to run
		# calculations.
		self.steps = np.int(steps)

		# Store uder-defined function for recalculating aero coefficients,
		# engine thrust, and thrust vector angle
		self.aero_dat = aero_dat
		self.thrust_dat = thrust_dat
		self.control_dat = control_dat

		# Store spacecraft constants
		self.A = A
		self.R_n = R_n
		self.m_init = m_init
		self.L = L

		# Generate storage stuctures for aero coefficients, forces, and mass
		self.Cd = np.zeros(self.steps)
		self.Cl = np.zeros(self.steps)
		self.Cs = np.zeros(self.steps)
		self.ballistic_coeff = np.zeros(self.steps)
		self.m = np.zeros(self.steps)
		self.F_D = np.zeros(self.steps)
		self.F_L = np.zeros(self.steps)
		self.F_T = np.zeros(self.steps)
		
		# Assign initial values for aero coefficients, forces, and mass
		self.Cd_init = Cd_init
		self.Cl_init = Cl_init
		self.Cs_init = Cs_init
		self.ballistic_coeff_init = ballistic_coeff(self.Cd_init, self.m, self.A)
		self.Cd[0] = Cd_init
		self.Cl[0] = Cl_init
		self.Cs[0] = Cs_init
		self.ballistic_coeff[0] = self.ballistic_coeff_init
		self.m[0] = self.m_init
		self.F_D[0] = fcl.aero_force(self.solver_rho[0], self.V[0], \
			self.spacecraft.Cd[0], self.spacecraft.A)
		self.F_L[0] = fcl.aero_force(self.solver_rho[0], self.V[0], \
			self.spacecraft.Cl[0], self.spacecraft.A)
		self.F_T[0] = self.spacecraft.thrust_dat(0)

		return None

	def update_aero(self, index, Re, Ma, Kn, p_inf, q_inf, rho_inf, gamma_var,
		Cd_prev, Cl_prev):

		self.Cd[index], self.Cl[index], self.Cs[index] = \
			self.aero_dat(Re, Ma, Kn, p_inf, q_inf, rho_inf, gamma_var,
			Cd_prev, Cl_prev)

		self.ballistic_coeff[index] = ballistic_coeff(self.Cd[index], \
			self.m, self.A)
		
		return None
		
	def update_thrust(self, index, t):
		self.F_T[index] = self.thrust_dat(t)
		return None

	def update_mass(self, index):
		self.m[index] -= self.F_T[index] / self.Isp
		return None

class atmosphere_us76:
	"""
	Atmosphere model using US Standard Atmosphere 1976 model.  Valid from 0km
	to 86km altitude.
	"""
	def __init__(self, h):
		# Cantera Solution object
		self.gas = ct.Solution('air.xml')

		# Discretised altitude steps
		self.h = h

		# Average molecular diameter of gas
		self.d = 4E-10

		self.steps = len(h)

		self.rho = np.zeros(self.steps)
		self.p = np.zeros(self.steps)
		self.T = np.zeros(self.steps)
		self.a = np.zeros(self.steps)
		self.k = np.zeros(self.steps)
		self.mu = np.zeros(self.steps)

		for index, alt in enumerate(self.h):
			self.rho[index] = atm.alt2density(alt, alt_units='m', density_units='kg/m**3')
			self.p[index] = atm.alt2press(alt, press_units='pa', alt_units='m')
			self.T[index] = atm.alt2temp(alt, alt_units='m', temp_units='K')
			self.a[index] = atm.temp2speed_of_sound(self.T[index], temp_units='K', speed_units='m/s')

		for index, alt in enumerate(self.h):
			self.gas.TP = self.T[index], self.p[index]
			self.k[index] = self.gas.cp / self.gas.cv
			self.mu[index] = self.gas.viscosity

		print('ATMOSPHERIC MODEL COMPUTED (US76)')

		return None

class atmosphere_j77:
	def __init__(self, h, T_thermosphere):
		# Cantera Solution object
		self.gas = ct.Solution('air.xml')

		# Discretised altitude steps
		self.h = h

		# Average molecular diameter of gas
		self.d = 4E-10

		# Ratio of specific heats
		self.steps = len(h)

		self.rho = np.zeros(self.steps)
		self.p = np.zeros(self.steps)
		self.T = np.zeros(self.steps)
		self.a = np.zeros(self.steps)
		self.k = np.zeros(self.steps)
		self.mu = np.zeros(self.steps)

		# Call Jacchia77 model
		data = j77.j77sri(np.max(h), T_thermosphere)
		data_np = np.array(data)

		h_int = spint.griddata
		T_int = spint.griddata
		mw_int = spint.griddata
		n = spint.griddata

		for index, alt in enumerate(self.h):
			self.rho[index] = atm.alt2density(alt, alt_units='m', density_units='kg/m**3')
			self.p[index] = atm.alt2press(alt, press_units='pa', alt_units='m')
			self.T[index] = atm.alt2temp(alt, alt_units='m', temp_units='K')
			self.a[index] = atm.temp2speed_of_sound(self.T[index], temp_units='K', speed_units='m/s')

		for index, alt in enumerate(self.h):
			self.gas.TP = self.T[index], self.p[index]
			self.k[index] = self.gas.cp / self.gas.cv
			self.mu[index] = self.gas.viscosity

		return None

class atmosphere_nrl:
	"""
	Atmosphere model using US Naval Research Laboratory Mass Spectrometer and
	Incoherent Scatter Radar atmosphere model.  Valid from 0km upwards.  Used
	primarily for satellite drag simulations.
	"""
	def __init__(self, h, doy=172, year=0, sec=29000, g_lat=60, g_long=120,
		lst=16, f107A=150, f107=150, ap=4, console_output=True, 
		transport_model='Mix'):

		# Cantera Solution object
		self.gas = ct.Solution('nasa_thermo_9.cti')
# 		self.gas = ct.Solution('nasa_thermo_9.xml')
# 		self.gas = ct.Solution('air.cti')
# 		self.gas = ct.Solution('airNASA9.cti')

		# Set transport model to multi component (as opposed to mixture-averaged)
		# Sacrifices speed for more accurate low temperature results
#		self.gas.transport_model = 'Multi'
		self.gas.transport_model = transport_model

		# Discretised altitude steps
		self.h = h

		# Average molecular diameter of gas
		self.d = 4E-10

		self.steps = len(h)

		self.output = [nrl_head.nrlmsise_output() for _ in range(self.steps)]
		self.input = [nrl_head.nrlmsise_input() for _ in range(self.steps)]
		self.flags = nrl_head.nrlmsise_flags()
		self.aph = nrl_head.ap_array()

		# Set magnetic values array
		for index in range(7):
			self.aph.a[index] = 100

		# Output in metres (as opposed to centimetres)
		self.flags.switches[0] = 1

		# Set other flags to TRUE (see docstring of nrlmsise_00_header.py)
		for index in range(1, 24):
			self.flags.switches[index] = 1

		for index in range(self.steps):
			self.input[index].doy = doy
			self.input[index].year = year
			self.input[index].sec = sec
			self.input[index].alt = self.h[index] / 1000
			self.input[index].g_lat = g_lat
			self.input[index].g_long = g_long
#			self.input[index].lst = lst
			self.input[index].lst = (sec / 3600.0) + (g_long / 15.0)
			self.input[index].f107A = f107A
			self.input[index].f107 = f107
			self.input[index].ap = ap
			#self.input[index].ap_a = self.aph

		# Run NRLMSISE00 model
		for index in range(self.steps):
#			nrlmsise.gtd7(self.input[index], self.flags, self.output[index])
			nrlmsise.gtd7d(self.input[index], self.flags, self.output[index])

		# Pre-allocate memory
		self.rho = np.zeros(self.steps)
		self.T = np.zeros(self.steps)
		self.T_exo = np.zeros(self.steps)
		self.a = np.zeros(self.steps)
		self.k = np.zeros(self.steps)
		self.mu = np.zeros(self.steps)
		self.cp = np.zeros(self.steps)
		self.cv = np.zeros(self.steps)
		self.n = np.zeros(self.steps)
		self.n_He = np.zeros(self.steps)
		self.n_O = np.zeros(self.steps)
		self.n_N2 = np.zeros(self.steps)
		self.n_O2 = np.zeros(self.steps)
		self.n_Ar = np.zeros(self.steps)
		self.n_H = np.zeros(self.steps)
		self.n_N = np.zeros(self.steps)
		self.n_AnomO = np.zeros(self.steps)
		self.X = np.zeros([self.steps, 10])

		# Extract density and temperature
		for index in range(self.steps):
			self.rho[index] = self.output[index].d[5]
			self.T[index] = self.output[index].t[1]
			self.T_exo[index] = self.output[index].t[0]
			self.n_He[index] = self.output[index].d[0]
			self.n_O[index] = self.output[index].d[1]
			self.n_N2[index] = self.output[index].d[2]
			self.n_O2[index] = self.output[index].d[3]
			self.n_Ar[index] = self.output[index].d[4]
			self.n_H[index] = self.output[index].d[6]
			self.n_N[index] = self.output[index].d[7]
			self.n_AnomO[index] = self.output[index].d[8]

			# Sum only number densities of species modelled by Cantera 'air.xml' object
#			self.n[index] = np.sum([self.n_He[index], self.n_O[index], \
#				self.n_N2[index], self.n_O2[index], self.n_Ar[index], \
#				self.n_H[index], self.n_N[index]])

			# Sum only number densitites initially modelled in 11-species air model
#			self.n[index] = np.sum([self.n_O[index], self.n_N2[index], \
#				self.n_O2[index], self.n_N[index]])

			# Sum full number density
			self.n[index] = np.sum([self.n_He[index], self.n_O[index], \
				self.n_N2[index], self.n_O2[index], self.n_Ar[index], \
				self.n_H[index], self.n_N[index], self.n_AnomO[index]])

			self.X_names = ['O', 'O2', 'N2', 'N', 'H', 'He', 'Ar']

			self.X[index, 0] = self.n_O[index] / self.n[index]
			self.X[index, 1] = self.n_O2[index] / self.n[index]
			self.X[index, 2] = self.n_N2[index] / self.n[index]
			self.X[index, 3] = self.n_N[index] / self.n[index]
			self.X[index, 4] = self.n_H[index] / self.n[index]
			self.X[index, 5] = self.n_He[index] / self.n[index]
			self.X[index, 6] = self.n_Ar[index] / self.n[index]
#			self.X[index, 7] = self.n_AnomO[index] / self.n[index]

#			self.X[index, 0] = self.n_O[index] / self.n[index]
#			self.X[index, 1] = self.n_O2[index] / self.n[index]
#			self.X[index, 2] = self.n_N[index] / self.n[index]
#			self.X[index, 6] = self.n_N2[index] / self.n[index]
#			self.X[index, 7] = self.n_Ar[index] / self.n[index]

		# Query Cantera for gas state
#		for index, alt in enumerate(self.h):
#			self.gas.TD = self.T[index], self.rho[index]
#			self.cp[index] = self.gas.cp
#			self.cv[index] = self.gas.cv
#			self.mu[index] = self.gas.viscosity

		for index, alt in enumerate(self.h):
			self.gas.X = self.X[index, :]
			self.gas.TD = self.T[index], self.rho[index]
			self.cp[index] = self.gas.cp
			self.cv[index] = self.gas.cv
			self.mu[index] = self.gas.viscosity

		# Perfect gas constant for air (J/kgK)
		self.R = self.cp - self.cv

		# Ratio of specific heats (J/kgK)
		self.k = self.cp / self.cv

		# Pressure and speed of sound
		self.p = self.rho * self.T * self.R
		self.a = fcl.speed_of_sound(self.k, self.R, self.T)

		# Mean free path
		self.mfp = fcl.mean_free_path(self.T, self.p)

		self.l = ['p', 'a', 'k', 'R', 'cp', 'cv', 'mu', 'rho', 'T', 'T_exo',
			'n_He', 'n_O', 'n_N2', 'n_O2', 'n_Ar', 'n_H', 'n_N', 'n', 'X',
			'n_AnomO', 'd', 'h']

		if console_output == True:
			print('ATMOSPHERIC MODEL COMPUTED (NRLMSISE00)')
		return None

class trajectory_ballistic:
	"""
	Ballistic trajectory calculator.  No lifting forces are considered in this
	model, only drag and gravity.  The integration step therefore is altitude,
	and all diffrenetial equations are formulated wrt. h (altitude).
	"""
	def __init__(self, vehicle, atmosphere, gamma_init, V_init, g_0, R):
		# NB: vehicle should be an instance of the class 'spacecraft'

		# Import atmospheric model
		self.atmosphere = atmosphere #atmosphere(self.h)

		# Copy altitude array for convenience
		self.h = self.atmosphere.h
		self.steps = self.atmosphere.steps

		# Import spacecraft entering atmosphere
		self.spacecraft = vehicle

		# Set astronomical constants
		self.R = R
		self.g_0 = g_0

		# Set initial values
		self.gamma_init = gamma_init
		self.V_init = V_init
		self.h_init = self.h[0]

		# Define integration points in h
		#self.h = h #np.linspace(h_init, h_end, steps)
		self.del_h = np.abs(self.h[1] - self.h[0])

		# Calculate variance in gravitational acceleration using inverse
		# square law
		self.g = grav_sphere(self.g_0, self.R, self.h)

		# Pre-allocate memory for iterative trajectory calculations
		self.V 		= np.zeros(self.steps)
		self.gamma 	= np.zeros(self.steps)
		self.t 		= np.zeros(self.steps)
		self.r 		= np.zeros(self.steps)
		self.dvdh 	= np.zeros(self.steps)
		self.dgdh	= np.zeros(self.steps)
		self.dtdh  	= np.zeros(self.steps)
		self.drdh 	= np.zeros(self.steps)
		self.p_dyn	= np.zeros(self.steps)
		self.Ma  	= np.zeros(self.steps)

		return None

	def initialise(self):
		self.V[0] = self.V_init
		self.gamma[0] = self.gamma_init
		self.p_dyn[0] = fcl.p_dyn(rho=self.atmosphere.rho[0], V=self.V[0])
		self.Ma[0] = self.V[0] / self.atmosphere.a[0]
		self.dvdh[0] = dv_dh(self.g[0], self.p_dyn[0], \
			self.spacecraft.ballistic_coeff, self.V[0], self.gamma[0])
		self.dgdh[0] = dgamma_dh(self.gamma[0], self.g[0], self.V[0], self.R, self.h[0])
		self.dtdh[0] = dt_dh(self.gamma[0], self.V[0])
		self.drdh[0] = dr_dh(self.R, self.gamma[0], self.h[0])

		return None

	def simulate_euler(self):
		"""
		Run trajectory calculations using forward Euler method
		"""
		for n in range(1, self.steps):
			# Set values for current step
			self.V[n] 	 	= self.V[n-1] + (self.dvdh[n-1] * self.del_h)
			self.gamma[n]		= self.gamma[n-1] + (self.dgdh[n-1] * self.del_h)
			self.t[n] 	 	= self.t[n-1] + (self.dtdh[n-1] * self.del_h)
			self.r[n] 		= self.r[n-1] + (self.drdh[n-1] * self.del_h)

			# Update dynamic pressure and Mach number for current step
			self.p_dyn[n] = fcl.p_dyn(rho=self.atmosphere.rho[n], V=self.V[n])
			self.Ma[n] = self.V[n] / self.atmosphere.a[n]

			# Update rates of change for current step
			self.dvdh[n] = dv_dh(self.g[n], self.p_dyn[n], \
				self.spacecraft.ballistic_coeff, self.V[n], self.gamma[n])
			self.dgdh[n] = dgamma_dh(self.gamma[n], self.g[n], self.V[n], self.R, self.h[n])
			self.dtdh[n] = dt_dh(self.gamma[n], self.V[n])
			self.drdh[n] = dr_dh(self.R, self.gamma[n], self.h[n])

		self.post_calc()

		print('TRAJECTORY COMPUTED (FWD. EULER)')

	def simulate_dopri(self):
		"""
		Run trajectory calcualtions using explicit Runge-Kutta method of order
		4(5) from Dormand & Prince
		"""
		# Create ODE object from SciPy using Dormand-Prince RK solver
		eq = integrate.ode(traj_3DOF_dh).set_integrator('dop853', nsteps=1E8,
			rtol=1E-10)

		# Set initial conditions
		y_init = [self.V_init, self.gamma_init, self.t[0], self.r[0]]
		eq.set_initial_value(y_init, t=self.h_init)

		# Create empty arrays for storage of results from ODE solver
		sol = np.zeros([self.steps, 4])
		h_input = np.zeros(self.steps)
		y_input = np.zeros([self.steps, 4])
		p_dyn = np.zeros(self.steps)

		# Solve ODE system over altitude range
		for index, val in enumerate(self.h):
			# Update parameters with atmospheric density at each altitude step
			params = [self.R, self.g_0, self.spacecraft.ballistic_coeff,
				self.atmosphere.rho[index], self.spacecraft.Cl, self.spacecraft.Cd]
			eq.set_f_params(params)

			# Solve ODE system
			sol[index, :] = eq.integrate(val)

			# Calculate dynamic pressure iteration results
			p_dyn[index] = fcl.p_dyn(rho=params[3], V=sol[index, 0])

			# Save inputs for inspection
			h_input[index] = eq.t
			y_input[index, :] = eq.y

		# Calculate Mach numbers
		Ma = sol[:, 0] / self.atmosphere.a

		# Copy ODE input and solution arrays to structures in trajectory object
		self.V = sol[:, 0]
		self.gamma = sol[:, 1]
		self.t = sol[:, 2]
		self.r = sol[:, 3]
		self.p_dyn = p_dyn
		self.Ma = Ma

		self.post_calc()

		print('TRAJECTORY COMPUTED (RK 4/5)')

		return [sol, h_input, y_input, p_dyn, Ma]

	def calculate_heating(self):
		# Generate new placeholder class for all heat fluxes
		self.qdot 	= placeholder()

		# Generate blank classes for different heat flux mechanisms
		self.qdot.conv 	= placeholder()
		self.qdot.rad 	= placeholder()

		# Generate empty arrays for different correlations
		self.qdot.conv.dh 	= np.zeros(self.steps)
		self.qdot.conv.bj 	= np.zeros(self.steps)
		self.qdot.conv.s 		= np.zeros(self.steps)
		self.qdot.conv.fr 	= np.zeros(self.steps)
		self.qdot.conv.sg 	= np.zeros(self.steps)
		self.qdot.rad.bj 		= np.zeros(self.steps)
		self.qdot.rad.s 		= np.zeros(self.steps)
		self.qdot.rad.ts 		= np.zeros(self.steps)

		# Call heat_flux_lib for actual calculations
		# Detra-Hidalgo
		self.qdot.conv.dh = hcl.detra_hidalgo(self.V, self.atmosphere.rho,
			self.spacecraft.R_n)

		# Brandis-Johnston (convective)
		for index, val in enumerate(self.h):
			self.qdot.conv.bj[index] = hcl.brandis_johnston(self.V[index],
				self.atmosphere.rho[index], self.spacecraft.R_n, mode='conv')

		# Smith (convective)
		self.qdot.conv.s = hcl.smith(self.V, self.atmosphere.rho,
			self.spacecraft.R_n, mode='conv', planet='Earth')

		# Fay-Riddell
		#self.qdot.conv.fr = hcl.fay_riddell()

		# Sutton-Graves (convective)
		self.qdot.conv.sg	= hcl.sutton_graves(self.V, self.atmosphere.rho,
			self.spacecraft.R_n)

		# Brandis-Johnston (radiative)
		for index, val in enumerate(self.h):
			self.qdot.rad.bj[index] = hcl.brandis_johnston(self.V[index],
				self.atmosphere.rho[index], self.spacecraft.R_n, mode='rad')

		# Smith (radiative)
		self.qdot.rad.bj = hcl.smith(self.V, self.atmosphere.rho,
			self.spacecraft.R_n, mode='rad', planet='Earth')

		# Tauber-Sutton (radiative)
		for index, val in enumerate(self.h):
			self.qdot.rad.ts[index] = hcl.tauber_sutton(self.V[index],
				self.atmosphere.rho[index], self.spacecraft.R_n)

		# Net flux (convective heating, radiative cooling)
		self.qdot.net = placeholder()
		self.qdot.net.bj = self.qdot.conv.bj + self.qdot.rad.bj
		self.qdot.net.s = self.qdot.conv.s + self.qdot.rad.s
		self.qdot.net.sg_ts = self.qdot.conv.sg + self.qdot.rad.ts

	def plot_trajectory(self):

		fig = plt.figure(figsize=[12, 10])
		ax1 = fig.add_subplot(224)
		ax2 = fig.add_subplot(223)
		ax4 = fig.add_subplot(211)

		#alt = self.h / 1000
		#g_range = self.r / 1000
		#gamma_deg = np.rad2deg(self.gamma)
		line_width = 2.0
		num = len(self.h) / 20

		quiv = np.array([self.r[0:-1:num]/1000, self.h[0:-1:num]/1000, \
			self.gamma[0:-1:num]]).T

		quiver_x = np.log(self.V[0:-1:num]) * np.cos(quiv[:, 2])
		quiver_y = -np.log(self.V[0:-1:num]) * np.sin(quiv[:, 2])

		ax4.plot((self.r/1000), (self.h/1000), color='c', linewidth=line_width, alpha=1.0)
		ax4.scatter(quiv[:, 0], quiv[:, 1], color='b', s=15, alpha=0.25, zorder=3)
		ax4.scatter(self.r[-1]/1000, self.h[-1]/1000, s=300, marker='x',
			color='r', alpha=0.3, zorder=3)
		quiv_plot = ax4.quiver(quiv[:, 0], quiv[:, 1], quiver_x, quiver_y, \
			scale=40, linewidth=1, edgecolor='b', headwidth=10, headlength=10,\
			width=0.001, alpha=0.2)
		ax4.quiverkey(quiv_plot, 0.9, 0.9, 5, 'log(Velocity), $log(V)$ [m/s]', color='b')
		ax4.set_xlabel('Ground range, r (km)')
		ax4.set_ylabel('Altitude, h (km)')
		ax4.grid(True)

		#ax0.plot(g_range, alt, color='g', linewidth=line_width)
		#ax0.set_xlabel('Ground range, r (m)')
		#ax0.set_ylabel('Altitude, h (km)')
		#ax0.grid(True)

		ax1.plot(self.Ma, self.h/1000, color='r', linewidth=line_width)
		ax1.scatter(self.Ma[-1], self.h[-1]/1000, s=300, marker='x',
			color='r', alpha=0.5, zorder=3)
		ax1.set_xlabel('Mach number, Ma')
		ax1.set_ylabel('Altitude, h (km)')
		ax1.set_xscale('log')
		ax1.grid(True)

		ax2.plot(self.V, self.h/1000, color='b', linewidth=line_width)
		ax2.scatter(self.V[-1], self.h[-1]/1000, s=300, marker='x',
			color='r', alpha=0.5, zorder=3)
		ax2.set_xlabel('Velocity, V (m/s)')
		ax2.set_ylabel('Altitude, h (km)')
		ax2.set_xscale('log')
		ax2.grid(True)

		#ax3.plot(gamma_deg, alt, color='g', linewidth=line_width)
		#ax3.set_xlabel(r'Flight path angle, $\gamma$ (degrees)')
		#ax3.set_ylabel('Altitude, h (km)')
		#ax3.grid(True)

		plt.tight_layout()

		return None

	def show_regimes(self):

		handle = plt.gcf()
		xspan = np.abs(handle.axes[-1].get_xlim()[0]) + \
			np.abs(handle.axes[-1].get_xlim()[1])
		xlim = handle.axes[-1].get_xlim()[0] + (0.02 * xspan)

		# REGIME HIGHLIGHTS
		reg_col = [[0, 1, 0],
						[0, 0.66, 0.33],
						[0, 0.33, 0.66],
						[0, 0, 1]]

		# Continuum
		if hasattr(self.regimes, 'continuum'):
			plt.axhspan(self.h[self.regimes.continuum[0]]/1000,
				self.h[self.regimes.continuum[-1]]/1000,
				facecolor=reg_col[0], alpha=0.05)
			plt.text(xlim, self.h[self.regimes.continuum[-1]]/1000 + 1,
				'CONTINUUM', fontsize=12, \
			    color=reg_col[0], alpha=0.5)

		# Slip
		if hasattr(self.regimes, 'slip'):
			plt.axhspan(self.h[self.regimes.slip[0]]/1000,
				self.h[self.regimes.slip[-1]]/1000,
				facecolor=reg_col[1], alpha=0.05)
			plt.text(xlim, self.h[self.regimes.slip[-1]]/1000 + 1,
				'SLIP', fontsize=12, \
			    color=reg_col[1], alpha=0.5)

		# Transition
		if hasattr(self.regimes, 'transition'):
			plt.axhspan(self.h[self.regimes.transition[0]]/1000,
				self.h[self.regimes.transition[-1]]/1000,
				facecolor=reg_col[2], alpha=0.05)
			plt.text(xlim, self.h[self.regimes.transition[-1]]/1000 + 1,
				'TRANSITION', fontsize=12, \
			    color=reg_col[2], alpha=0.5)

		# Free molecular
		if hasattr(self.regimes, 'free_molecular'):
			plt.axhspan(self.h[self.regimes.free_molecular[0]]/1000,
				self.h[self.regimes.free_molecular[-1]]/1000,
				facecolor=reg_col[3], alpha=0.05)
			plt.text(xlim, self.h[self.regimes.free_molecular[-1]]/1000 + 1,
				'FREE MOLECULAR', fontsize=12, \
			    color=reg_col[3], alpha=0.5)

		# REGIME BOUNDARY LINES
		# Continuum-Slip
		if (self.index_cont_slip != None):
			plt.axhline(y=self.h[self.index_cont_slip]/1000,
				color=reg_col[1], alpha=0.2, zorder=0, linewidth=2.0)

		# Slip-Transition
		if (self.index_slip_tran != None):
			plt.axhline(y=self.h[self.index_slip_tran]/1000,
				color=reg_col[2], alpha=0.2, zorder=0, linewidth=2.0)

		# Transition-Free molecular
		if (self.index_tran_freemol != None):
			plt.axhline(y=self.h[self.index_tran_freemol]/1000,
				color=reg_col[3], alpha=0.2, zorder=0, linewidth=2.0)

		return None

	def post_calc(self):
		"""
		Perform post-integration calculations (such as Knudsen and Reynolds
		numbers, location of regimes, etc.)
		"""

		self.mfp = fcl.mean_free_path(self.atmosphere.T, self.atmosphere.p,
			self.atmosphere.d)
		self.Kn = self.mfp / self.spacecraft.L
#		self.Re = fcl.KnReMa(self.atmosphere.k, Kn=self.Kn,
#			Ma=self.Ma)
		self.Re = fcl.Reynolds(self.atmosphere.rho, self.V, self.spacecraft.L,
			self.atmosphere.mu)

		# Continuum: 0 < Kn < 0.1
		# Slip: 0.1 <= Kn < 1.0
		# Transition: 1.0 <= Kn < 10
		# Free molecular: 10 < Kn

		self.regimes = placeholder()

		if len(np.argwhere(self.Kn > 10)) != 0:
			self.index_tran_freemol = np.argwhere(self.Kn > 10)[-1]
			self.regimes.free_molecular = np.argwhere(self.Kn >= 10)
		else:
			self.index_tran_freemol = None

		if len(np.argwhere(self.Kn > 1.0)) != 0:
			self.index_slip_tran = np.argwhere(self.Kn > 1.0)[-1]
			self.regimes.transition = np.argwhere((self.Kn < 10) & (self.Kn >= 1.0))
		else:
			self.index_slip_tran = None

		if len(np.argwhere(self.Kn > 0.1)) != 0:
			self.index_cont_slip = np.argwhere(self.Kn > 0.1)[-1]
			self.regimes.slip = np.argwhere((self.Kn < 1.0) & (self.Kn >= 0.1))
		else:
			self.index_cont_slip = None

		if len(np.argwhere((self.Kn > 0) & (self.Kn <= 0.1))) != 0:
			self.regimes.continuum = np.argwhere((self.Kn < 0.1) & (self.Kn >= 0))
		else:
			self.index_cont_slip = None

		return [self.mfp, self.Kn, self.Re]

class trajectory_lifting:
	"""
	Lifting trajectory calculator.  Lift, drag and gravitational forces are
	considered in this model.  The integration variable for differential
	equations is time.  Atmospheric models are queried on a step-by-step
	basis to update variables.
	"""
	def __init__(self, vehicle, atmosphere, planet, gamma_init, V_init, 
		h_init, h_final, steps, console_output=True, console_interval=100):
		# NB: vehicle should be an instance of the class 'spacecraft'

		# Verbose solver output flag
		self.console_output = console_output
		
		# Number of console outputs (# of steps / console_interval)
		self.console_interval = console_interval

		# Import atmospheric model
		self.atmosphere = atmosphere #atmosphere(self.h)

		# Copy altitude array for convenience
		#self.h = self.atmosphere.h
		self.steps_atm = self.atmosphere.steps
		self.steps_storage = np.int(steps)#self.steps * num

		# Import spacecraft entering atmosphere
		self.spacecraft = vehicle
		
		# Import planet properties
		self.planet = planet

		# Set astronomical constants
		self.R = self.planet.R
		self.g_0 = self.planet.g_0

		# Set initial values
		self.gamma_init = gamma_init
		self.V_init = V_init
		self.h_init = h_init #self.atmosphere.h[0]
		self.h_final = h_final #self.atmosphere.h[-1]

		# Define integration points in h
		#self.h = h #np.linspace(h_init, h_end, steps)
		#self.del_h = np.abs(self.h[1] - self.h[0])

		# Calculate variance in gravitational acceleration using inverse
		# square law
		#self.g = grav_sphere(self.g_0, self.R, self.h)

		# Pre-allocate memory for iterative trajectory calculations
		self.V 		= np.zeros(self.steps_storage)
		self.gamma 	= np.zeros(self.steps_storage)
#		self.time	= np.zeros(self.steps_storage)
		self.r 		= np.zeros(self.steps_storage)
#		self.dvdt 	= np.zeros(self.steps_storage)
#		self.dgdt	= np.zeros(self.steps_storage)
#		self.dhdt  	= np.zeros(self.steps_storage)
#		self.drdt 	= np.zeros(self.steps_storage)
		self.p_dyn	= np.zeros(self.steps_storage)
		self.Ma  	= np.zeros(self.steps_storage)
		self.Kn  	= np.zeros(self.steps_storage)
		self.Re  	= np.zeros(self.steps_storage)
		self.mfp  	= np.zeros(self.steps_storage)
		self.h 	 	= np.zeros(self.steps_storage)
		self.g 	 	= np.zeros(self.steps_storage)

		self.energyKinetic	= np.zeros(self.steps_storage)
		self.energyPotential 	= np.zeros(self.steps_storage)
		self.energyOrbitalSpecific	= np.zeros(self.steps_storage)

		# Create empty arrays for storage of results from ODE solver
		self.sol 		= np.zeros([self.steps_storage, 4])
		self.solver_time = np.zeros(self.steps_storage)
		self.solver_rho 	= np.zeros(self.steps_storage)
		self.solver_mu 	= np.zeros(self.steps_storage)
		self.solver_a 	= np.zeros(self.steps_storage)
		self.solver_p 	= np.zeros(self.steps_storage)
		self.solver_T 	= np.zeros(self.steps_storage)
		self.solver_Cp	= np.zeros(self.steps_storage)
		self.solver_Cv 	= np.zeros(self.steps_storage)
		self.y_input 	= np.zeros([self.steps_storage, 4])

		# Define list of keys for obejct dict (self.__dict__)
		# To be used by truncation, event interpolation, and variable
		# assignment functions
		self.l = ['V', 'p_dyn', 'g', 'gamma', 'Ma', 'Kn', 'Re', 'h', 'r',
			'solver_time', 'solver_rho', 'solver_p', 'solver_T', 'solver_mu',
			'solver_a', 'mfp', 'solver_Cp', 'solver_Cv', 'energyKinetic',
			'energyPotential', 'energyOrbitalSpecific']

		# Set up simulation termination flags
		self.out_of_bounds_error = False

		return None

	def initialise(self):
		self.h[0] = self.h_init
		self.V[0] = self.V_init
		self.gamma[0] = self.gamma_init
		self.g[0] = grav_sphere(self.g_0, self.R, self.h_init)

		self.solver_rho[0], self.solver_a[0], self.solver_p[0], \
			self.solver_T[0], self.solver_mu[0], self.solver_Cp[0], \
			self.solver_Cv[0] = \
			interpolate_atmosphere(self, self.h_init)

		self.p_dyn[0] = fcl.p_dyn(rho=self.solver_rho[0], V=self.V[0])
		self.Ma[0] = self.V[0] / self.solver_a[0]
		self.mfp[0] = fcl.mean_free_path(self.solver_T[0], self.solver_p[0],
			self.atmosphere.d)
		self.Kn[0] = self.mfp[0] / self.spacecraft.L
		self.Re[0] = fcl.Reynolds(self.solver_rho[0], self.V[0],
			self.spacecraft.L, self.solver_mu[0])

		self.mu = calculateGravitationalParameter(self.spacecraft.m, \
			self.planet.m)
		
		self.energyKinetic[0] = calculateKineticEnergy(self.spacecraft.m, 
			self.V_init)
		self.energyPotential[0] = calculatePotentialEnergy(self.spacecraft.m, 
			self.mu, self.h[0], self.planet.R)
		self.energyOrbitalSpecific[0] = calculateSpecificOrbitalEnergy(\
			self.energyKinetic[0], self.energyPotential[0], self.spacecraft.m,\
			self.gamma_init)

#		self.spacecraft.Cd[0] = self.spacecraft.aero_dat(self.Ma[0], self.Re[0], self.Kn[0])[0]
#		self.spacecraft.Cl[0] = self.spacecraft.aero_dat(self.Ma[0], self.Re[0], self.Kn[0])[1]
#		self.spacecraft.Cs[0] = self.spacecraft.aero_dat(self.Ma[0], self.Re[0], self.Kn[0])[2]

		#self.dvdt[0] = dv_dt(self.g[0], self.p_dyn[0], \
		#	self.spacecraft.ballistic_coeff, self.V[0], self.gamma[0])
		#self.dgdt[0] = dgamma_dt(self.gamma[0], self.g[0], self.V[0], self.R, self.h[0])
		#self.dhdt[0] = dh_dt(self.gamma[0], self.V[0])
		#self.drdt[0] = dr_dt(self.R, self.gamma[0], self.h[0])

		if self.console_output == True:
#			print('\033[1;32mGreen like Grass\033[1;m')
			print('\033[1;34m=== TRAJ_CALC ===\033[1;m')
			print('\033[1;34mMODEL INITIALISED.  INITIAL STEP COUNT: %i\033[1;m' % self.steps_storage	)

		return None

	def extend(self):
		# Extend solver time array
		self.time_steps = np.append(self.time_steps, self.time_steps[-1] +
			(self.dt * np.ones(self.steps/10)))

		# Extend solver time array
		self.solver_time = np.append(self.solver_time, np.zeros(self.steps/10))

		# Extend solver input array
		self.y_input = np.vstack([self.y_input, np.zeros([self.steps/10, 4])])

		# Extend dynamic pressure array
		self.p_dyn = np.append(self.p_dyn, np.zeros(self.steps/10))

		# Extend gravitational acceleration array
		self.g = np.append(self.g, np.zeros(self.steps/10))

		# Extend interpolated atmospheric arrays
		self.solver_rho = np.append(self.solver_rho, np.zeros(self.steps/10))
		self.solver_a = np.append(self.solver_a, np.zeros(self.steps/10))
		self.solver_mu = np.append(self.solver_mu, np.zeros(self.steps/10))
		self.solver_p = np.append(self.solver_p, np.zeros(self.steps/10))
		self.solver_T = np.append(self.solver_T, np.zeros(self.steps/10))

		# Extend Mach number array
		self.Ma = np.append(self.Ma, np.zeros(self.steps/10))

		# Extend solution array with zeros to avoid repeated use of append function
		self.sol = np.vstack([self.sol, np.zeros([self.steps/10, 4])])

		# Extend trajectory parameter arrays
		self.V = np.append(self.V, np.zeros(self.steps/10))
		self.gamma = np.append(self.gamma, np.zeros(self.steps/10))
		self.h = np.append(self.h, np.zeros(self.steps/10))
		self.r = np.append(self.r, np.zeros(self.steps/10))

		if self.console_output == True:
			print('STEP COUNT LIMIT REACHED.  EXTENDING SOLUTION BY %i STEPS' % self.steps/10)

		return None

	def truncate(self):
		# Truncate solution arrays to remove trailing zeros (from unused elements)
#		self.V 			= np.delete(self.V, np.arange(self.index+1, len(self.V)))
#		self.gamma 		= np.delete(self.gamma, np.arange(self.index+1, len(self.gamma)))
#		self.h 			= np.delete(self.h, np.arange(self.index+1, len(self.h)))
#		self.r 			= np.delete(self.r, np.arange(self.index+1, len(self.r)))
#		self.p_dyn 		= np.delete(self.p_dyn, np.arange(self.index+1, len(self.p_dyn)))
#		self.solver_time 	= np.delete(self.solver_time, np.arange(self.index+1, len(self.solver_time)))
#		self.solver_rho 		= np.delete(self.solver_rho, np.arange(self.index+1, len(self.solver_rho)))
#		self.solver_p 		= np.delete(self.solver_p, np.arange(self.index+1, len(self.solver_p)))
#		self.solver_T 		= np.delete(self.solver_T, np.arange(self.index+1, len(self.solver_T)))
#		self.solver_mu 		= np.delete(self.solver_mu, np.arange(self.index+1, len(self.solver_mu)))
#		self.solver_a 		= np.delete(self.solver_a, np.arange(self.index+1, len(self.solver_a)))
#		self.g 			= np.delete(self.g, np.arange(self.index+1, len(self.g)))
#		self.Ma 			= np.delete(self.Ma, np.arange(self.index+1, len(self.Ma)))
#		self.Kn 			= np.delete(self.Kn, np.arange(self.index+1, len(self.Kn)))
#		self.Re 			= np.delete(self.Re, np.arange(self.index+1, len(self.Re)))
#		self.mfp 			= np.delete(self.mfp, np.arange(self.index+1, len(self.mfp)))
#
#		sol_temp = self.sol
#		self.sol = np.zeros([self.index, 4])
#		self.sol = sol_temp[0:self.index, :]

		truncate(self, self.index+1, self.l)

		return None

	def final_step_event(self):
		# Interpolation routine to find conditions at time of final step
		# i.e. when h = h_final
		self.final_values = placeholder()

#		final_time = spint.griddata(self.h, self.solver_time,
#			self.h_final, method='linear')
#		final_V = spint.griddata(self.h, self.V, self.h_final,
#			method='linear')
#		final_gamma = spint.griddata(self.h, self.gamma,
#			self.h_final, method='linear')
#		final_rho = spint.griddata(self.atmosphere.h, self.atmosphere.rho,
#			self.h_final, method='linear')
#		final_p = spint.griddata(self.atmosphere.h, self.atmosphere.p,
#			self.h_final, method='linear')
#		final_T = spint.griddata(self.atmosphere.h, self.atmosphere.T,
#			self.h_final, method='linear')
#		final_mu = spint.griddata(self.atmosphere.h, self.atmosphere.mu,
#			self.h_final, method='linear')
#		final_p_dyn = fcl.p_dyn(rho=final_rho, V=final_V)
#		final_g = grav_sphere(self.g_0, self.R, 0)
#		final_a = spint.griddata(self.atmosphere.h, self.atmosphere.a,
#			self.h_final, method='linear')
#		final_Ma = final_V / final_a

		final_list = interpolate_event(self, self.h[self.index], self.l)

#		self.final_values.time		= final_time
#		self.final_values.V         = final_V
#		self.final_values.gamma     = final_gamma
#		self.final_values.rho       = final_rho
#		self.final_values.p         = final_p
#		self.final_values.T         = final_T
#		self.final_values.mu        = final_mu
#		self.final_values.p_dyn     = final_p_dyn
#		self.final_values.g         = final_g
#		self.final_values.g         = final_a
#		self.final_values.Ma        = final_Ma
#		self.final_values.h 		= self.h_final

		for index, val in enumerate(self.l):
			#self.final_values.__dict__[val] = final_list[index]
			self.final_values.__dict__.update({val : final_list[index]})

		if self.console_output == True:
			print('\033[1;34mEND EVENT CONDITIONS CALCULATED\033[1;m')

#		return [final_time, final_V, final_gamma, final_rho, final_p, final_T,
#					final_mu, final_p_dyn, final_g, final_a, final_Ma]

		return None

	def final_step_assign(self):
		# Assign values calculated for final step
#		self.solver_time[self.index] 	= self.final_values.time
#		self.V[self.index] 				= self.final_values.V
#		self.gamma[self.index] 			= self.final_values.gamma
#		self.solver_rho[self.index] 	= self.final_values.rho
#		self.solver_p[self.index] 		= self.final_values.p
#		self.solver_T[self.index] 		= self.final_values.T
#		self.solver_mu[self.index] 		= self.final_values.mu
#		self.p_dyn[self.index] 			= self.final_values.p_dyn
#		self.g[self.index] 				= self.final_values.g
#		self.solver_a[self.index] 		= self.final_values.g
#		self.Ma[self.index] 			= self.final_values.Ma
#		self.h[self.index]			= self.final_values.h

		assign(self, self.final_values, self.i, self.l)

		return None

#	def interpolate_atmosphere(self, h_interp):
#		rho_interp = spint.griddata(self.atmosphere.h, self.atmosphere.rho,
#				h_interp, method='linear')
#		a_interp = spint.griddata(self.atmosphere.h, self.atmosphere.a,
#				h_interp, method='linear')
#		p_interp = spint.griddata(self.atmosphere.h, self.atmosphere.p,
#				h_interp, method='linear')
#		T_interp = spint.griddata(self.atmosphere.h, self.atmosphere.T,
#				h_interp, method='linear')
#		mu_interp = spint.griddata(self.atmosphere.h, self.atmosphere.mu,
#				h_interp, method='linear')
#
#		return [rho_interp, a_interp, p_interp, T_interp, mu_interp]

	def simulate_dopri(self, dt=1E-2):
		"""
		Run trajectory calculations using explicit Runge-Kutta method of order
		4(5) from Dormand & Prince
		"""
		# Set timestep for ODE solver
		self.dt = dt
		self.time_steps = np.cumsum(self.dt * np.ones(self.steps_storage))

		# Create ODE object from SciPy using Dormand-Prince RK solver
		self.eq = integrate.ode(traj_3DOF_dt).set_integrator('dop853', nsteps=1E8,
			rtol=1E-10)

		# Set initial conditions
		y_init = [self.V_init, self.gamma_init, self.h_init, self.r[0]]
		self.eq.set_initial_value(y_init, t=self.time_steps[0])

#		# Create empty arrays for storage of results from ODE solver
#		self.sol = np.zeros([self.steps, 4])
#		self.solver_time = np.zeros(self.steps)
#		self.solver_rho = np.zeros(self.steps)
#		self.solver_a = np.zeros(self.steps)
#		self.y_input = np.zeros([self.steps, 4])

		# Generate counter
		index = 1
		self.index = index

		# Initial conditions are: V, gamma, h, r.  These are at index = 0
		# Other parameters (like dynamic pressure and gravitational
		# attraction) are calculated for this step (also index = 0)
		# ODE solver then calculates V, gamma, h, and r at the next step (index = 1)
		# Then parameters and updated as above, and the loop continues.
		# So:
		# INIT:  Define V, gamma, h, r @ start
		#	 	Calculate parameters @ start
		# SOLVE: Find V, gamma, h, r
		#

		# Solve ODE system using conditional statement based on altitude
		while self.h[index-1] > 0:

			# Update ODE solver parameters from spacecraft object and
			# atmospheric model at each separate time step
			if self.spacecraft.aero_coeffs_type == 'CONSTANT':
				params = [self.R, self.g[index-1], self.spacecraft.ballistic_coeff,
					self.solver_rho[index-1], self.spacecraft.Cl, self.spacecraft.Cd]
				self.eq.set_f_params(params)

			elif self.spacecraft.aero_coeffs_type == 'VARIABLE':
				self.spacecraft.update_aero(self.index, self.Re[index-1],
					self.Ma[index-1], self.Kn[index-1], self.solver_p[index-1],
					self.p_dyn[index-1], self.solver_rho[index-1],
					(self.solver_Cp[index-1] / self.solver_Cv[index-1]),
					self.spacecraft.Cd[index-1], self.spacecraft.Cl[index-1])
				
				params = [self.R, self.g[index-1], self.spacecraft.ballistic_coeff[index-1],
					self.solver_rho[index-1], self.spacecraft.Cl[index-1],
					self.spacecraft.Cd[index-1]]
				
				self.eq.set_f_params(params)

			# Update parameters with atmospheric density at each altitude step
#			params = [self.R, self.g[index-1], self.spacecraft.ballistic_coeff,
#				self.solver_rho[index-1], self.spacecraft.Cl, self.spacecraft.Cd]
#			self.eq.set_f_params(params)

			# Solve ODE system (sol[V, gamma, h, r])
			self.sol[index, :] = self.eq.integrate(self.time_steps[index])

			# Unpack ODE solver results into storage structures
			self.V[index] = self.sol[index, 0]
			self.gamma[index] = self.sol[index, 1]
			self.h[index] = self.sol[index, 2]
			self.r[index] = self.sol[index, 3]

			# Interpolate for freestream density in atmosphere model
			# (this avoids a direct call to an atmosphere model, allowing more
			# flexibility when coding as different models have different interfaces)
#			rho_interp = spint.griddata(self.atmosphere.h, self.atmosphere.rho,
#				self.h[index], method='linear')
#			self.solver_rho[index] = rho_interp
			self.solver_rho[index], self.solver_a[index], \
				self.solver_p[index], self.solver_T[index], \
				self.solver_mu[index], self.solver_Cp[index], \
				self.solver_Cv[index] = \
				interpolate_atmosphere(self, self.h[index])

			# Calculate energies
			self.energyKinetic[index] = calculateKineticEnergy( \
				self.spacecraft.m, self.V[index])
			self.energyPotential[index] = calculatePotentialEnergy( \
				self.spacecraft.m, self.mu, self.h[index], self.planet.R)
			self.energyOrbitalSpecific[index] = calculateSpecificOrbitalEnergy(\
				self.energyKinetic[index], self.energyPotential[index], \
				self.spacecraft.m, self.gamma[index])

			# Calculate gravitational acceleration at current altitude
			self.g[index] = grav_sphere(self.g_0, self.R, self.h[index])

			# Calculate dynamic pressure iteration results
			self.p_dyn[index] = fcl.p_dyn(rho=params[3], V=self.sol[index, 0])

			# Calculate Mach, Knudsen, and Reynolds numbers
			self.Ma[index] = self.V[index] / self.solver_a[index]
			self.mfp[index] = fcl.mean_free_path(self.solver_T[index],
				self.solver_p[index], self.atmosphere.d)
			self.Kn[index] = self.mfp[index] / self.spacecraft.L
			self.Re[index] = fcl.Reynolds(self.solver_rho[index],
				self.V[index], self.spacecraft.L, self.solver_mu[index])

			# Save inputs for inspection
			self.solver_time[index] = self.eq.t
			self.y_input[index, :] = self.eq.y

			# Advance iteration counter
			index += 1
			self.index = index

			# Check if solution storage array has reached maximum size
			if index == len(self.sol)-10:
				self.extend()

			#print(index)
			# Print solution progress to check for stability
			if self.console_output == True:
				if np.mod(index, self.steps_storage/self.console_interval) == 0:
					print('\033[1;31mITER: \033[1;m' \
					'\033[1;37m%i; \033[1;m' \
					'\033[1;32mALT: \033[1;m' \
					'\033[1;37m%3.2f km; \033[1;m' \
					'\033[1;36mORBITAL ENERGY: \033[1;m' \
					'\033[1;37m%3.2e MJ/kg\033[1;m' % \
					(index, self.h[index-1]/1E3, \
					self.energyOrbitalSpecific[index-1]/1E6))

			# Check for atmospheric model interpolation errors
			# (OUT_OF_BOUNDS error)
			error_out_of_bounds(self, self.index)
			if self.out_of_bounds_error == True:
				break
			else:
				pass

#			# Update ODE solver params
#			update_params = [self.F_x[i], self.F_y[i], self.F_z[i],
#				self.spacecraft.m, self.mu]
#			self.eq.set_f_params(update_params)

		if (self.out_of_bounds_error == False):
			print('\033[1;32m=== SIMULATION COMPLETE ===\033[1;m')
#		# Calculate Mach numbers
#		Ma = self.sol[:, 0] / self.atmosphere.a

		# Copy ODE input and solution arrays to structures in trajectory object
		#self.V = self.sol[:, 0]
		#self.gamma = self.sol[:, 1]
		#self.h = self.sol[:, 2]
		#self.r = self.sol[:, 3]
		#self.p_dyn = p_dyn
		#self.Ma = Ma

		# Compute final step values for non-solver variables
		#self.Ma[t.index] =

		# Subtract 1 from counter so that indexing is more convenient later on
		self.index -= 1

		# Truncate solution arrays to remove trailing zeros
		self.truncate()

		# Perform final step calculations for p_dyn, g, etc.
		self.final_step_event()
		#self.final_step_assign()

		# Perform post solver calculations
		#self.post_calc()

		print('\033[1;34mTRAJECTORY COMPUTED (RK 4/5)\033[1;m')
		print('\033[1;34m%i ITERATIONS, TIMESTEP = %f s, TOTAL TIME = %f s\033[1;m' % \
			(self.index, self.dt, self.solver_time[self.index-1]))

		return [self.sol, self.h, self.y_input, self.p_dyn, self.Ma]

	def calculate_heating(self):
		# Generate new placeholder class for all heat fluxes
		self.qdot 	= placeholder()

		# Generate blank classes for different heat flux mechanisms
		self.qdot.conv 	= placeholder()
		self.qdot.rad 	= placeholder()

		# Generate empty arrays for different correlations
		self.qdot.conv.dh 	= np.zeros(self.index+1)
		self.qdot.conv.bj 	= np.zeros(self.index+1)
		self.qdot.conv.s 	= np.zeros(self.index+1)
		self.qdot.conv.fr 	= np.zeros(self.index+1)
		self.qdot.conv.sg 	= np.zeros(self.index+1)
		self.qdot.rad.bj 	= np.zeros(self.index+1)
		self.qdot.rad.s 		= np.zeros(self.index+1)
		self.qdot.rad.ts 	= np.zeros(self.index+1)

		# Call heat_flux_lib for actual calculations
		# Detra-Hidalgo
		self.qdot.conv.dh = hcl.detra_hidalgo(self.V, self.solver_rho,
			self.spacecraft.R_n)

		# Brandis-Johnston (convective)
		for index, val in enumerate(self.h):
			self.qdot.conv.bj[index] = hcl.brandis_johnston(self.V[index],
				self.solver_rho[index], self.spacecraft.R_n, mode='conv')

		# Smith (convective)
		self.qdot.conv.s = hcl.smith(self.V, self.solver_rho,
			self.spacecraft.R_n, mode='conv', planet='Earth')

		# Fay-Riddell
		FR = hcl.FayRiddellHelper(gas_file='air.xml', d=4e-10, thetav=5500)
		for index, val in enumerate(self.h):
			try:
				self.qdot.conv.fr[index] = FR.calculate(self.Ma[index], 
					self.solver_p[index], self.solver_T[index], 300, 
					self.spacecraft.R_n, geom='sph', chem='equil')
			except:
				self.qdot.conv.fr[index] = 0.0
			
			
# 		for index, val in enumerate(self.h):
# 			try:
# 				self.qdot.conv.fr[index] = hcl.fay_riddell_helper(self.Ma[index], 
# 					self.solver_p[index], self.solver_T[index], 300, 
# 					self.spacecraft.R_n, geom='sph', chem='equil')
# 			except:
# 				self.qdot.conv.fr[index] = 0.0

		# Sutton-Graves (convective)
		self.qdot.conv.sg	= hcl.sutton_graves(self.V, self.solver_rho,
			self.spacecraft.R_n)

		# Brandis-Johnston (radiative)
		for index, val in enumerate(self.h):
			self.qdot.rad.bj[index] = hcl.brandis_johnston(self.V[index],
				self.solver_rho[index], self.spacecraft.R_n, mode='rad')

		# Smith (radiative)
		self.qdot.rad.bj = hcl.smith(self.V, self.solver_rho,
			self.spacecraft.R_n, mode='rad', planet='Earth')

		# Tauber-Sutton (radiative)
		rho_ratio = fcl.normal_shock_ratios(self.Ma, self.solver_Cp / \
			self.solver_Cv)[4]
		for index, val in enumerate(self.h):
			self.qdot.rad.ts[index] = hcl.tauber_sutton(self.V[index],
				self.solver_rho[index], self.spacecraft.R_n, rho_ratio[index])

		# Net flux (convective heating, radiative cooling)
		self.qdot.net = placeholder()
		self.qdot.net.bj = self.qdot.conv.bj + self.qdot.rad.bj
		self.qdot.net.s = self.qdot.conv.s + self.qdot.rad.s
		self.qdot.net.sg_ts = self.qdot.conv.sg + self.qdot.rad.ts

	def plot_triple(self):

		fig = plt.figure(figsize=[12, 10])
		ax1 = fig.add_subplot(224)
		ax2 = fig.add_subplot(223)
		ax4 = fig.add_subplot(211)

		#alt = self.h / 1000
		#g_range = self.r / 1000
		#gamma_deg = np.rad2deg(self.gamma)
		line_width = 2.0
		num = int(len(self.h) / 20)

		quiv = np.array([self.r[0:-1:num]/1000, self.h[0:-1:num]/1000, \
			self.gamma[0:-1:num]]).T

		quiver_x = np.log(self.V[0:-1:num]) * np.cos(quiv[:, 2])
		quiver_y = -np.log(self.V[0:-1:num]) * np.sin(quiv[:, 2])

		ax4.plot((self.r/1000), (self.h/1000), color='c', linewidth=line_width, alpha=1.0)
		ax4.scatter(quiv[:, 0], quiv[:, 1], color='b', s=15, alpha=0.25, zorder=3)
		ax4.scatter(self.r[-1]/1000, self.h[-1]/1000, s=300, marker='x',
			color='r', alpha=0.3, zorder=3)
		quiv_plot = ax4.quiver(quiv[:, 0], quiv[:, 1], quiver_x, quiver_y, \
			scale=100, linewidth=1, edgecolor='b', headwidth=10, headlength=10,\
			width=0.001, alpha=0.2)
		ax4.quiverkey(quiv_plot, 0.9, 0.9, 5, 'log(Velocity), $log(V)$ [m/s]', color='b')
		ax4.set_xlabel('Ground range, r (km)')
		ax4.set_ylabel('Altitude, h (km)')
		ax4.grid(True)

		#ax0.plot(g_range, alt, color='g', linewidth=line_width)
		#ax0.set_xlabel('Ground range, r (m)')
		#ax0.set_ylabel('Altitude, h (km)')
		#ax0.grid(True)

		ax1.plot(self.Ma, self.h/1000, color='r', linewidth=line_width)
		ax1.scatter(self.Ma[-1], self.h[-1]/1000, s=300, marker='x',
			color='r', alpha=0.5, zorder=3)
		ax1.set_xlabel('Mach number, Ma')
		ax1.set_ylabel('Altitude, h (km)')
		ax1.set_xscale('log')
		ax1.grid(True)

		ax2.plot(self.V, self.h/1000, color='b', linewidth=line_width)
		ax2.scatter(self.V[-1], self.h[-1]/1000, s=300, marker='x',
			color='r', alpha=0.5, zorder=3)
		ax2.set_xlabel('Velocity, V (m/s)')
		ax2.set_ylabel('Altitude, h (km)')
		ax2.set_xscale('log')
		ax2.grid(True)

		#ax3.plot(gamma_deg, alt, color='g', linewidth=line_width)
		#ax3.set_xlabel(r'Flight path angle, $\gamma$ (degrees)')
		#ax3.set_ylabel('Altitude, h (km)')
		#ax3.grid(True)

		plt.tight_layout()

		return None

	def plot_trajectory(self):
		fig = plt.figure(figsize=[12, 6])
		ax4 = fig.add_subplot(111)

		#alt = self.h / 1000
		#g_range = self.r / 1000
		#gamma_deg = np.rad2deg(self.gamma)
		line_width = 2.0
		num = int(len(self.h) / 20)

		quiv = np.array([self.r[0:-1:num]/1000, self.h[0:-1:num]/1000, \
			self.gamma[0:-1:num]]).T

		quiver_x = np.log(self.V[0:-1:num]) * np.cos(quiv[:, 2])
		quiver_y = -np.log(self.V[0:-1:num]) * np.sin(quiv[:, 2])

		ax4.plot((self.r/1000), (self.h/1000), color='c', linewidth=line_width, alpha=1.0)
		ax4.scatter(quiv[:, 0], quiv[:, 1], color='b', s=15, alpha=0.25, zorder=3)
		ax4.scatter(self.r[-1]/1000, self.h[-1]/1000, s=300, marker='x',
			color='r', alpha=0.3, zorder=3)
		quiv_plot = ax4.quiver(quiv[:, 0], quiv[:, 1], quiver_x, quiver_y, \
			scale=100, linewidth=1, edgecolor='b', headwidth=10, headlength=10,\
			width=0.001, alpha=0.2)
		ax4.quiverkey(quiv_plot, 0.85, 0.9, 5, 'log(Velocity), $log(V)$ [m/s]', color='b')
		ax4.set_xlabel('Ground range, r (km)')
		ax4.set_ylabel('Altitude, h (km)')
		ax4.grid(True)

		#ax0.plot(g_range, alt, color='g', linewidth=line_width)
		#ax0.set_xlabel('Ground range, r (m)')
		#ax0.set_ylabel('Altitude, h (km)')
		#ax0.grid(True)

		plt.tight_layout()

		return None

	def show_regimes(self):

		handle = plt.gcf()
		xspan = np.abs(handle.axes[-1].get_xlim()[0]) + \
			np.abs(handle.axes[-1].get_xlim()[1])
		xlim = handle.axes[-1].get_xlim()[0] + (0.02 * xspan)

		# REGIME HIGHLIGHTS
		reg_col = [[0, 1, 0],
						[0, 0.66, 0.33],
						[0, 0.33, 0.66],
						[0, 0, 1]]

		# Continuum
		if hasattr(self.regimes, 'continuum'):
			plt.axhspan(self.h[self.regimes.continuum[0]]/1000,
				self.h[self.regimes.continuum[-1]]/1000,
				facecolor=reg_col[0], alpha=0.05)
			plt.text(xlim, self.h[self.regimes.continuum[-1]]/1000 + 1,
				'CONTINUUM', fontsize=12, \
			    color=reg_col[0], alpha=0.5)

		# Slip
		if hasattr(self.regimes, 'slip'):
			plt.axhspan(self.h[self.regimes.slip[0]]/1000,
				self.h[self.regimes.slip[-1]]/1000,
				facecolor=reg_col[1], alpha=0.05)
			plt.text(xlim, self.h[self.regimes.slip[-1]]/1000 + 1,
				'SLIP', fontsize=12, \
			    color=reg_col[1], alpha=0.5)

		# Transition
		if hasattr(self.regimes, 'transition'):
			plt.axhspan(self.h[self.regimes.transition[0]]/1000,
				self.h[self.regimes.transition[-1]]/1000,
				facecolor=reg_col[2], alpha=0.05)
			plt.text(xlim, self.h[self.regimes.transition[-1]]/1000 + 1,
				'TRANSITION', fontsize=12, \
			    color=reg_col[2], alpha=0.5)

		# Free molecular
		if hasattr(self.regimes, 'free_molecular'):
			plt.axhspan(self.h[self.regimes.free_molecular[0]]/1000,
				self.h[self.regimes.free_molecular[-1]]/1000,
				facecolor=reg_col[3], alpha=0.05)
			plt.text(xlim, self.h[self.regimes.free_molecular[-1]]/1000 + 1,
				'FREE MOLECULAR', fontsize=12, \
			    color=reg_col[3], alpha=0.5)

		# REGIME BOUNDARY LINES
		# Continuum-Slip
		if (self.index_cont_slip != None):
			plt.axhline(y=self.h[self.index_cont_slip]/1000,
				color=reg_col[1], alpha=0.2, zorder=0, linewidth=2.0)

		# Slip-Transition
		if (self.index_slip_tran != None):
			plt.axhline(y=self.h[self.index_slip_tran]/1000,
				color=reg_col[2], alpha=0.2, zorder=0, linewidth=2.0)

		# Transition-Free molecular
		if (self.index_tran_freemol != None):
			plt.axhline(y=self.h[self.index_tran_freemol]/1000,
				color=reg_col[3], alpha=0.2, zorder=0, linewidth=2.0)

		return None

	def post_calc(self):
		"""
		Perform post-integration calculations (such as Knudsen and Reynolds
		numbers, location of regimes, etc.)
		"""

#		self.mfp = fcl.mean_free_path(self.solver_T, self.solver_p,
#			self.atmosphere.d)
#		self.Kn = self.mfp / self.spacecraft.L
##		self.Re = fcl.KnReMa(self.atmosphere.k, Kn=self.Kn,
##			Ma=self.Ma)
#		self.Re = fcl.Reynolds(self.solver_rho, self.V, self.spacecraft.L,
#			self.solver_mu)

		# Continuum: 0 < Kn < 0.001
		# Slip: 0.001 <= Kn < 0.1
		# Transition: 0.1 <= Kn < 10
		# Free molecular: 10 < Kn

		self.regimes = placeholder()

		if len(np.argwhere(self.Kn > 10)) != 0:
			self.index_tran_freemol = np.argwhere(self.Kn > 10)[-1]
			self.regimes.free_molecular = np.argwhere(self.Kn >= 10)
		else:
			self.index_tran_freemol = None

		if len(np.argwhere(self.Kn > 0.1)) != 0:
			self.index_slip_tran = np.argwhere(self.Kn > 0.1)[-1]
			self.regimes.transition = np.argwhere((self.Kn < 10) & (self.Kn >= 0.1))
		else:
			self.index_slip_tran = None

		if len(np.argwhere(self.Kn > 0.001)) != 0:
			self.index_cont_slip = np.argwhere(self.Kn > 0.001)[-1]
			self.regimes.slip = np.argwhere((self.Kn < 0.1) & (self.Kn >= 0.001))
		else:
			self.index_cont_slip = None

		if len(np.argwhere((self.Kn > 0) & (self.Kn <= 0.001))) != 0:
			self.regimes.continuum = np.argwhere((self.Kn < 0.001) & (self.Kn >= 0))
		else:
			self.index_cont_slip = None

		return [self.mfp, self.Kn, self.Re]

	def check_for_out_of_bounds_error(self):
		check_a = np.isnan(self.solver_a)
		check_rho = np.isnan(self.solver_rho)
		sum_error_a = np.sum(check_a)
		sum_error_rho = np.sum(check_rho)

		if (sum_error_a != 0) or (sum_error_rho != 0):
			print('NaN entries found in atmospheric interpolation model.  Try expanding altitude bounds.')
		else:
			print('No NaN entries found in atmospheric interpolation model.')
		return None

	def export_dict(self):
		"""
		Export first level class contents as Python dictionary object
		(inherited classes are not exported)
		"""

		t = dict(self.__dict__)

		del t['regimes']
		del t['qdot']
		del t['atmosphere']
		del t['final_values']
		del t['spacecraft']
		del t['eq']

		return [t, qdot, regimes, atmosphere, spacecraft, eq]

class trajectory_aerobraking:
	def __init__(self, mu_planet, spacecraft, atmosphere, V0, X0, sim_time, R,
		console_output=True, dt=0.1):
		# Pull in spacecraft and atmosphere classes for convenient variable access
		self.spacecraft = spacecraft
		self.atmosphere = atmosphere

		# Verbose solver output flag
		self.console_output = console_output

		# Miscellaneous initial conditions and constant variables
		self.mu = mu_planet
		self.InitalV = V0
		self.initalpos = X0
		self.simtime = sim_time
		self.dt = dt
		self.N = np.int(self.simtime / dt)
		self.R = R

		# Create empty arrays for storage of results from ODE solver
		self.sol = np.zeros([self.N, 6])
		self.V_xyz = np.zeros([self.N, 3])
		self.pos_xyz = np.zeros([self.N, 3])
		self.V_mag = np.zeros(self.N)
		self.pos_mag = np.zeros(self.N)
		self.theta = np.zeros(self.N)
		self.theta = np.zeros(self.N)
		self.h = np.zeros(self.N)

		# Forces in inertial frame
		self.F_x = np.zeros([self.N])
		self.F_y = np.zeros([self.N])
		self.F_z = np.zeros([self.N])
		self.F_mag = np.zeros(self.N)

		self.V_xyz[0, :] = V0
		self.pos_xyz[0, :] = X0

		# Velocity and position vector magnitudes
		self.V_mag[0] = np.linalg.norm(self.V_xyz[0, :])
		self.pos_mag[0] = np.linalg.norm(self.pos_xyz[0, :])

		# Interpolated atmospheric variables
		self.solver_rho 	= np.zeros(self.N)
		self.solver_mu 	= np.zeros(self.N)
		self.solver_a 	= np.zeros(self.N)
		self.solver_p 	= np.zeros(self.N)
		self.solver_T 	= np.zeros(self.N)
		self.solver_time 	= np.zeros(self.N)

		# Altitude above surface of planet (assumed to be perfectly spherical)
		self.h[0] = np.linalg.norm(self.pos_xyz[0, :]) - self.R

		self.lift = np.zeros(self.N)
		self.drag = np.zeros(self.N)
		self.side_force = np.zeros(self.N)
		self.forces_rotating = np.zeros([self.N, 3])
		self.forces_inertial = np.zeros([self.N, 3])
		self.alpha = np.zeros(self.N)
		self.theta = np.zeros(self.N)

		# Interpolate atmospheric variables
#		self.solver_rho[0], self.solver_a[0], \
#			self.solver_p[0], self.solver_T[0], \
#			self.solver_mu[0] = interpolate_atmosphere(self, self.h[0])
		self.solver_rho[0], self.solver_a[0], \
			self.solver_p[0], self.solver_T[0], \
			self.solver_mu[0], _, _ = atmosphere_nrl_query(self.h[0])

		self.lift[0] = fcl.aero_force(self.solver_rho[0], self.V_mag[0],
			self.spacecraft.Cl, self.spacecraft.A)
		self.drag[0] = fcl.aero_force(self.solver_rho[0], self.V_mag[0],
			self.spacecraft.Cd, self.spacecraft.A)
		self.side_force[0] = fcl.aero_force(self.solver_rho[0], self.V_mag[0],
			self.spacecraft.Cs, self.spacecraft.A)
		self.forces_rotating[0, :] = np.array([self.lift[0], self.drag[0],
			self.side_force[0]])

		# Calculate Euler angles (pitch and yaw; roll is assumed to be zero)
		self.alpha[0], self.theta[0] = rotate_lib.vector_to_euler(self.pos_xyz[0, 0],
			self.pos_xyz[0, 1], self.pos_xyz[0, 2])

		# Transform aero forces from rotating to inertial frame
		self.forces_inertial[0, :] = rotate_lib.roty(self.forces_rotating[0, :],
			self.alpha[0], mode='rad')
		self.forces_inertial[0, :] = rotate_lib.rotz(self.forces_rotating[0, :],
			self.theta[0], mode='rad')

		# Split forces into components
		self.F_x[0] = self.forces_inertial[0, 0]
		self.F_y[0] = self.forces_inertial[0, 1]
		self.F_z[0] = self.forces_inertial[0, 2]

		# Calculate magnitude of forces
		self.F_mag[0] = np.abs(np.linalg.norm([self.F_x[0], self.F_y[0],
			self.F_z[0]]))

		# Define list of keys for obejct dict (self.__dict__)
		# To be used by truncation, event interpolation, and variable
		# assignment functions
		self.l = ['V_xyz', 'V_mag', 'F_mag', 'F_x', 'F_y', 'F_z', 'pos_xyz',
			'pos_mag', 'alpha', 'theta', 'drag', 'lift', 'side_force',
			'h', 'solver_time', 'solver_rho',
			'solver_p', 'solver_T', 'solver_mu', 'solver_a',
			'forces_inertial', 'forces_rotating']

		# Set simulation termination flags
		self.ground_strike = False
		self.out_of_bounds_error = False

		if self.console_output == True:
			print('MODEL INITIALISED.  INITIAL STEP COUNT: %i' % self.N)

		return None

#	def interpolate_atmosphere(self, h_interp):
#		rho_interp = spint.griddata(self.atmosphere.h, self.atmosphere.rho,
#				h_interp, method='linear')
#		a_interp = spint.griddata(self.atmosphere.h, self.atmosphere.a,
#				h_interp, method='linear')
#		p_interp = spint.griddata(self.atmosphere.h, self.atmosphere.p,
#				h_interp, method='linear')
#		T_interp = spint.griddata(self.atmosphere.h, self.atmosphere.T,
#				h_interp, method='linear')
#		mu_interp = spint.griddata(self.atmosphere.h, self.atmosphere.mu,
#				h_interp, method='linear')
#
#		return [rho_interp, a_interp, p_interp, T_interp, mu_interp]

	def simulate_dopri(self, rtol=1E-4, nsteps=1E8):
		# Store ODE solver variables
		#self.dt = dt
		self.rtol = rtol
		self.nsteps = nsteps

		# Set up ODE solver
		self.eq = integrate.ode(orbit_xyz).set_integrator('dop853',
			nsteps=self.nsteps, rtol=self.rtol)

		y_init = [self.InitalV[0], self.InitalV[1], self.InitalV[2],
			self.initalpos[0], self.initalpos[1],  self.initalpos[2]]
		self.y_init = y_init
		self.eq.set_initial_value(y_init, t=0)

		#set inital values
		params = [self.F_x[0], self.F_y[0], self.F_z[0],
			self.spacecraft.m, self.mu]
		self.eq.set_f_params(params)

		for i in range(1, self.N):
			# Update stored counter
			self.i = i

			# Update ODE solver params
			update_params = [self.F_x[i-1], self.F_y[i-1], self.F_z[i-1],
				self.spacecraft.m, self.mu]
			self.eq.set_f_params(update_params)

			# Solve ODE system
			self.sol[i, :] = self.eq.integrate(self.eq.t + self.dt)
			self.solver_time[i] = self.eq.t

			# Unpack ODE solver results
			self.V_xyz[i, :] = self.sol[i, 0:3]
			self.V_mag[i] = np.linalg.norm(self.V_xyz[i, :])
			self.pos_xyz[i, :] = self.sol[i, 3:6]
			self.pos_mag[i] = np.linalg.norm(self.pos_xyz[i, :])

			# Update altitude
			self.h[i] = np.linalg.norm(self.pos_xyz[i, :]) - self.R

			# Interpolate atmospheric variables
			self.solver_rho[i], self.solver_a[i], \
				self.solver_p[i], self.solver_T[i], \
				self.solver_mu[i] = interpolate_atmosphere(self, self.h[i])
#			self.solver_rho[i], self.solver_a[i], \
#				self.solver_p[i], self.solver_T[i], \
#				self.solver_mu[i] = atmosphere_nrl_query(self.h[i])

			# Axial aero forces
			self.lift[i] = fcl.aero_force(self.solver_rho[i], self.V_mag[i],
				self.spacecraft.Cl, self.spacecraft.A)
			self.drag[i] = fcl.aero_force(self.solver_rho[i], self.V_mag[i],
				self.spacecraft.Cd, self.spacecraft.A)
			self.side_force[i] = fcl.aero_force(self.solver_rho[i], self.V_mag[i],
				self.spacecraft.Cs, self.spacecraft.A)
			self.forces_rotating[i, :] = np.array([self.lift[i], self.drag[i],
				self.side_force[i]])

			# Calculate Euler angles (pitch and yaw; roll is assumed to be zero)
			self.alpha[i], self.theta[i] = rotate_lib.vector_to_euler(self.sol[i, 3],
				self.sol[i, 4], self.sol[i, 5])

			# Transform aero forces from rotating to inertial frame
			self.forces_inertial[i, :] = rotate_lib.roty(self.forces_rotating[i, :],
				self.alpha[i], mode='rad')
			self.forces_inertial[i, :] = rotate_lib.rotz(self.forces_rotating[i, :],
				self.theta[i], mode='rad')

			# Split forces into components
			self.F_x[i] = -self.forces_inertial[i, 0]
			self.F_y[i] = -self.forces_inertial[i, 1]
			self.F_z[i] = -self.forces_inertial[i, 2]

			# Calculate magnitude of forces
			self.F_mag[i] = np.abs(np.linalg.norm([self.F_x[i], self.F_y[i],
				self.F_z[i]]))

			# Print integration progress
			if self.console_output == True:
				if np.mod(i, self.N/100) == 0:
					print('%3.1f%%; ITERATION: %i; ALTITUDE: %f km' % \
						(100*(np.float(i)/self.N), i, self.h[i]/1000))

			# Check for ground strike
			ground_strike(self, self.i)
			if self.ground_strike == True:
				break
			else:
				pass
#			if self.h[i] <= 0:
#				self.ground_strike = True
#				print('GROUND STRIKE EVENT (ALTITUDE = 0) DETECTED BETWEEN ' \
#					'INDEXES %i AND %i' % (i-1, i))
#				break

			# Check for atmospheric model interpolation errors
			# (OUT_OF_BOUNDS error)
			error_out_of_bounds(self, self.i)
			if self.out_of_bounds_error == True:
				break
			else:
				pass
#			if np.isnan(self.solver_rho[i]) == True:
#				print('ERROR: ATMOSPHERIC INTERPOLATION OUT OF BOUNDS AT ' \
#					'INDEX %i, TRY EXPANDING ALTITUDE RANGE' % i)
#				break

#			# Update ODE solver params
#			update_params = [self.F_x[i], self.F_y[i], self.F_z[i],
#				self.spacecraft.m, self.mu]
#			self.eq.set_f_params(update_params)

		if (self.out_of_bounds_error == False) and (self.ground_strike == False):
			print('=== SIMULATION COMPLETE ===')

#		# Subtract 1 from counter so that indexing is more convenient later on
#		#self.i -= 1
#
		self.truncate()

		if self.ground_strike == True:
			self.final_step_event()
			self.final_step_assign()

		return self.sol

	def check_for_out_of_bounds_error(self):
		check_a = np.isnan(self.solver_a)
		check_rho = np.isnan(self.solver_rho)
		sum_error_a = np.sum(check_a)
		sum_error_rho = np.sum(check_rho)

		if (sum_error_a != 0) or (sum_error_rho != 0):
			print('NaN entries found in atmospheric interpolation model.  Try expanding altitude bounds.')
		else:
			print('No NaN entries found in atmospheric interpolation model.')
		return None

	def truncate(self):
		# Truncate solution arrays to remove trailing zeros (from unused elements)
#		self.V_xyz		= np.delete(self.V_xyz, np.arange(self.index+1, len(self.V_xyz)))
#		self.V_mag		= np.delete(self.V_mag, np.arange(self.index+1, len(self.V_mag)))
#		self.F_mag		= np.delete(self.F_mag, np.arange(self.index+1, len(self.F_mag)))
#		self.F_x			= np.delete(self.F_x, np.arange(self.index+1, len(self.F_x)))
#		self.F_y			= np.delete(self.F_y, np.arange(self.index+1, len(self.F_y)))
#		self.F_z			= np.delete(self.F_z, np.arange(self.index+1, len(self.F_z)))
#		self.alpha		= np.delete(self.alpha, np.arange(self.index+1, len(self.alpha)))
#		self.theta		= np.delete(self.theta, np.arange(self.index+1, len(self.theta)))
#		self.drag		= np.delete(self.drag, np.arange(self.index+1, len(self.drag)))
#		self.lift		= np.delete(self.lift, np.arange(self.index+1, len(self.lift)))
#		self.side_force	= np.delete(self.side_force, np.arange(self.index+1, len(self.side_force)))
#		self.pos_mag		= np.delete(self.pos_mag, np.arange(self.index+1, len(self.pos_mag)))
#		self.pos_xyz		= np.delete(self.pos_xyz, np.arange(self.index+1, len(self.pos_xyz)))
#		self.h 			= np.delete(self.h, np.arange(self.index+1, len(self.h)))
#		self.r 			= np.delete(self.r, np.arange(self.index+1, len(self.r)))
#		self.solver_time 	= np.delete(self.solver_time, np.arange(self.index+1, len(self.solver_time)))
#		self.solver_rho 	= np.delete(self.solver_rho, np.arange(self.index+1, len(self.solver_rho)))
#		self.solver_p 	= np.delete(self.solver_p, np.arange(self.index+1, len(self.solver_p)))
#		self.solver_T 	= np.delete(self.solver_T, np.arange(self.index+1, len(self.solver_T)))
#		self.solver_mu 	= np.delete(self.solver_mu, np.arange(self.index+1, len(self.solver_mu)))
#		self.solver_a 	= np.delete(self.solver_a, np.arange(self.index+1, len(self.solver_a)))
#		#self.g 			= np.delete(self.g, np.arange(self.index+1, len(self.g)))
#		self.Ma 			= np.delete(self.Ma, np.arange(self.index+1, len(self.Ma)))
#		self.Kn 			= np.delete(self.Kn, np.arange(self.index+1, len(self.Kn)))
#		self.Re 			= np.delete(self.Re, np.arange(self.index+1, len(self.Re)))
#		self.mfp 		= np.delete(self.mfp, np.arange(self.index+1, len(self.mfp)))
#		self.force_inertial	= np.delete(self.force_inertial,
#			np.arange(self.index+1, len(self.force_inertial)))
#		self.force_rotating	= np.delete(self.force_rotating,
#			np.arange(self.index+1, len(self.force_rotating)))
#
#		sol_temp = self.sol
#		self.sol = np.zeros([self.index, 4])
#		self.sol = sol_temp[0:self.index, :]

		truncate(self, self.i+1, self.l)

		return None

	def final_step_event(self):
		# Interpolation routine to find conditions at time of final step
		# i.e. when h = h_final
#		self.final_values = placeholder()
#
#		final_time = spint.griddata(self.h, self.solver_time,
#			self.h_final, method='linear')
#		final_V_mag = spint.griddata(self.h, self.V_mag, self.h_final,
#			method='linear')
##		final_gamma = spint.griddata(self.h, self.gamma,
##			self.h_final, method='linear')
#		final_rho = spint.griddata(self.atmosphere.h, self.atmosphere.rho,
#			self.h_final, method='linear')
#		final_p = spint.griddata(self.atmosphere.h, self.atmosphere.p,
#			self.h_final, method='linear')
#		final_T = spint.griddata(self.atmosphere.h, self.atmosphere.T,
#			self.h_final, method='linear')
#		final_mu = spint.griddata(self.atmosphere.h, self.atmosphere.mu,
#			self.h_final, method='linear')
#		final_p_dyn = fcl.p_dyn(rho=final_rho, V=final_V)
#		final_g = grav_sphere(self.g_0, self.R, 0)
#		final_a = spint.griddata(self.atmosphere.h, self.atmosphere.a,
#			self.h_final, method='linear')
#		final_Ma = final_V / final_a
#
#		self.final_values.time		= final_time
#		self.final_values.V         = final_V
#		self.final_values.gamma     = final_gamma
#		self.final_values.rho       = final_rho
#		self.final_values.p         = final_p
#		self.final_values.T         = final_T
#		self.final_values.mu        = final_mu
#		self.final_values.p_dyn     = final_p_dyn
#		self.final_values.g         = final_g
#		self.final_values.g         = final_a
#		self.final_values.Ma        = final_Ma
#		self.final_values.h 		= self.h_final

		self.final_values = placeholder()
		final_list = interpolate_event(self, 0, self.l)

		for index, val in enumerate(self.l):
			#self.final_values.__dict__[val] = final_list[index]
			self.final_values.__dict__.update({val : final_list[index]})

		if self.console_output == True:
			print('END EVENT CONDITIONS CALCULATED')

#		return [final_time, final_V, final_gamma, final_rho, final_p, final_T,
#					final_mu, final_p_dyn, final_g, final_a, final_Ma]

		return None

	def final_step_assign(self):
		# Assign values calculated for final step
#		self.solver_time[self.index] 	= self.final_values.time
#		self.V[self.index] 			= self.final_values.V
#		self.gamma[self.index] 		= self.final_values.gamma
#		self.solver_rho[self.index] 	= self.final_values.rho
#		self.solver_p[self.index] 		= self.final_values.p
#		self.solver_T[self.index] 		= self.final_values.T
#		self.solver_mu[self.index] 	= self.final_values.mu
#		self.p_dyn[self.index] 		= self.final_values.p_dyn
#		self.g[self.index] 			= self.final_values.g
#		self.solver_a[self.index] 		= self.final_values.g
#		self.Ma[self.index] 			= self.final_values.Ma
#		self.h[self.index]			= self.final_values.h

		assign(self, self.final_values, self.i, self.l)

		return None
	
class trajectory_ascent:
	"""
	Ascent trajectory calculator.  Lift, drag, thrust, and gravitational forces are
	considered in this model.  The integration variable for differential
	equations is time.  Atmospheric models are queried on a step-by-step
	basis to update variables.
	"""
	def __init__(self, vehicle, atmosphere, planet, gamma_init, alpha_init, V_init, 
		h_init, h_final, steps, console_output=True, console_interval=100):
		# NB: vehicle should be an instance of the class 'spacecraft'

		# Verbose solver output flag
		self.console_output = console_output
		
		# Number of console outputs (# of steps / console_interval)
		self.console_interval = console_interval

		# Import atmospheric model
		self.atmosphere = atmosphere #atmosphere(self.h)

		# Copy altitude array for convenience
		#self.h = self.atmosphere.h
		self.steps_atm = self.atmosphere.steps
		self.steps_storage = np.int(steps)#self.steps * num

		# Import spacecraft entering atmosphere
		self.spacecraft = vehicle
		
		# Import planet properties
		self.planet = planet

		# Set astronomical constants
		self.R = self.planet.R
		self.g_0 = self.planet.g_0

		# Set initial values
		self.gamma_init = gamma_init
		self.alpha_init = alpha_init
		self.V_init = V_init
		self.h_init = h_init #self.atmosphere.h[0]
		self.h_final = h_final #self.atmosphere.h[-1]

		# Define integration points in h
		#self.h = h #np.linspace(h_init, h_end, steps)
		#self.del_h = np.abs(self.h[1] - self.h[0])

		# Calculate variance in gravitational acceleration using inverse
		# square law
		#self.g = grav_sphere(self.g_0, self.R, self.h)

		# Pre-allocate memory for iterative trajectory calculations
		self.V 		= np.zeros(self.steps_storage)
		self.gamma 	= np.zeros(self.steps_storage)
		self.alpha 	= np.zeros(self.steps_storage)
		self.r 		= np.zeros(self.steps_storage)
		self.p_dyn	= np.zeros(self.steps_storage)
		self.Ma  	= np.zeros(self.steps_storage)
		self.Kn  	= np.zeros(self.steps_storage)
		self.Re  	= np.zeros(self.steps_storage)
		self.mfp  	= np.zeros(self.steps_storage)
		self.h 	 	= np.zeros(self.steps_storage)
		self.g 	 	= np.zeros(self.steps_storage)

		self.energyKinetic	= np.zeros(self.steps_storage)
		self.energyPotential 	= np.zeros(self.steps_storage)
		self.energyOrbitalSpecific	= np.zeros(self.steps_storage)

		# Create empty arrays for storage of results from ODE solver
		self.sol 		= np.zeros([self.steps_storage, 4])
		self.solver_time = np.zeros(self.steps_storage)
		self.solver_rho 	= np.zeros(self.steps_storage)
		self.solver_mu 	= np.zeros(self.steps_storage)
		self.solver_a 	= np.zeros(self.steps_storage)
		self.solver_p 	= np.zeros(self.steps_storage)
		self.solver_T 	= np.zeros(self.steps_storage)
		self.solver_Cp	= np.zeros(self.steps_storage)
		self.solver_Cv 	= np.zeros(self.steps_storage)
		self.y_input 	= np.zeros([self.steps_storage, 4])

		# Define list of keys for obejct dict (self.__dict__)
		# To be used by truncation, event interpolation, and variable
		# assignment functions
		self.l = ['V', 'p_dyn', 'g', 'gamma', 'Ma', 'Kn', 'Re', 'h', 'r',
			'solver_time', 'solver_rho', 'solver_p', 'solver_T', 'solver_mu',
			'solver_a', 'mfp', 'solver_Cp', 'solver_Cv', 'energyKinetic',
			'energyPotential', 'energyOrbitalSpecific', 'alpha', 'F_D', 'F_L', 
			'F_T']

		# Set up simulation termination flags
		self.out_of_bounds_error = False

		return None

	def initialise(self):
		self.h[0] = self.h_init
		self.V[0] = self.V_init
		self.gamma[0] = self.gamma_init
		self.alpha[0] = self.alpha_init
		self.g[0] = grav_sphere(self.g_0, self.R, self.h_init)

		self.solver_rho[0], self.solver_a[0], self.solver_p[0], \
			self.solver_T[0], self.solver_mu[0], self.solver_Cp[0], \
			self.solver_Cv[0] = \
			interpolate_atmosphere(self, self.h_init)

		self.p_dyn[0] = fcl.p_dyn(rho=self.solver_rho[0], V=self.V[0])
		self.Ma[0] = self.V[0] / self.solver_a[0]
		self.mfp[0] = fcl.mean_free_path(self.solver_T[0], self.solver_p[0],
			self.atmosphere.d)
		self.Kn[0] = self.mfp[0] / self.spacecraft.L
		self.Re[0] = fcl.Reynolds(self.solver_rho[0], self.V[0],
			self.spacecraft.L, self.solver_mu[0])

#		self.F_D[0] = fcl.aero_force(self.solver_rho[0], self.V[0], \
#			self.spacecraft.Cd[0], self.spacecraft.A)
#		self.F_L[0] = fcl.aero_force(self.solver_rho[0], self.V[0], \
#			self.spacecraft.Cl[0], self.spacecraft.A)
#		self.F_T[0] = self.spacecraft.thrust_dat(0)

		self.mu = calculateGravitationalParameter(self.spacecraft.m, \
			self.planet.m)
		
		self.energyKinetic[0] = calculateKineticEnergy(self.spacecraft.m, 
			self.V_init)
		self.energyPotential[0] = calculatePotentialEnergy(self.spacecraft.m, 
			self.mu, self.h[0], self.planet.R)
		self.energyOrbitalSpecific[0] = calculateSpecificOrbitalEnergy(\
			self.energyKinetic[0], self.energyPotential[0], self.spacecraft.m,\
			self.gamma_init)

#		self.spacecraft.Cd[0] = self.spacecraft.aero_dat(self.Ma[0], self.Re[0], self.Kn[0])[0]
#		self.spacecraft.Cl[0] = self.spacecraft.aero_dat(self.Ma[0], self.Re[0], self.Kn[0])[1]
#		self.spacecraft.Cs[0] = self.spacecraft.aero_dat(self.Ma[0], self.Re[0], self.Kn[0])[2]

		#self.dvdt[0] = dv_dt(self.g[0], self.p_dyn[0], \
		#	self.spacecraft.ballistic_coeff, self.V[0], self.gamma[0])
		#self.dgdt[0] = dgamma_dt(self.gamma[0], self.g[0], self.V[0], self.R, self.h[0])
		#self.dhdt[0] = dh_dt(self.gamma[0], self.V[0])
		#self.drdt[0] = dr_dt(self.R, self.gamma[0], self.h[0])

		if self.console_output == True:
#			print('\033[1;32mGreen like Grass\033[1;m')
			print('\033[1;34m=== TRAJ_CALC ===\033[1;m')
			print('\033[1;34mMODEL INITIALISED.  INITIAL STEP COUNT: %i\033[1;m' % self.steps_storage)

		return None

#	def extend(self):
#		# Extend solver time array
#		self.time_steps = np.append(self.time_steps, self.time_steps[-1] +
#			(self.dt * np.ones(self.steps/10)))
#
#		# Extend solver time array
#		self.solver_time = np.append(self.solver_time, np.zeros(self.steps/10))
#
#		# Extend solver input array
#		self.y_input = np.vstack([self.y_input, np.zeros([self.steps/10, 4])])
#
#		# Extend dynamic pressure array
#		self.p_dyn = np.append(self.p_dyn, np.zeros(self.steps/10))
#
#		# Extend gravitational acceleration array
#		self.g = np.append(self.g, np.zeros(self.steps/10))
#
#		# Extend interpolated atmospheric arrays
#		self.solver_rho = np.append(self.solver_rho, np.zeros(self.steps/10))
#		self.solver_a = np.append(self.solver_a, np.zeros(self.steps/10))
#		self.solver_mu = np.append(self.solver_mu, np.zeros(self.steps/10))
#		self.solver_p = np.append(self.solver_p, np.zeros(self.steps/10))
#		self.solver_T = np.append(self.solver_T, np.zeros(self.steps/10))
#
#		# Extend Mach number array
#		self.Ma = np.append(self.Ma, np.zeros(self.steps/10))
#
#		# Extend solution array with zeros to avoid repeated use of append function
#		self.sol = np.vstack([self.sol, np.zeros([self.steps/10, 4])])
#
#		# Extend trajectory parameter arrays
#		self.V = np.append(self.V, np.zeros(self.steps/10))
#		self.gamma = np.append(self.gamma, np.zeros(self.steps/10))
#		self.h = np.append(self.h, np.zeros(self.steps/10))
#		self.r = np.append(self.r, np.zeros(self.steps/10))
#
#		if self.console_output == True:
#			print('STEP COUNT LIMIT REACHED.  EXTENDING SOLUTION BY %i STEPS' % self.steps/10)
#
#		return None

	def truncate(self):
		# Truncate solution arrays to remove trailing zeros (from unused elements)
		truncate(self, self.index+1, self.l)

		return None

	def final_step_event(self):
		# Interpolation routine to find conditions at time of final step
		# i.e. when h = h_final
		self.final_values = placeholder()
		final_list = interpolate_event(self, self.h[self.index], self.l)

		for index, val in enumerate(self.l):
			#self.final_values.__dict__[val] = final_list[index]
			self.final_values.__dict__.update({val : final_list[index]})

		if self.console_output == True:
			print('\033[1;34mEND EVENT CONDITIONS CALCULATED\033[1;m')

		return None

	def final_step_assign(self):
		# Assign values calculated for final step
		assign(self, self.final_values, self.i, self.l)
		
		return None

	def simulate_dopri(self, dt=1E-2):
		"""
		Run trajectory calculations using explicit Runge-Kutta method of order
		4(5) from Dormand & Prince
		"""
		# Set timestep for ODE solver
		self.dt = dt
		self.time_steps = np.cumsum(self.dt * np.ones(self.steps_storage))

		# Create ODE object from SciPy using Dormand-Prince RK solver
		self.eq = integrate.ode(traj_3DOF_dt).set_integrator('dop853', nsteps=1E8,
			rtol=1E-10)

		# Set initial conditions
		y_init = [self.V_init, self.gamma_init, self.h_init, self.r[0]]
		self.eq.set_initial_value(y_init, t=self.time_steps[0])

		# Generate counter
		index = 1
		self.index = index

		# Solve ODE system using conditional statement based on altitude
		while self.h[index-1] > 0:

			# Update ODE solver parameters from spacecraft object and
			# atmospheric model at each separate time step
			if self.spacecraft.aero_coeffs_type == 'CONSTANT':
				params = [self.R, self.g[index-1], self.spacecraft.ballistic_coeff,
					self.solver_rho[index-1], self.spacecraft.Cl, self.spacecraft.Cd]
				self.eq.set_f_params(params)

			elif self.spacecraft.aero_coeffs_type == 'VARIABLE':
				self.spacecraft.update_aero(self.index, self.Re[index-1],
					self.Ma[index-1], self.Kn[index-1], self.solver_p[index-1],
					self.p_dyn[index-1], self.solver_rho[index-1],
					(self.solver_Cp[index-1] / self.solver_Cv[index-1]),
					self.spacecraft.Cd[index-1], self.spacecraft.Cl[index-1])
				
				params = [self.R, self.g[index-1], self.spacecraft.ballistic_coeff[index-1],
					self.solver_rho[index-1], self.spacecraft.Cl[index-1],
					self.spacecraft.Cd[index-1]]
				
				self.eq.set_f_params(params)

			# Solve ODE system (sol[V, gamma, h, r])
			self.sol[index, :] = self.eq.integrate(self.time_steps[index])

			# Unpack ODE solver results into storage structures
			self.V[index] = self.sol[index, 0]
			self.gamma[index] = self.sol[index, 1]
			self.h[index] = self.sol[index, 2]
			self.r[index] = self.sol[index, 3]

			# Interpolate for freestream density in atmosphere model
			# (this avoids a direct call to an atmosphere model, allowing more
			# flexibility when coding as different models have different interfaces)
			self.solver_rho[index], self.solver_a[index], \
				self.solver_p[index], self.solver_T[index], \
				self.solver_mu[index], self.solver_Cp[index], \
				self.solver_Cv[index] = \
				interpolate_atmosphere(self, self.h[index])

			# Calculate energies
			self.energyKinetic[index] = calculateKineticEnergy( \
				self.spacecraft.m, self.V[index])
			self.energyPotential[index] = calculatePotentialEnergy( \
				self.spacecraft.m, self.mu, self.h[index], self.planet.R)
			self.energyOrbitalSpecific[index] = calculateSpecificOrbitalEnergy(\
				self.energyKinetic[index], self.energyPotential[index], \
				self.spacecraft.m, self.gamma[index])

			# Calculate gravitational acceleration at current altitude
			self.g[index] = grav_sphere(self.g_0, self.R, self.h[index])

			# Calculate dynamic pressure iteration results
			self.p_dyn[index] = fcl.p_dyn(rho=params[3], V=self.sol[index, 0])

			# Calculate Mach, Knudsen, and Reynolds numbers
			self.Ma[index] = self.V[index] / self.solver_a[index]
			self.mfp[index] = fcl.mean_free_path(self.solver_T[index],
				self.solver_p[index], self.atmosphere.d)
			self.Kn[index] = self.mfp[index] / self.spacecraft.L
			self.Re[index] = fcl.Reynolds(self.solver_rho[index],
				self.V[index], self.spacecraft.L, self.solver_mu[index])

			# Save inputs for inspection
			self.solver_time[index] = self.eq.t
			self.y_input[index, :] = self.eq.y

			# Advance iteration counter
			index += 1
			self.index = index

			# Check if solution storage array has reached maximum size
			if index == len(self.sol)-10:
				self.extend()

			#print(index)
			# Print solution progress to check for stability
			if self.console_output == True:
				if np.mod(index, self.steps_storage/self.console_interval) == 0:
					print('\033[1;31mITER: \033[1;m' \
					'\033[1;37m%i; \033[1;m' \
					'\033[1;32mALT: \033[1;m' \
					'\033[1;37m%3.2f km; \033[1;m' \
					'\033[1;36mORBITAL ENERGY: \033[1;m' \
					'\033[1;37m%3.2e MJ/kg\033[1;m' % \
					(index, self.h[index-1]/1E3, \
					self.energyOrbitalSpecific[index-1]/1E6))

			# Check for atmospheric model interpolation errors
			# (OUT_OF_BOUNDS error)
			error_out_of_bounds(self, self.index)
			if self.out_of_bounds_error == True:
				break
			else:
				pass

		if (self.out_of_bounds_error == False):
			print('\033[1;32m=== SIMULATION COMPLETE ===\033[1;m')

		# Subtract 1 from counter so that indexing is more convenient later on
		self.index -= 1

		# Truncate solution arrays to remove trailing zeros
		self.truncate()

		# Perform final step calculations for p_dyn, g, etc.
		self.final_step_event()

		print('\033[1;34mTRAJECTORY COMPUTED (RK 4/5)\033[1;m')
		print('\033[1;34m%i ITERATIONS, TIMESTEP = %f s, TOTAL TIME = %f s\033[1;m' % \
			(self.index, self.dt, self.solver_time[self.index-1]))

		return [self.sol, self.h, self.y_input, self.p_dyn, self.Ma]

	def calculate_heating(self):
		# Generate new placeholder class for all heat fluxes
		self.qdot 	= placeholder()

		# Generate blank classes for different heat flux mechanisms
		self.qdot.conv 	= placeholder()
		self.qdot.rad 	= placeholder()

		# Generate empty arrays for different correlations
		self.qdot.conv.dh 	= np.zeros(self.index+1)
		self.qdot.conv.bj 	= np.zeros(self.index+1)
		self.qdot.conv.s 	= np.zeros(self.index+1)
		self.qdot.conv.fr 	= np.zeros(self.index+1)
		self.qdot.conv.sg 	= np.zeros(self.index+1)
		self.qdot.rad.bj 	= np.zeros(self.index+1)
		self.qdot.rad.s 		= np.zeros(self.index+1)
		self.qdot.rad.ts 	= np.zeros(self.index+1)

		# Call heat_flux_lib for actual calculations
		# Detra-Hidalgo
		self.qdot.conv.dh = hcl.detra_hidalgo(self.V, self.solver_rho,
			self.spacecraft.R_n)

		# Brandis-Johnston (convective)
		for index, val in enumerate(self.h):
			self.qdot.conv.bj[index] = hcl.brandis_johnston(self.V[index],
				self.solver_rho[index], self.spacecraft.R_n, mode='conv')

		# Smith (convective)
		self.qdot.conv.s = hcl.smith(self.V, self.solver_rho,
			self.spacecraft.R_n, mode='conv', planet='Earth')

		# Fay-Riddell
		#self.qdot.conv.fr = hcl.fay_riddell()

		# Sutton-Graves (convective)
		self.qdot.conv.sg	= hcl.sutton_graves(self.V, self.solver_rho,
			self.spacecraft.R_n)

		# Brandis-Johnston (radiative)
		for index, val in enumerate(self.h):
			self.qdot.rad.bj[index] = hcl.brandis_johnston(self.V[index],
				self.solver_rho[index], self.spacecraft.R_n, mode='rad')

		# Smith (radiative)
		self.qdot.rad.bj = hcl.smith(self.V, self.solver_rho,
			self.spacecraft.R_n, mode='rad', planet='Earth')

		# Tauber-Sutton (radiative)
		rho_ratio = fcl.normal_shock_ratios(self.Ma, self.solver_Cp / \
			self.solver_Cv)[4]
		for index, val in enumerate(self.h):
			self.qdot.rad.ts[index] = hcl.tauber_sutton(self.V[index],
				self.solver_rho[index], self.spacecraft.R_n, rho_ratio[index])

		# Net flux (convective heating, radiative cooling)
		self.qdot.net = placeholder()
		self.qdot.net.bj = self.qdot.conv.bj + self.qdot.rad.bj
		self.qdot.net.s = self.qdot.conv.s + self.qdot.rad.s
		self.qdot.net.sg_ts = self.qdot.conv.sg + self.qdot.rad.ts

	def plot_triple(self):

		fig = plt.figure(figsize=[12, 10])
		ax1 = fig.add_subplot(224)
		ax2 = fig.add_subplot(223)
		ax4 = fig.add_subplot(211)

		#alt = self.h / 1000
		#g_range = self.r / 1000
		#gamma_deg = np.rad2deg(self.gamma)
		line_width = 2.0
		num = len(self.h) / 20

		quiv = np.array([self.r[0:-1:num]/1000, self.h[0:-1:num]/1000, \
			self.gamma[0:-1:num]]).T

		quiver_x = np.log(self.V[0:-1:num]) * np.cos(quiv[:, 2])
		quiver_y = -np.log(self.V[0:-1:num]) * np.sin(quiv[:, 2])

		ax4.plot((self.r/1000), (self.h/1000), color='c', linewidth=line_width, alpha=1.0)
		ax4.scatter(quiv[:, 0], quiv[:, 1], color='b', s=15, alpha=0.25, zorder=3)
		ax4.scatter(self.r[-1]/1000, self.h[-1]/1000, s=300, marker='x',
			color='r', alpha=0.3, zorder=3)
		quiv_plot = ax4.quiver(quiv[:, 0], quiv[:, 1], quiver_x, quiver_y, \
			scale=100, linewidth=1, edgecolor='b', headwidth=10, headlength=10,\
			width=0.001, alpha=0.2)
		ax4.quiverkey(quiv_plot, 0.9, 0.9, 5, 'log(Velocity), $log(V)$ [m/s]', color='b')
		ax4.set_xlabel('Ground range, r (km)')
		ax4.set_ylabel('Altitude, h (km)')
		ax4.grid(True)

		#ax0.plot(g_range, alt, color='g', linewidth=line_width)
		#ax0.set_xlabel('Ground range, r (m)')
		#ax0.set_ylabel('Altitude, h (km)')
		#ax0.grid(True)

		ax1.plot(self.Ma, self.h/1000, color='r', linewidth=line_width)
		ax1.scatter(self.Ma[-1], self.h[-1]/1000, s=300, marker='x',
			color='r', alpha=0.5, zorder=3)
		ax1.set_xlabel('Mach number, Ma')
		ax1.set_ylabel('Altitude, h (km)')
		ax1.set_xscale('log')
		ax1.grid(True)

		ax2.plot(self.V, self.h/1000, color='b', linewidth=line_width)
		ax2.scatter(self.V[-1], self.h[-1]/1000, s=300, marker='x',
			color='r', alpha=0.5, zorder=3)
		ax2.set_xlabel('Velocity, V (m/s)')
		ax2.set_ylabel('Altitude, h (km)')
		ax2.set_xscale('log')
		ax2.grid(True)

		#ax3.plot(gamma_deg, alt, color='g', linewidth=line_width)
		#ax3.set_xlabel(r'Flight path angle, $\gamma$ (degrees)')
		#ax3.set_ylabel('Altitude, h (km)')
		#ax3.grid(True)

		plt.tight_layout()

		return None

	def plot_trajectory(self):
		fig = plt.figure(figsize=[12, 6])
		ax4 = fig.add_subplot(111)

		#alt = self.h / 1000
		#g_range = self.r / 1000
		#gamma_deg = np.rad2deg(self.gamma)
		line_width = 2.0
		num = len(self.h) / 20

		quiv = np.array([self.r[0:-1:num]/1000, self.h[0:-1:num]/1000, \
			self.gamma[0:-1:num]]).T

		quiver_x = np.log(self.V[0:-1:num]) * np.cos(quiv[:, 2])
		quiver_y = -np.log(self.V[0:-1:num]) * np.sin(quiv[:, 2])

		ax4.plot((self.r/1000), (self.h/1000), color='c', linewidth=line_width, alpha=1.0)
		ax4.scatter(quiv[:, 0], quiv[:, 1], color='b', s=15, alpha=0.25, zorder=3)
		ax4.scatter(self.r[-1]/1000, self.h[-1]/1000, s=300, marker='x',
			color='r', alpha=0.3, zorder=3)
		quiv_plot = ax4.quiver(quiv[:, 0], quiv[:, 1], quiver_x, quiver_y, \
			scale=100, linewidth=1, edgecolor='b', headwidth=10, headlength=10,\
			width=0.001, alpha=0.2)
		ax4.quiverkey(quiv_plot, 0.85, 0.9, 5, 'log(Velocity), $log(V)$ [m/s]', color='b')
		ax4.set_xlabel('Ground range, r (km)')
		ax4.set_ylabel('Altitude, h (km)')
		ax4.grid(True)

		#ax0.plot(g_range, alt, color='g', linewidth=line_width)
		#ax0.set_xlabel('Ground range, r (m)')
		#ax0.set_ylabel('Altitude, h (km)')
		#ax0.grid(True)

		plt.tight_layout()

		return None

	def show_regimes(self):

		handle = plt.gcf()
		xspan = np.abs(handle.axes[-1].get_xlim()[0]) + \
			np.abs(handle.axes[-1].get_xlim()[1])
		xlim = handle.axes[-1].get_xlim()[0] + (0.02 * xspan)

		# REGIME HIGHLIGHTS
		reg_col = [[0, 1, 0],
						[0, 0.66, 0.33],
						[0, 0.33, 0.66],
						[0, 0, 1]]

		# Continuum
		if hasattr(self.regimes, 'continuum'):
			plt.axhspan(self.h[self.regimes.continuum[0]]/1000,
				self.h[self.regimes.continuum[-1]]/1000,
				facecolor=reg_col[0], alpha=0.05)
			plt.text(xlim, self.h[self.regimes.continuum[-1]]/1000 + 1,
				'CONTINUUM', fontsize=12, \
			    color=reg_col[0], alpha=0.5)

		# Slip
		if hasattr(self.regimes, 'slip'):
			plt.axhspan(self.h[self.regimes.slip[0]]/1000,
				self.h[self.regimes.slip[-1]]/1000,
				facecolor=reg_col[1], alpha=0.05)
			plt.text(xlim, self.h[self.regimes.slip[-1]]/1000 + 1,
				'SLIP', fontsize=12, \
			    color=reg_col[1], alpha=0.5)

		# Transition
		if hasattr(self.regimes, 'transition'):
			plt.axhspan(self.h[self.regimes.transition[0]]/1000,
				self.h[self.regimes.transition[-1]]/1000,
				facecolor=reg_col[2], alpha=0.05)
			plt.text(xlim, self.h[self.regimes.transition[-1]]/1000 + 1,
				'TRANSITION', fontsize=12, \
			    color=reg_col[2], alpha=0.5)

		# Free molecular
		if hasattr(self.regimes, 'free_molecular'):
			plt.axhspan(self.h[self.regimes.free_molecular[0]]/1000,
				self.h[self.regimes.free_molecular[-1]]/1000,
				facecolor=reg_col[3], alpha=0.05)
			plt.text(xlim, self.h[self.regimes.free_molecular[-1]]/1000 + 1,
				'FREE MOLECULAR', fontsize=12, \
			    color=reg_col[3], alpha=0.5)

		# REGIME BOUNDARY LINES
		# Continuum-Slip
		if (self.index_cont_slip != None):
			plt.axhline(y=self.h[self.index_cont_slip]/1000,
				color=reg_col[1], alpha=0.2, zorder=0, linewidth=2.0)

		# Slip-Transition
		if (self.index_slip_tran != None):
			plt.axhline(y=self.h[self.index_slip_tran]/1000,
				color=reg_col[2], alpha=0.2, zorder=0, linewidth=2.0)

		# Transition-Free molecular
		if (self.index_tran_freemol != None):
			plt.axhline(y=self.h[self.index_tran_freemol]/1000,
				color=reg_col[3], alpha=0.2, zorder=0, linewidth=2.0)

		return None

	def post_calc(self):
		"""
		Perform post-integration calculations (such as Knudsen and Reynolds
		numbers, location of regimes, etc.)
		"""

#		self.mfp = fcl.mean_free_path(self.solver_T, self.solver_p,
#			self.atmosphere.d)
#		self.Kn = self.mfp / self.spacecraft.L
##		self.Re = fcl.KnReMa(self.atmosphere.k, Kn=self.Kn,
##			Ma=self.Ma)
#		self.Re = fcl.Reynolds(self.solver_rho, self.V, self.spacecraft.L,
#			self.solver_mu)

		# Continuum: 0 < Kn < 0.001
		# Slip: 0.001 <= Kn < 0.1
		# Transition: 0.1 <= Kn < 10
		# Free molecular: 10 < Kn

		self.regimes = placeholder()

		if len(np.argwhere(self.Kn > 10)) != 0:
			self.index_tran_freemol = np.argwhere(self.Kn > 10)[-1]
			self.regimes.free_molecular = np.argwhere(self.Kn >= 10)
		else:
			self.index_tran_freemol = None

		if len(np.argwhere(self.Kn > 0.1)) != 0:
			self.index_slip_tran = np.argwhere(self.Kn > 0.1)[-1]
			self.regimes.transition = np.argwhere((self.Kn < 10) & (self.Kn >= 0.1))
		else:
			self.index_slip_tran = None

		if len(np.argwhere(self.Kn > 0.001)) != 0:
			self.index_cont_slip = np.argwhere(self.Kn > 0.001)[-1]
			self.regimes.slip = np.argwhere((self.Kn < 0.1) & (self.Kn >= 0.001))
		else:
			self.index_cont_slip = None

		if len(np.argwhere((self.Kn > 0) & (self.Kn <= 0.001))) != 0:
			self.regimes.continuum = np.argwhere((self.Kn < 0.001) & (self.Kn >= 0))
		else:
			self.index_cont_slip = None

		return [self.mfp, self.Kn, self.Re]

	def check_for_out_of_bounds_error(self):
		check_a = np.isnan(self.solver_a)
		check_rho = np.isnan(self.solver_rho)
		sum_error_a = np.sum(check_a)
		sum_error_rho = np.sum(check_rho)

		if (sum_error_a != 0) or (sum_error_rho != 0):
			print('NaN entries found in atmospheric interpolation model.  Try expanding altitude bounds.')
		else:
			print('No NaN entries found in atmospheric interpolation model.')
		return None

	def export_dict(self):
		"""
		Export first level class contents as Python dictionary object
		(inherited classes are not exported)
		"""

		t = dict(self.__dict__)

		del t['regimes']
		del t['qdot']
		del t['atmosphere']
		del t['final_values']
		del t['spacecraft']
		del t['eq']

		return [t, qdot, regimes, atmosphere, spacecraft, eq]