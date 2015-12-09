# -*- coding: utf-8 -*-
"""
=== TRAJ CALC ===
Re-entry trajectory calculator
"""

try:
	import numpy as np
	import aerocalc.std_atm as atm
	import flow_calc_lib as fcl
	import heat_flux_lib as hcl
	from scipy import integrate
	import cantera as ct
	import nrlmsise_00_header as nrl_head
	import nrlmsise_00 as nrlmsise
	import j77sri as j77
	import matplotlib.pyplot as plt
	import scipy.interpolate as spint
	import rotate
except:
	print 'ERROR: Dependencies are not satisfied'

__author__ = 'Nathan Donaldson'
__contributor__ = 'Hilbert van Pelt'
__email__ = 'nathan.donaldson@eng.ox.ac.uk'
__status__ = 'Release'
__version__ = '1.4'
__license__ = 'MIT'

# Time derivatives for forward Euler solver and ODE solver initialisation
# Velocity
def dv_dt(g, p_dyn, beta, gamma):
	dvdt = (-p_dyn / beta) + (g * np.sin(gamma))
	return dvdt

# Flightpath angle
def dgamma_dt(gamma, g, V, R, h, L_D_ratio, p_dyn, beta):
	dgdt = (((-p_dyn / beta) * L_D_ratio) + ((np.cos(gamma)) * (g - ((V**2) / (R + h))))) / V
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

def traj_3DOF_dt(t, y, params):
	# Function to be called by ODE solver when time integration of governing
	# equations is required
	V = y[0]
	gamma = y[1]
	h = y[2]
	r = y[3]

	R = params[0]
	g = params[1]
	beta = params[2]
	rho = params[3]
	C_L = params[4]
	C_D = params[5]

	L_D_ratio = C_L / C_D
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
	dy[0] = Fx / Ms - (mu * y[3]) / ((y[3]**2 + y[4]**2 + y[5]**3)**(3.0 / 2.0))
	dy[1] = Fy / Ms - (mu * y[4]) / ((y[3]**2 + y[4]**2 + y[5]**3)**(3.0 / 2.0))
	dy[2] = Fz / Ms - (mu * y[5]) / ((y[3]**2 + y[4]**2 + y[5]**3)**(3.0 / 2.0))

	# Position in X, Y and Z directions (respectively)
	dy[3] = y[0]
	dy[4] = y[1]
	dy[5] = y[2]

	return dy

def truncate(t, i, l):
	for index, item in enumerate(l):
		t.__dict__[item] = np.delete(t.__dict__[item], 
			np.arange(t.index+1, len(t.__dict__[item])))
			
	sol_temp = t.sol
	t.sol = np.zeros([t.index, 4])
	t.sol = sol_temp[0:t.index, :]
	
	return None
	
def interpolate_event(t, i, l):
	final_list = []
	for index, item in enumerate(l):
		final_list.append(spint.griddata(t.h, t.__dict__[item], 
			i, method='linear'))
			
	return final_list

class placeholder:
	def __init__(self):
		pass

class spacecraft:
	def __init__(self, Cd, m, A, R_n, L, Cl=0, Cs=0):
		self.A = A
		self.Cd = Cd
		self.Cl = Cl
		self.Cs = Cs
		self.R_n = R_n
		self.m = m
		self.L = L
		self.ballistic_coeff = (self.m) / (self.Cd * self.A)
		
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

		print 'ATMOSPHERIC MODEL COMPUTED (US76)'

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
		lst=16, f107A=150, f107=150, ap=4):

		# Cantera Solution object
		self.gas = ct.Solution('air.xml')

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
		self.flags.switches[0] = 0

		# Set other flags to TRUE (see docstring of nrlmsise_00_header.py)
		for index in range(24):
			self.flags.switches[index] = 1

		for index in range(self.steps):
			self.input[index].doy = doy
			self.input[index].year = year
			self.input[index].sec = sec
			self.input[index].alt = self.h[index] / 1000
			self.input[index].g_lat = g_lat
			self.input[index].g_long = g_long
			self.input[index].lst = lst
			self.input[index].f107A = f107A
			self.input[index].f107 = f107
			self.input[index].ap = ap
			#self.input[index].ap_a = self.aph

		# Run NRLMSISE00 model
		for index in range(self.steps):
			nrlmsise.gtd7(self.input[index], self.flags, self.output[index])

		# Pre-allocate memory
		self.rho = np.zeros(self.steps)
		self.T = np.zeros(self.steps)
		self.a = np.zeros(self.steps)
		self.k = np.zeros(self.steps)
		self.mu = np.zeros(self.steps)
		self.cp = np.zeros(self.steps)
		self.cv = np.zeros(self.steps)

		# Extract density and temperature
		for index in range(self.steps):
			self.rho[index] = self.output[index].d[5]
			self.T[index] = self.output[index].t[1]

		# Query Cantera for gas state
		for index, alt in enumerate(self.h):
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

		print 'ATMOSPHERIC MODEL COMPUTED (NRLMSISE00)'

		return None

class trajectory_lifting:
	"""
	Lifting trajectory calculator.  Lift, drag and gravitational forces are
	considered in this model.  The integration variable for differential
	equations is time.  Atmospheric models are interpolated on a step-by-step
	basis to update variables.
	"""
	def __init__(self, vehicle, atmosphere, gamma_init, V_init, g_0, R, h_init,
		h_final, steps, console_output=True):
		# NB: vehicle should be an instance of the class 'spacecraft'

		# Verbose solver output flag
		self.console_output = console_output

		# Import atmospheric model
		self.atmosphere = atmosphere #atmosphere(self.h)

		# Copy altitude array for convenience
		#self.h = self.atmosphere.h
		self.steps_atm = self.atmosphere.steps
		self.steps_storage = steps#self.steps * num

		# Import spacecraft entering atmosphere
		self.spacecraft = vehicle

		# Set astronomical constants
		self.R = R
		self.g_0 = g_0

		# Set initial values
		self.gamma_init = gamma_init
		self.V_init = V_init
		self.h_init = h_init 
		self.h_final = h_final 

		# Pre-allocate memory for iterative trajectory calculations
		self.V 		= np.zeros(self.steps_storage)
		self.gamma 	= np.zeros(self.steps_storage)
		self.r 		= np.zeros(self.steps_storage)
		self.p_dyn	= np.zeros(self.steps_storage)
		self.Ma  	= np.zeros(self.steps_storage)
		self.Kn  	= np.zeros(self.steps_storage)
		self.Re  	= np.zeros(self.steps_storage)
		self.mfp  	= np.zeros(self.steps_storage)
		self.h 	 	= np.zeros(self.steps_storage)
		self.g 	 	= np.zeros(self.steps_storage)

		# Create empty arrays for storage of results from ODE solver
		self.sol 		= np.zeros([self.steps_storage, 4])
		self.solver_time 	= np.zeros(self.steps_storage)
		self.solver_rho 	= np.zeros(self.steps_storage)
		self.solver_mu 	= np.zeros(self.steps_storage)
		self.solver_a 	= np.zeros(self.steps_storage)
		self.solver_p 	= np.zeros(self.steps_storage)
		self.solver_T 	= np.zeros(self.steps_storage)
		self.y_input 		= np.zeros([self.steps_storage, 4])

		# Define list of keys for obejct dict (self.__dict__)
		# To be used by truncation, event interpolation, and variable
		# assignment functions 
		self.l = ['V', 'p_dyn', 'g', 'gamma', 'Ma', 'Kn', 'Re', 'h', 'r',
			'solver_time', 'solver_rho', 'solver_p', 'solver_T', 'solver_mu', 
			'solver_a', 'mfp']

		return None

	def initialise(self):
		self.h[0] = self.h_init
		self.V[0] = self.V_init
		self.gamma[0] = self.gamma_init
		self.g[0] = grav_sphere(self.g_0, self.R, self.h_init)

		self.solver_rho[0], self.solver_a[0], self.solver_p[0], \
			self.solver_T[0], self.solver_mu[0] = \
			self.interpolate_atmosphere(self.h_init)

		self.p_dyn[0] = fcl.p_dyn(rho=self.solver_rho[0], V=self.V[0])
		self.Ma[0] = self.V[0] / self.solver_a[0]
		self.mfp[0] = fcl.mean_free_path(self.solver_T[0], self.solver_p[0],
			self.atmosphere.d)
		self.Kn[0] = self.mfp[0] / self.spacecraft.L
		self.Re[0] = fcl.Reynolds(self.solver_rho[0], self.V[0], 
			self.spacecraft.L, self.solver_mu[0])

		if self.console_output == True:
			print 'MODEL INITIALISED.  INITIAL STEP COUNT: %i' % self.steps_storage

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
			print 'STEP COUNT LIMIT REACHED.  EXTENDING SOLUTION BY %i STEPS' % self.steps/10

		return None

	def truncate(self):
		# Truncate solution arrays to remove trailing zeros (from unused elements)
		self.V 			= np.delete(self.V, np.arange(self.index+1, len(self.V)))
		self.gamma 		= np.delete(self.gamma, np.arange(self.index+1, len(self.gamma)))
		self.h 			= np.delete(self.h, np.arange(self.index+1, len(self.h)))
		self.r 			= np.delete(self.r, np.arange(self.index+1, len(self.r)))
		self.p_dyn 		= np.delete(self.p_dyn, np.arange(self.index+1, len(self.p_dyn)))
		self.solver_time 	= np.delete(self.solver_time, np.arange(self.index+1, len(self.solver_time)))
		self.solver_rho 		= np.delete(self.solver_rho, np.arange(self.index+1, len(self.solver_rho)))
		self.solver_p 		= np.delete(self.solver_p, np.arange(self.index+1, len(self.solver_p)))
		self.solver_T 		= np.delete(self.solver_T, np.arange(self.index+1, len(self.solver_T)))
		self.solver_mu 		= np.delete(self.solver_mu, np.arange(self.index+1, len(self.solver_mu)))
		self.solver_a 		= np.delete(self.solver_a, np.arange(self.index+1, len(self.solver_a)))
		self.g 			= np.delete(self.g, np.arange(self.index+1, len(self.g)))
		self.Ma 			= np.delete(self.Ma, np.arange(self.index+1, len(self.Ma)))
		self.Kn 			= np.delete(self.Kn, np.arange(self.index+1, len(self.Kn)))
		self.Re 			= np.delete(self.Re, np.arange(self.index+1, len(self.Re)))
		self.mfp 			= np.delete(self.mfp, np.arange(self.index+1, len(self.mfp)))

		sol_temp = self.sol
		self.sol = np.zeros([self.index, 4])
		self.sol = sol_temp[0:self.index, :]
		return None

	def final_step_event(self):
		# Interpolation routine to find conditions at time of final step
		# i.e. when h = h_final
		self.final_values = placeholder()

		final_list = interpolate_event(self, self.h[self.index], self.l)

		for index, val in enumerate(self.l):
			self.final_values.__dict__.update({val : final_list[index]})

		if self.console_output == True:
			print 'END EVENT CONDITIONS CALCULATED'

#		return [final_time, final_V, final_gamma, final_rho, final_p, final_T,
#					final_mu, final_p_dyn, final_g, final_a, final_Ma]

		return None

	def final_step_assign(self):
		# Assign values calculated for final step
		self.solver_time[self.index] 	= self.final_values.time
		self.V[self.index] 				= self.final_values.V
		self.gamma[self.index] 			= self.final_values.gamma
		self.solver_rho[self.index] 	= self.final_values.rho
		self.solver_p[self.index] 		= self.final_values.p
		self.solver_T[self.index] 		= self.final_values.T
		self.solver_mu[self.index] 		= self.final_values.mu
		self.p_dyn[self.index] 			= self.final_values.p_dyn
		self.g[self.index] 				= self.final_values.g
		self.solver_a[self.index] 		= self.final_values.g
		self.Ma[self.index] 			= self.final_values.Ma
		self.h[self.index]			= self.final_values.h

		return None

	def interpolate_atmosphere(self, h_interp):
		rho_interp = spint.griddata(self.atmosphere.h, self.atmosphere.rho,
				h_interp, method='linear')
		a_interp = spint.griddata(self.atmosphere.h, self.atmosphere.a,
				h_interp, method='linear')
		p_interp = spint.griddata(self.atmosphere.h, self.atmosphere.p,
				h_interp, method='linear')
		T_interp = spint.griddata(self.atmosphere.h, self.atmosphere.T,
				h_interp, method='linear')
		mu_interp = spint.griddata(self.atmosphere.h, self.atmosphere.mu,
				h_interp, method='linear')

		return [rho_interp, a_interp, p_interp, T_interp, mu_interp]

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
			# Update parameters with atmospheric density at each altitude step
			params = [self.R, self.g[index-1], self.spacecraft.ballistic_coeff,
				self.solver_rho[index-1], self.spacecraft.Cl, self.spacecraft.Cd]
			self.eq.set_f_params(params)

			# Solve ODE system (sol[V, gamma, h, r])
			self.sol[index, :] = self.eq.integrate(self.time_steps[index])

			# Unpack ODE solver results into storage stuctures
			self.V[index] = self.sol[index, 0]
			self.gamma[index] = self.sol[index, 1]
			self.h[index] = self.sol[index, 2]
			self.r[index] = self.sol[index, 3]

			# Interpolate for freestream density in atmosphere model
			# (this avoids a direct call to an atmosphere model, allowing more
			# flexibility when coding as different models have different interfaces)
			self.solver_rho[index], self.solver_a[index], \
				self.solver_p[index], self.solver_T[index], \
				self.solver_mu[index] = self.interpolate_atmosphere(self.h[index])

			# Calculate gravitational acceleration at current altitude
			self.g[index] = grav_sphere(self.g_0, self.R, self.h[index])

			# Calculate dynamic pressure iteration results
			self.p_dyn[index] = fcl.p_dyn(rho=params[3], V=self.sol[index, 0])

			# Calculate Mach numbers
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

			#print index
			# Print solution progress to check for stability
			if self.console_output == True:
				if np.mod(index, self.steps_storage/100) == 0:
					print 'ITERATION: %i; ALTITUDE: %f km' % (index, self.h[index-1]/1000)

		# Subtract 1 from counter so that indexing is more convenient later on
		self.index -= 1

		# Truncate solution arrays to remove trailing zeros
		self.truncate()

		# Perform final step calculations for p_dyn, g, etc.
		self.final_step_event()
		#self.final_step_assign()

		# Perform post solver calculations
		#self.post_calc()

		print 'TRAJECTORY COMPUTED (RK 4/5)'
		print '%i ITERATIONS, TIMESTEP = %f s, TOTAL TIME = %f s' % \
			(self.index, self.dt, self.solver_time[self.index-1])

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
		self.qdot.conv.s 		= np.zeros(self.index+1)
		self.qdot.conv.fr 	= np.zeros(self.index+1)
		self.qdot.conv.sg 	= np.zeros(self.index+1)
		self.qdot.rad.bj 		= np.zeros(self.index+1)
		self.qdot.rad.s 		= np.zeros(self.index+1)
		self.qdot.rad.ts 		= np.zeros(self.index+1)

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
		for index, val in enumerate(self.h):
			self.qdot.rad.ts[index] = hcl.tauber_sutton(self.V[index],
				self.solver_rho[index], self.spacecraft.R_n)

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

		plt.tight_layout()

		return None

	def plot_trajectory(self):
		fig = plt.figure(figsize=[12, 6])
		ax4 = fig.add_subplot(111)

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

	def check_for_out_of_bounds_error(self):
		check_a = np.isnan(self.solver_a)
		check_rho = np.isnan(self.solver_rho)
		sum_error_a = np.sum(check_a)
		sum_error_rho = np.sum(check_rho)

		if (sum_error_a != 0) or (sum_error_rho != 0):
			print 'NaN entries found in atmospheric interpolation model.  Try expanding altitude bounds.'
		else:
			print 'No NaN entries found in atmospheric interpolation model.'
		return None

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

		self.solver_rho[0], self.solver_a[0], \
			self.solver_p[0], self.solver_T[0], \
			self.solver_mu[0] = self.interpolate_atmosphere(self.h[0])

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
		self.solver_rho[0], self.solver_a[0], \
			self.solver_p[0], self.solver_T[0], \
			self.solver_mu[0] = self.interpolate_atmosphere(self.h[0])

		self.lift[0] = fcl.aero_force(self.solver_rho[0], self.V_mag[0],
			self.spacecraft.Cl, self.spacecraft.A)
		self.drag[0] = fcl.aero_force(self.solver_rho[0], self.V_mag[0],
			self.spacecraft.Cd, self.spacecraft.A)
		self.side_force[0] = fcl.aero_force(self.solver_rho[0], self.V_mag[0],
			self.spacecraft.Cs, self.spacecraft.A)
		self.forces_rotating[0, :] = np.array([self.lift[0], self.drag[0],
			self.side_force[0]])

		# Calculate Euler angles (pitch and yaw; roll is assumed to be zero)
		self.alpha[0], self.theta[0] = rotate.vector_to_euler(self.pos_xyz[0, 0],
			self.pos_xyz[0, 1], self.pos_xyz[0, 2])

		# Transform aero forces from rotating to inertial frame
		self.forces_inertial[0, :] = rotate.roty(self.forces_rotating[0, :],
			self.alpha[0], mode='rad')
		self.forces_inertial[0, :] = rotate.rotz(self.forces_rotating[0, :],
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

		if self.console_output == True:
			print 'MODEL INITIALISED.  INITIAL STEP COUNT: %i' % self.N

		return None

	def interpolate_atmosphere(self, h_interp):
		rho_interp = spint.griddata(self.atmosphere.h, self.atmosphere.rho,
				h_interp, method='linear')
		a_interp = spint.griddata(self.atmosphere.h, self.atmosphere.a,
				h_interp, method='linear')
		p_interp = spint.griddata(self.atmosphere.h, self.atmosphere.p,
				h_interp, method='linear')
		T_interp = spint.griddata(self.atmosphere.h, self.atmosphere.T,
				h_interp, method='linear')
		mu_interp = spint.griddata(self.atmosphere.h, self.atmosphere.mu,
				h_interp, method='linear')

		return [rho_interp, a_interp, p_interp, T_interp, mu_interp]

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
				self.solver_mu[i] = self.interpolate_atmosphere(self.h[i])

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
			self.alpha[i], self.theta[i] = rotate.vector_to_euler(self.sol[i, 3],
				self.sol[i, 4], self.sol[i, 5])

			# Transform aero forces from rotating to inertial frame
			self.forces_inertial[i, :] = rotate.roty(self.forces_rotating[i, :],
				self.alpha[i], mode='rad')
			self.forces_inertial[i, :] = rotate.rotz(self.forces_rotating[i, :],
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
					print '%3.1f%%; ITERATION: %i; ALTITUDE: %f km' % \
						(100*(np.float(i)/self.N), i, self.h[i]/1000)

			# Check for ground strike
			if self.h[i] <= 0:
				print 'GROUND STRIKE EVENT (ALTITUDE = 0) DETECTED BETWEEN ' \
					'INDEXES %i AND %i' % (i-1, i)
				break
			
			# Check for atmospheric model interpolation errors
			# (OUT_OF_BOUNDS error)
			if np.isnan(self.solver_rho[i]) == True:
				print 'ERROR: ATMOSPHERIC INTERPOLATION OUT OF BOUNDS AT ' \
					'INDEX %i, TRY EXPANDING ALTITUDE RANGE' % i
				break

		self.final_step_event()

		return self.sol

	def check_for_out_of_bounds_error(self):
		check_a = np.isnan(self.solver_a)
		check_rho = np.isnan(self.solver_rho)
		sum_error_a = np.sum(check_a)
		sum_error_rho = np.sum(check_rho)

		if (sum_error_a != 0) or (sum_error_rho != 0):
			print 'NaN entries found in atmospheric interpolation model.  Try expanding altitude bounds.'
		else:
			print 'No NaN entries found in atmospheric interpolation model.'
		return None
		
	def truncate(self):			
		# Truncate solution arrays to remove trailing zeros (from unused elements)
		self.V_xyz		= np.delete(self.V_xyz, np.arange(self.index+1, len(self.V_xyz)))
		self.V_mag		= np.delete(self.V_mag, np.arange(self.index+1, len(self.V_mag)))
		self.F_mag		= np.delete(self.F_mag, np.arange(self.index+1, len(self.F_mag)))
		self.F_x			= np.delete(self.F_x, np.arange(self.index+1, len(self.F_x)))
		self.F_y			= np.delete(self.F_y, np.arange(self.index+1, len(self.F_y)))
		self.F_z			= np.delete(self.F_z, np.arange(self.index+1, len(self.F_z)))
		self.alpha		= np.delete(self.alpha, np.arange(self.index+1, len(self.alpha)))
		self.theta		= np.delete(self.theta, np.arange(self.index+1, len(self.theta)))
		self.drag		= np.delete(self.drag, np.arange(self.index+1, len(self.drag)))
		self.lift		= np.delete(self.lift, np.arange(self.index+1, len(self.lift)))
		self.side_force	= np.delete(self.side_force, np.arange(self.index+1, len(self.side_force)))
		self.pos_mag		= np.delete(self.pos_mag, np.arange(self.index+1, len(self.pos_mag)))
		self.pos_xyz		= np.delete(self.pos_xyz, np.arange(self.index+1, len(self.pos_xyz)))
		self.h 			= np.delete(self.h, np.arange(self.index+1, len(self.h)))
		self.r 			= np.delete(self.r, np.arange(self.index+1, len(self.r)))
		self.solver_time 	= np.delete(self.solver_time, np.arange(self.index+1, len(self.solver_time)))
		self.solver_rho 	= np.delete(self.solver_rho, np.arange(self.index+1, len(self.solver_rho)))
		self.solver_p 	= np.delete(self.solver_p, np.arange(self.index+1, len(self.solver_p)))
		self.solver_T 	= np.delete(self.solver_T, np.arange(self.index+1, len(self.solver_T)))
		self.solver_mu 	= np.delete(self.solver_mu, np.arange(self.index+1, len(self.solver_mu)))
		self.solver_a 	= np.delete(self.solver_a, np.arange(self.index+1, len(self.solver_a)))
		self.Ma 			= np.delete(self.Ma, np.arange(self.index+1, len(self.Ma)))
		self.Kn 			= np.delete(self.Kn, np.arange(self.index+1, len(self.Kn)))
		self.Re 			= np.delete(self.Re, np.arange(self.index+1, len(self.Re)))
		self.mfp 		= np.delete(self.mfp, np.arange(self.index+1, len(self.mfp)))
		self.force_inertial	= np.delete(self.force_inertial, 
			np.arange(self.index+1, len(self.force_inertial)))
		self.force_rotating	= np.delete(self.force_rotating, 
			np.arange(self.index+1, len(self.force_rotating)))

		sol_temp = self.sol
		self.sol = np.zeros([self.index, 4])
		self.sol = sol_temp[0:self.index, :]
		return None
		
	def final_step_event(self):
		# Interpolation routine to find conditions at time of final step
		# i.e. when h = h_final

		self.final_values = placeholder()
		final_list = interpolate_event(self, self.h[self.i], self.l)

		for index, val in enumerate(self.l):
			#self.final_values.__dict__[val] = final_list[index]
			self.final_values.__dict__.update({val : final_list[index]})

		if self.console_output == True:
			print 'END EVENT CONDITIONS CALCULATED'

#		return [final_time, final_V, final_gamma, final_rho, final_p, final_T,
#					final_mu, final_p_dyn, final_g, final_a, final_Ma]

		return None

	def final_step_assign(self):
		# Assign values calculated for final step
		self.solver_time[self.index] 	= self.final_values.time
		self.V[self.index] 			= self.final_values.V
		self.gamma[self.index] 		= self.final_values.gamma
		self.solver_rho[self.index] 	= self.final_values.rho
		self.solver_p[self.index] 		= self.final_values.p
		self.solver_T[self.index] 		= self.final_values.T
		self.solver_mu[self.index] 	= self.final_values.mu
		self.p_dyn[self.index] 		= self.final_values.p_dyn
		self.g[self.index] 			= self.final_values.g
		self.solver_a[self.index] 		= self.final_values.g
		self.Ma[self.index] 			= self.final_values.Ma
		self.h[self.index]			= self.final_values.h

		return None