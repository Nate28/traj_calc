# -*- coding: utf-8 -*-
"""
=== FLOW CALC LIB ===
Common calculations for fluid mechanics.

Created on Sat Jul  5 15:28:47 2014
@author: Nathan Donaldson
"""

__author__ = 'Nathan Donaldson'
__email__ = 'nathan.donaldson@eng.ox.ac.uk'
__status__ = 'Development'
__version__ = '0.7'
__license__ = 'MIT'

def aero_force(rho, V, C, A):
	"""
	Aerodynamic lift/drag equation

	Input variables:
		rho	:	Fluid density
		V	:	Fluid velocity
		C 	: 	Lift/drag coefficient
		A 	: 	Reference area
	"""

	F = 0.5 * rho * (V**2) * C * A

	return F

def normal_shock_ratios(Ma_1, gamma_var):
	"""
	Returns normal shock ratios for static and stagnation pressure and
	temperature, and density.  Also returns the Mach number following the
	shock (http://www.grc.nasa.gov/WWW/k-12/airplane/normal.html).

	Note that input variables are for flow UPSTREAM of shock, while returns
	are for the flow DOWNSTREAM of the shock.  Returned ratios are of the form:
	'Downstream condition' / 'Upstream condition' i.e.
	'Condition beyond shock' / ' Condition in front of shock'

	Input variables:
		Ma_1		:	Mach number upstream of shock
		gamma_var	:	Ratio of specific heats
	"""

	import numpy as np

	p_ratio = ((2 * gamma_var * (Ma_1**2)) - (gamma_var - 1)) / (gamma_var + 1)

	T_ratio = (((2 * gamma_var * (Ma_1**2)) - (gamma_var - 1)) * \
		(((gamma_var - 1) * (Ma_1**2)) + 2)) / (((gamma_var + 1)**2) * (Ma_1**2))

	rho_ratio = ((gamma_var + 1) * (Ma_1**2)) / (((gamma_var - 1) * \
		(Ma_1**2)) + 2)

	p_0_ratio = ((((gamma_var + 1) * (Ma_1**2)) / (((gamma_var - 1) * \
		(Ma_1**2)) + 2))**(gamma_var / (gamma_var - 1))) * \
		(((gamma_var + 1) / ((2 * gamma_var * (Ma_1**2)) - \
		(gamma_var - 1)))**(1 / (gamma_var - 1)))

	T_0_ratio = 1.0

	Ma_2 = np.sqrt((((gamma_var - 1) * (Ma_1**2)) + 2) / ((2 * gamma_var * \
		(Ma_1**2)) - (gamma_var - 1)))

	return [p_ratio, T_ratio, p_0_ratio, T_0_ratio, rho_ratio, Ma_2]

def viscosity(**kwargs):
	"""
	Calculates the viscosity of a gas using one of the following:
	1) Sutherland's law
		(http://www.cfd-online.com/Wiki/Sutherland's_law)
		(http://en.wikipedia.org/wiki/Viscosity)
		(http://mac6.ma.psu.edu/stirling/simulations/DHT/ViscosityTemperatureSutherland.html)
	2) Chapman-Enskog equation
	(http://www.owlnet.rice.edu/~ceng402/ed1projects/proj00/clop/mainproj2.html)

	Input variables:
		mode	:	'S' (Sutherland) or 'C-E' (Chapman-Enskog)
		T		:	Gas temperature

	Sutherland variables:
		mu_ref	:	Reference viscosity
		T_ref	:	Reference temperature
		C1		:	Sutherland's law constant
		gas		:	Common gas properties
		S		:	Sutherland temperature

	Chapman-Enskog variables:
		M		:	Molecular weight
		sigma	:	Lennard-Jones parameter (collision diameter)
		omega	:	Collision integral
	"""

	from scipy.constants import k
	import numpy as np

	if (kwargs['mode'] == 'S') or (kwargs['mode'] == 's') or \
	(kwargs['mode'] == 'Sutherland') or (kwargs['mode'] == 'sutherland'):

		# Sutherland constants for common gases (C1, S, mu_ref, T_ref)
		gas_dict = {
					'air'	:	[1.4580000000-6, 110.4, 1.716E-5, 273.15],
					'N2'	:	[1.406732195E-6, 111, 17.81E-6, 300.55],
					'O2'	:	[1.693411300E-6, 127, 20.18E-6, 292.25],
					'CO2'	:	[1.572085931E-6, 240, 14.8E-6, 293.15],
					'CO'	:	[1.428193225E-6, 118, 17.2E-6, 288.15],
					'H2'	:	[0.636236562E-6, 72, 8.76E-6, 293.85],
					'NH3'	:	[1.297443379E-6, 370, 9.82E-6, 293.15],
					'SO2'	:	[1.768466086E-6, 416, 12.54E-6, 293.65],
					'He'	:	[1.484381490E-6, 79.4, 19E-6, 273],
					'CH4'	:	[1.252898823E-6, 197.8, 12.01E-6, 273.15]
					}

		if ('gas' in kwargs) and (kwargs['gas'] in gas_dict):
			kwargs.update({'C1'	:	gas_dict[kwargs['gas']][0]})
			kwargs.update({'S'		:	gas_dict[kwargs['gas']][1]})
			kwargs.update({'mu_ref':	gas_dict[kwargs['gas']][2]})
			kwargs.update({'T_ref'	:	gas_dict[kwargs['gas']][3]})

		if ('mu_ref' in kwargs) and ('T_ref' in kwargs) and ('T' in kwargs) \
			and ('S' in kwargs):
			mu = kwargs['mu_ref'] * ((kwargs['T'] / \
			kwargs['T_ref'])**(3.0/2.0)) * ((kwargs['T_ref'] + \
			kwargs['S']) / (kwargs['T'] + kwargs['S']))
		elif ('T' in kwargs) and ('S' in kwargs) and ('C1' in kwargs):
			mu = kwargs['C_1'] * ((kwargs['T']**(3.0/2.0)) / (kwargs['T'] + kwargs['S']))
		else:
			raise KeyError('Incorrect variable assignment')

	elif (kwargs['mode'] == 'C-E') or (kwargs['mode'] == 'c-e') or \
	(kwargs['mode'] == 'Chapman-Enskog') or (kwargs['mode'] == 'chapman-enskog'):

		mu = 2.6693E-5 * (np.sqrt(kwargs['M'] * kwargs['T'])) / \
			(kwargs['omega'] * (kwargs['sigma']**2))

	return mu

def mean_free_path(T, p, d=4E-10):
    """
    Calculates the molecular mean free path in a gaseous flow

    Input variables:
        T   :   Gas temperature
        p   :   Gas pressure
        d   :   Molecular diameter (default is air)
    """

    from scipy.constants import k
    import numpy as np

    mfp = (k * T) / (np.sqrt(2) * np.pi * (d**2) * p)

    return mfp

def Knudsen(T, p, L, d=4E-10):
	"""
	Calculates the Knudsen number in a gaseous flow

    Input variables:
		T   :   Gas temperature
		p   :   Gas pressure
		L   :   Characteristic length scale
		d   :   Molecular diameter (default is air)
	"""

	Kn = mean_free_path(T, p, d) / L

	return Kn

def Mach(a, V):
	"""
	Calculates flow Mach number
	"""

	Ma = V / a

	return Ma

def Reynolds(rho, U, L, mu):
	"""
	Calculates flow Reynolds number
	"""

	Re = (rho * U * L) / mu

	return Re

def KnReMa(gamma_var, **kwargs):
	import numpy as np

	if ('Kn' in kwargs) and ('Ma' in kwargs):
		# Calculate Re
		ans = (kwargs['Ma'] / kwargs['Kn']) * (((gamma_var * np.pi) / 2)**0.5)
	elif ('Kn' in kwargs) and ('Re' in kwargs):
		# Calculate Ma
		ans = (kwargs['Kn'] * kwargs['Re']) / (((gamma_var * np.pi) / 2)**0.5)
	elif ('Ma' in kwargs) and ('Re' in kwargs):
		# Calculate Kn
		ans = (kwargs['Ma'] / kwargs['Re']) * (((gamma_var * np.pi) / 2)**0.5)

	return ans

def isen_nozzle_ratios(M_E, gamma_var):
    """
    Calculates ratio between stagnation and exit pressure and temperature in
    and isentropic nozzle.

    Input variables:
        M_E         :   Mach number at exit
        gamma_var   :   ratio of specific heats
    """

    T_ratio = 1 / (1 + (((gamma_var - 1) / 2) * (M_E**2)))
    p_ratio = T_ratio ** (gamma_var / (gamma_var - 1))

    return p_ratio, T_ratio

def isen_nozzle_A_ratio(M_E, gamma_var):
    """
    Calculates ratio between exit and throat areas in an isentropic nozzle.

    Input variables:
        M_E         :   Mach number at exit
        gamma_var   :   Ratio of specific heats
    """

    A_ratio = (((gamma_var + 1) / 2)**-((gamma_var + 1) / \
        (2 * (gamma_var - 1)))) * ((1 + (((gamma_var - 1) / 2) * \
        (M_E**2)))**((gamma_var + 1) / (2 * (gamma_var - 1)))) * (1/M_E)

    return A_ratio

def isen_nozzle_Ma(A_ratio_sol, gamma_var, tol=1E-10, step_size=0.1):
    """
    Iteratively solves the isentropic expansion equation for converging-
    diverging nozzles in order to find the Mach number produced by a given
    exit/throat area ratio.

    Input variables:
        A_ratio_sol :   Nozzle area ratio (exit/throat)
        gamma_var   :   ratio of specific heats
    """

    # Initialise solution
    M_E = 1.0
    A_ratio = isen_nozzle_A_ratio(M_E, gamma_var)

    # Begin iteration loop
    while (A_ratio <= (A_ratio_sol - tol)) or (A_ratio >= (A_ratio_sol + tol)):

        # If current solution is smaller than (required value - tolerance), iterate
        # to next value of M_E and repeat calculation
        if A_ratio < (A_ratio_sol - tol):
            M_E += step_size
            A_ratio = isen_nozzle_A_ratio(M_E, gamma_var)

        # If current solution is larger than (required value + tolerance), reverse
        # direction of solver and half step size
        elif A_ratio > (A_ratio_sol + tol):
            step_size /= 2
            M_E -= step_size
            A_ratio = isen_nozzle_A_ratio(M_E, gamma_var)

    #print '\nSolution computed!\nA_E/A*:\t%f\nM_E:\t%f' % (A_ratio, M_E)

    return M_E

def isen_nozzle_mass_flow(A_t, p_t, T_t, gamma_var, R, M):
	"""
	Calculates mass flow through a nozzle which is isentropically expanding
	a given flow

	Input variables:
		A_t 		: 	nozzle throat area
		gamma_var 	: 	ratio of specific heats
		p_t 		: 	pressure at throat
		T_t 		: 	temperature at throat
		M 		: 	Mach number at throat
		R 		: 	Perfect gas constant
	"""

	m_dot = (A_t * p_t * (T_t**0.5)) * ((gamma_var / R)**0.5) * \
		M * ((1 + (((gamma_var - 1) / 2) * \
		(M**2)))**(-((gamma_var + 1) / (2 * (gamma_var - 1)))))

	return m_dot

def isen_nozzle_choked_mass_flow(A_t, p_0, h_0, gamma_var):
	"""
	Calculates mass flow through a nozzle which is isentropically expanding
	a given flow and is choked (Mach number at throat is 1.0)

	Input variables:
		A_t 		: 	nozzle throat area
		gamma_var 	: 	ratio of specific heats
		p_t 		: 	stagnation chamber
		T_t 		: 	stagnation temperature
	"""

	#m_dot = ((gamma_var * p0) * (()**())) * (()**())

	return None

def T_static(**kwargs):
    """
    Calculates static temperature based upon either of two input variable sets.

    First method:
        T_static(C_p = specific heat capacity,
            V = fluid velocity,
            T_0 = stagnation temperature)

    Second method:
        T_static(Ma = fluid Mach number,
            gamma_var = ratio of specific heats,
            T_0 = stagnation temperature)
    """

    if ('C_p' in kwargs) and ('V' in kwargs) and ('T_0' in kwargs):
        T = kwargs['T_0'] - ((kwargs['V']**2) / (2 * kwargs['C_p']))
    elif ('Ma' in kwargs) and ('gamma_var' in kwargs) and ('T_0' in kwargs):
        T = kwargs['T_0'] / (1 + (((kwargs['gamma_var'] - 1) / 2) * \
        (kwargs['Ma']**2)))
    else:
        raise KeyError('Incorrect variable assignment')

    return T

def T_stag(**kwargs):
    """
    Calculates stagnation temperature based upon either of two input
    variable sets.

    First method:
        T_stag(C_p = specific heat capacity,
            V = fluid velocity,
            T = static temperature)

    Second method:
        T_stag(Ma = fluid Mach number,
            gamma_var = ratio of specific heats,
            T = static temperature)
    """

    if ('C_p' in kwargs) and ('V' in kwargs) and ('T' in kwargs):
        T_0 = kwargs['T'] + ((kwargs['V']**2) / (2 * kwargs['C_p']))
    elif ('Ma' in kwargs) and ('gamma_var' in kwargs) and ('T' in kwargs):
        T_0 = kwargs['T'] * (1 + (((kwargs['gamma_var'] - 1) / 2) * \
        (kwargs['Ma']**2)))
    else:
        raise KeyError('Incorrect variable assignment')

    return T_0

def p_stag(**kwargs):
	"""
	Calculates stagnation pressure based upon either of three input
	variable sets.  Optionally returns the ratio between stagnation
	and freestream pressure if no pressure term is supplied.

	First method:
		p_stag(rho = fluid density,
			V = fluid velocity,
			p = static pressure)

	Second method:
		p_stag(Ma = fluid Mach number,
			gamma_var = ratio of specific heats,
			p = static pressure)

	Return ratio:
		p_stag(Ma = fluid Mach number,
			gamma_var = ratio of specific heats)
	"""

	if ('rho' in kwargs) and ('V' in kwargs) and ('p' in kwargs):
		p_0 = kwargs['p'] + (0.5 * kwargs['rho'] * (kwargs['V']**2))
	elif ('p' in kwargs) and ('gamma_var' in kwargs) and ('Ma' in kwargs):
		p_0 = kwargs['p'] * ((1 + (((kwargs['gamma_var'] - 1) / 2) * \
			(kwargs['Ma']**2)))**(kwargs['gamma_var'] / (kwargs['gamma_var'] - 1)))
	elif ('gamma_var' in kwargs) and ('Ma' in kwargs):
		p_0 = ((1 + (((kwargs['gamma_var'] - 1) / 2) * \
			(kwargs['Ma']**2)))**(kwargs['gamma_var'] / (kwargs['gamma_var'] - 1)))
	else:
		raise KeyError('Incorrect variable assignment')

	return p_0

def p_static(**kwargs):
	"""
	Calculates static pressure based upon either of three input
	variable sets.  Optionally returns the ratio between freestream
	and stagnation pressure if no pressure term is supplied.

	First method:
		p_static(rho = fluid density,
			V = fluid velocity,
			p_0 = stagnation pressure)

	Second method:
		p_static(Ma = fluid Mach number,
			gamma_var = ratio of specific heats,
			p_0 = stagnation pressure)

	Return ratio:
		p_static(Ma = fluid Mach number,
			gamma_var = ratio of specific heats)
	"""

	if ('rho' in kwargs) and ('V' in kwargs) and ('p_0' in kwargs):
		p = kwargs['p_0'] - (0.5 * kwargs['rho'] * (kwargs['V']**2))
	elif ('p_0' in kwargs) and ('gamma_var' in kwargs) and ('Ma' in kwargs):
		p = kwargs['p_0'] / ((1 + (((kwargs['gamma_var'] - 1) / 2) * \
			(kwargs['Ma']**2)))**(kwargs['gamma_var'] / (kwargs['gamma_var'] - 1)))
	elif ('gamma_var' in kwargs) and ('Ma' in kwargs):
		p = 1 / (((1 + (((kwargs['gamma_var'] - 1) / 2) * \
			(kwargs['Ma']**2)))**(kwargs['gamma_var'] / (kwargs['gamma_var'] - 1))))
	else:
		raise KeyError('Incorrect variable assignment')

	return p

def p_dyn(**kwargs):
	"""
	Calculates dynamic pressure based upon either of two input
	variable sets.

	First method:
		p_dyn(rho = fluid density,
			V = fluid velocity)

	Second method:
		p_dyn(Ma = fluid Mach number,
			gamma_var = ratio of specific heats,
			p = static pressure)
	"""

	if ('rho') and ('V') in kwargs:
		q = 0.5 * kwargs['rho'] * (kwargs['V']**2)
	elif ('Ma' in kwargs) and ('p' in kwargs) and ('gamma_var' in kwargs):
		q = 0.5 * kwargs['gamma'] * kwargs['p'] * (kwargs['Ma']**2)

	return q

def p_e(**kwargs):
	"""
	Calculates pressure at point immediately behind shock in
	subsonic or supersonic flow for thermodynamic equilibrium and either
	a calorically perfect or imperfect gas.
	"""

	import numpy as np

	Ma_inf = kwargs['Ma_inf']
	gamma_var = kwargs['gamma_var']
	p_inf = kwargs['p_inf']

	if ('T0p' in kwargs):
		T0p = kwargs['T0p']

	if ('R' in kwargs):
		R = kwargs['R']

	if ('T_inf' in kwargs):
		T_inf = kwargs['T_inf']

	if ('theta_v' in kwargs):
		theta_v = kwargs['theta_v']
	else:
		theta_v = -1

	if (theta_v != -1):

		# Calorically imperfect gas
		term_a = gamma_var / (gamma_var - 1)
		term_b = 0.5 * theta_v

		if (Ma_inf < 1):

			# Subsonic flow
			P_con = np.log(p_inf) - \
					(term_a * np.log(T_inf)) - \
					(2 * (term_b / T_inf) * (1 + (1 / (np.exp(((term_b / T_inf)**2) - 1))))) + \
					np.log(np.exp(((term_b / T_inf)**2) - 1))

			dlogps = (term_a * np.log(T0p)) + \
				(2 * (term_b / T0p) * (1 + (1 / (np.exp(((term_b / T0p)**2) - 1))))) - \
				np.log(np.exp(((term_b / T0p)**2) - 1)) + \
				P_con

			P0p = np.exp(dlogps)

		elif (Ma_inf > 1):

			# Supersonic flow
			V_inf = Ma_inf * speed_of_sound(gamma_var, R, T_inf)
			rho_inf = p_inf / (R * T_inf)
			h_inf = enthalpy(gamma_var=gamma_var, R=R, T=T_inf)

			epsilon = 0.5
			epsilon_1 = 0.0
			epsilon_2 = 0.0
			tol = 1E-6
			lap = 3

			for n in range(100):
				p_shock = p_inf + (rho_inf * (V_inf**2) * (1 - epsilon))
				h_shock = h_inf + (0.5 * (V_inf**2) * (1 - (epsilon**2)))
				T_shock = h_to_T(h_shock, gamma_var, R, theta_v)
				rho_shock = p_shock / (R * T_shock)

				# Accelerate convergence with Aitken's delta-squared process
				# Update epsilon values for current iteration
				epsilon_2 = epsilon_1
				epsilon_1 = epsilon
				epsilon = rho_inf / rho_shock

				if (n == lap):
					# Run Aitken's delta squared process every three iterations of FOR loop
					# (epsilon values are updated sequentially on a 3 loop cycle)
					lap += 3
					epsilon = (epsilon_2 - ((epsilon_1 - epsilon_2)**2)) / (epsilon - (2 * (epsilon_1 + epsilon_2)))

				if abs(epsilon - epsilon_1) <= tol:
					break
				else:
					pass

				P_con = np.log(p_shock) - \
					(term_a * np.log(T_shock)) - \
					(2 * (term_b / T_shock) * (1 + (1 / (np.exp(((term_b / T_shock)**2) - 1))))) + \
					np.log(np.exp(((term_b / T_shock)**2) - 1))

				dlogps = (term_a * np.log(T0p)) + \
					(2 * (term_b / T0p) * (1 + (1 / (np.exp(((term_b / T0p)**2) - 1))))) - \
					np.log(np.exp(((term_b / T0p)**2) - 1)) + \
					P_con

				P0p = np.exp(dlogps)

	elif (theta_v == -1):

		if (Ma_inf > 1):
		# Calorically perfect gas, supersonic flow
			P0p = p_inf * normal_shock_ratios(Ma_inf, gamma_var)[2]
			#P0p = P_inf * (((gamma_var + 1) * (Ma**2) / 2)**(gamma_var / (gamma_var - 1))) *
			#	(((gamma_var + 1) / (2 * gamma_var * (Ma**2) - gamma_var + 1))**(1 / (1 - gamma_var)))

		elif (Ma_inf < 1):
		# Calorically perfect gas, subsonic flow
			P0p = p_inf * (T0p**(gamma_var / (gamma_var - 1)))

	return P0p

def Prandtl(**kwargs):
    """
    Calculates Prandtl number based upon either of two input variable sets.

    First method:
        Pr(C_p = specific heat capacity,
           mu = dynamic viscosity,
           k = thermal conductivity)

    Second method:
       Pr(nu = kinematic viscosity,
          alpha = thermal diffusivity)
    """

    if ('C_p' in kwargs) and ('k' in kwargs) and ('mu' in kwargs):
        prandtl_number = (kwargs['C_p'] * kwargs['mu']) / kwargs['k']
    elif ('nu' in kwargs) and ('alpha' in kwargs):
        prandtl_number = kwargs['nu'] / kwargs['alpha']
    else:
        raise KeyError('Incorrect variable assignment')

    return prandtl_number

def Schmidt(**kwargs):
    """
    Calculates Schmidt number based upon either of two input variable sets.

    First method:
       Sc(nu = kinematic viscosity,
          alpha = thermal diffusivity)

    Second method:
        Sc(D = mass diffusivity,
           mu = dynamic viscosity,
           rho = fluid density)
    """

    if ('mu' in kwargs) and ('D' in kwargs) and ('rho' in kwargs):
        schmidt_number = kwargs['mu'] / (kwargs['rho'] * kwargs['D'])
    elif ('nu' in kwargs) and ('D' in kwargs):
        schmidt_number = kwargs['nu'] / kwargs['D']
    else:
        raise KeyError('Incorrect variable assignment')

    return schmidt_number

def Lewis(**kwargs):
    """
    Calculates Lewis number based upon either of two input variable sets.

    First method:
       Le(Sc =  Schmidt number,
          Pr = Prandtl number)

    Second method:
        Le(D = mass diffusivity,
           alpha = thermal diffusivity)
    """

    if ('Sc' in kwargs) and ('Pr' in kwargs):
        lewis_number = kwargs['Sc'] / kwargs['Pr']
    elif ('alpha' in kwargs) and ('D' in kwargs):
        lewis_number = kwargs['alpha'] / kwargs['D']
    else:
        raise KeyError('Incorrect variable assignment')

    return lewis_number

def thermal_diffusivity(k, rho, C_p):
    """
    Calculates thermal diffusion coefficient of a fluid.

    Input variables:
        T   :   Thermal conductivity
        rho :   Fluid density
        C_p :   Specific heat capacity
    """

    alpha = k / (rho * C_p)

    return alpha

def kinematic_viscosity(mu, rho):
    """
    Calculates kinematic viscosity of a fluid.

    Input variables:
        T   :   Thermal conductivity
        rho :   Fluid density
        C_p :   Specific heat capacity
    """

    nu = mu / rho

    return nu

def vel_gradient(**kwargs):

    """
    Calculates velocity gradient across surface object in supersonic
    flow (from stagnation point) based upon either of two input variable
    sets.

    First method:
	vel_gradient(R_n = Object radius (or equivalent radius, for
             shapes that are not axisymmetric),
         p_0 = flow stagnation pressure,
         p_inf = flow freestream static pressure
         rho = flow density)

    Second method:
        vel_gardient(R_n = Object radius (or equivalent radius, for
						shapes that are	not axisymmetric),
					delta = Shock stand-off distance (from object
						stagnation point),
					U_s = Flow velocity immediately behind shock)
    """
    if ('R_n' in kwargs) and ('p_0' in kwargs) and ('p_inf' in kwargs) and \
    ('rho' in kwargs):
        from numpy import sqrt
        vel_gradient = (1 / kwargs['R_n']) * sqrt((2 * (kwargs['p_0'] - \
            kwargs['p_inf'])) / kwargs['rho'])
    elif ('R_n' in kwargs) and ('U_s' in kwargs) and ('delta' in kwargs):
        b = kwargs['delta'] + kwargs['R_n']
        vel_gradient = (kwargs['U_s'] / kwargs['R_n']) * (1 + ((2 + ((b**3) / \
            (kwargs['R_n']**3))) / (2 * (((b**3) / (kwargs['R_n']**3)) - 1))))
    else:
        raise KeyError('Incorrect variable assignment')

    return vel_gradient

def shock_standoff(R_n, rho_inf, rho_s):
	"""
	Approximates supersonic shock stand-off distance for the stagnation
	point of an obstacle with equivalent radius R_n

    Input variables:
        R_n 	:   Obstacle radius
        rho_inf :   Freestream density
        rho_s 	:   Density at point immediately behind shock
	"""

	delta = R_n * (rho_inf / rho_s)

	return delta

def enthalpy_wall_del(T_0, T_w, C_p):
    """
    Calculates specific enthalpy difference between total conditions and
    those at the stagnation point of a sphere in supersonic flow.

    Input variables:
        T_0 :   Gas total temperature
        T_w :   Sphere wall temperature
        C_p :   Specific heat capacity
    """

    del_h = C_p * (T_0 - T_w)

    return del_h

def C_p_calc(**kwargs):
	"""
	Calculates specific heat of a given fluid with (either calorically
	perfect or imperfect, depending on input variables)

	For calorically perfect gas:
		C_p_calc(R = gas constant,
			gamma_var = ratio of specific heats)

	For calorically imperfect gas:
		C_p_calc(R = gas constant,
			gamma_var = ratio of specific heats,
			theta_v = vibrational temperature,
			T = gas temperature)
	"""

	import numpy as np

	if ('gamma_var' in kwargs) and ('R' in kwargs):
		C_p = (kwargs['gamma_var'] * kwargs['R']) / (kwargs['gamma_var'] - 1)
	elif ('theta_v' in kwargs) and ('gamma_var' in kwargs) and ('R' in kwargs) and ('T' in kwargs):
		term = 0.5 * (kwargs['theta_v'] / kwargs['T'])
		C_p = (kwargs['gamma_var'] * kwargs['R']) / (kwargs['gamma_var'] - 1)
		C_p += kwargs['R'] * ((term / np.sinh(term))**2)
	else:
		raise KeyError('Incorrect variable assignment')

	return C_p

def enthalpy(**kwargs):
	"""
	Calculates specific enthalpy based upon one of three
	input variable sets.  Calculates stagnation enthalpy
	if arguments C_p, T, U and mode=total are supplied.

	First method:
		enthalpy(C_p = specific heat capacity,
			T = gas temperature)

	Second method (total/stagnation enthalpy):
		enthalpy(C_p = specific heat capacity,
			T_0 = flow total static temperature,
			Vel = flow velocity)

	Third method:
		enthalpy(gamma_var = gamma,
			R = specific gas constant,
			T = gas temperature)

	Fourth method (includes vibrational energy contributions):
		enthalpy(gamma_var = gamma,
			R = specific gas constant,
			T = gas temperature,
			theta_v = characteristic vibrational temperature of gas)

	Fifth method:
		enthalpy(U = internal energy,
			p = gas pressure,
			Vol = gas volume)
	"""

	import numpy as np

	# Find C_p for a calorically perfect gas from gamma and R if not already supplied
	if (('gamma_var' in kwargs) and ('R' in kwargs) and ('C_p' not in kwargs)
		and ('theta_v' in kwargs)):
		kwargs['C_p'] = C_p_calc(gamma_var=kwargs['gamma_var'], R=kwargs['R'],
			theta_v=kwargs['theta_v'], T=kwargs['T'])
	elif ('gamma_var' in kwargs) and ('R' in kwargs) and ('C_p' not in kwargs):
		kwargs['C_p'] = C_p_calc(gamma_var=kwargs['gamma_var'], R=kwargs['R'])
	elif ('C_p' in kwargs):
		pass
	else:
		raise KeyError('Incorrect variable assignment')

	# Calculate enthalpy for calorically perfect gas
	if ('T' in kwargs) and ('C_p' in kwargs):
		h = kwargs['C_p'] * kwargs['T']
	elif ('T_0)' in kwargs) and ('C_p' in kwargs) and ('Vel' in kwargs):
		h = (kwargs['C_p'] * kwargs['T_0']) + ((kwargs['Vel']**2) / 2)
	elif ('U' in kwargs) and ('p' in kwargs) and ('Vol' in kwargs):
		h = kwargs['U'] + (kwargs['p'] * kwargs['Vol'])
	else:
		raise KeyError('Incorrect variable assignment')

	# Check for inclusion of vibrational temperature variable, theta_v
	# Calculate enthalpy for calorically imperfect gas
	if ('theta_v' in kwargs) and ('R' in kwargs) and ('T' in kwargs):
		term = kwargs['theta_v'] / kwargs['T']
		h += (term / (np.exp(term) - 1)) * kwargs['R'] * kwargs['T']
	else:
		pass

	return h

def h_to_T(h, gamma_var, R, theta_v):
	"""
	Extracts total temperature given enthalpy using Newton-Raphson iteration.

	Input variables:
		h			:	Fluid specific enthalpy
		gamma_var	:	Ratio of specific heats
		R			:	Gas constant
		theta_v		:	Vibrational temperature
	"""

	T = h / ((gamma_var * R) / (gamma_var - 1))

	for n in range(100):
		eval = enthalpy(T=T, gamma_var=gamma_var, R=R, theta_v=theta_v) - h
		devaldt = C_p_calc(T=T, gamma_var=gamma_var, R=R, theta_v=theta_v)
		delta = -eval / devaldt
		T = T + delta
		if abs(delta) <= 1E-6:
			break

	if (n == 100) and (abs(delta) > 1E-6):
		print 'ERROR: Newton-Raphson method failed to converge in 100 iterations'

	return T

def self_diffusion(n, T, d=4E-10, m=28.97E-3):
    """
    Calculates self diffusion coefficient of a molecule in single species
    comprised of molecules of the same type.

    Input variables:
        T   :   Fluid temperature
        n   :   Number density
        d   :   Molecular diameter (default is air) [metres]
        m   :   Molar mass (default is air) [kg/mol]
    """

    from numpy import sqrt, pi
    from scipy.constants import k

    D_11 = (3 / (8 * n * (d**2))) * sqrt((k * T) / (pi * m))

    return D_11

def fay_riddell(solid, bound, rho_e, mu_e, rho_w, mu_w, Pr, Le, \
    vel_grad, h_0, h_D, enthalpy_wall_del):

	import numpy as np

	# Empirical coefficient selected based on shape of geometry (sphere
	# or cylinder)
	if solid == 'cylinder' or solid == 'cyl':
		A = 0.57
	elif solid == 'sphere' or solid == 'sph':
		B = 0.76
	else:
		raise ValueError('Invalid input parameter.  Expected \'cylinder'
		'\', \'cyl\', \'sphere\' or \'sph\'')

	# Selection of factors to include in calculation (based on boundary
	# layer conditions)
	if bound == 'equil':
		B = 1 + (((Le**0.52) - 1) * (h_D/h_0))
	elif bound == 'fr_cat':
		B = 1 + (((Le**0.63) - 1) * (h_D/h_0))
	elif bound == 'fr_noncat':
		B = 1 - (h_D/h_0)
	else:
		raise ValueError('Invalid input paarmeter.  Expected \'equil\','
		' \'fr_cat\' or \'fr_noncat\'')

	q_dot = A * (Pr**-0.6) * ((rho_e * mu_e)**0.4) * ((rho_w * mu_w)**0.1) * \
            np.sqrt(vel_grad) * enthalpy_wall_del * B

def Mach_vector(M_inf, alpha, theta, mode='deg'):
	"""
	Returns vector of components of Mach number based upon pitch and yaw
	angles of freestream flow direction.

	Input variables:
		M_inf	:	Freestream Mach number
		alpha	:	Freestream pitch angle
		theta	:	Freestream yaw angle
		mode	:	Toggles between angle input types;
					'rad' specifies radians, while 'deg' specifies degrees
					('deg' is the default, and need not be specified)
	"""

	import numpy as np

	if mode == 'rad':
		pass
	elif mode == 'deg':
		alpha = np.deg2rad(alpha)
		theta = np.deg2rad(theta)
	else:
		print 'ERROR: incorrect angular input specified; assuming radians'

	y = np.sin(alpha)
	z = np.sin(theta) * np.cos(alpha)
	x = np.cos(theta) * np.cos(alpha)

	M = M_inf * np.array([np.float(x), np.float(y), np.float(z)])

	return M

def pres_coeff_mod_newton(N, M_vector, Cp_max=2):
	"""
	Calculates pressure coefficient along a surface in supersonic/hypersonic flow
	using the Modified Newtonian method.

	Input variables:
		N		:	Array of face normal vectors for the surface being assessed
		M		:	Freestream Mach number (array or list of components, 3 floats)
		Cp_max	:	Maximum pressure coefficient, evaluated at a stagnation
					point behind a normal shock wave (scalar, float)
	"""

	import numpy as np

	# Initalise variables
	len_N = len(N)
	delta = np.zeros([len_N])
	mag_N = np.zeros([len_N])
	mag_M = np.linalg.norm(M_vector)

	# Angle between two 3D vectors
	for index, value in enumerate(N):
		mag_N[index] = np.linalg.norm(N[index, :])
		delta[index] = np.arccos(np.dot(N[index], M_vector) / (mag_N[index] * mag_M))

	# Calculate Cp for all elements
	Cp = Cp_max * ((np.cos(delta))**2)

	# Apply element shielding assumption
	# i.e. if delta is greater than 90°, set Cp to 0
	for index, value in enumerate(Cp):
		if abs(delta[index]) > (np.pi / 2):
			Cp[index] = 0

	return Cp, delta

def pres_coeff_max(M, gamma_var):
	"""
	Calculates the maximum pressure coefficient on a surface following a normal
	shock at the stagnation point

	Input variables:
		M			:	Freestream Mach number (scalar, float)
		gamma_var	:	Specific heat ratio (scalar, float)
	"""

	p01_inf_ratio = p_stag(Ma=M, gamma_var=gamma_var)
	_, _, _, p02_p01_ratio, _, _ = normal_shock_ratios(M, gamma_var)
	p02_inf_ratio = p02_p01_ratio * p01_inf_ratio

	Cp_max = (2 / (gamma_var * (M**2))) * (p02_inf_ratio - 1)

	return Cp_max

def pres_from_Cp(Cp, p_inf, rho_inf, V_inf):
	"""
	Calculates pressure from freestream conditions based upon local pressure
	coefficient, Cp.

	Input variables:
		Cp			:	Local pressure coefficient
		p_inf		:	Freestream pressure
		rho_inf		:	Freestream density
		V_inf		:	Freestream velocity
	"""

	p = p_inf + (Cp * 0.5 * rho_inf * (V_inf**2))

	return p

def surface_force(p, N, A):
	"""
	Calculates the XYZ components of the force acting upon a three dimensional
	surface element given the local pressure, element area, and element normal
	vector.

	Input variables:
		p	:	Pressure acting upon surface
		N	:	Surface normal vector
		A	:	Surface area
	"""

	import numpy as np

	F_mag = p * A
	F_x = -F_mag * N[:, 0]
	F_y = -F_mag * N[:, 1]
	F_z = -F_mag * N[:, 2]
	F = np.array([F_x, F_y, F_z]).T

	return F

def surface_moment(F, C, CG):
	"""
	Calculates moment(s) exerted about the centre of gravity of a structure by
	forces acting on individual surface panels.  Moment arm vectors are
	calculated about the centre of gravity based upon the surface element
	centres.

	Input variables:
		F	:	Force XYZ components (array or list, m X 3 floats)
		C	:	Coordinates of surface element centres (array or list, m X 3 floats)
		CG	:	Coordinates of centre of gravity (array or list, 3 floats)
	"""

	import numpy as np

	# Calculate moment arms (XYZ vector components)
	L = C - CG

	# Split moment arms into axis directions
	L_x = L[:, 0]
	L_y = L[:, 1]
	L_z = L[:, 2]

	# Split forces into directional components
	F_x = F[:, 0]
	F_y = F[:, 1]
	F_z = F[:, 2]

	# Calculate moments
	M_x = (F_y * L_z) + (F_z * L_y)
	M_y = (F_x * L_z) + (F_z * L_x)
	M_z = (F_x * L_y) + (F_y * L_x)

	# Combine moments for export
	M = np.array([M_x, M_y, M_z]).T

	return [M, L]

def speed_of_sound(gamma_var, R, T):
	"""
	Calculates local speed of sound.

	Input variables:
		gamma_var	:	Ratio of specific heats
		R			:	Specific gas constant for given gas
		T			:	Gas temperature
	"""

	import numpy as np
	a = np.sqrt(gamma_var * R * T)

	return a

def aero_coeff(F, M, A_ref, L_ref, rho_inf, V_inf, alpha, beta):
	"""
	Calculates aerodynamic coefficients from the sum of forces and moments
	acting on a body and about its centre of gravity.

	Input variables:
		F		:	Forces acting on elements (array or list, m X 3 floats)
		M		:	Moments acting on elements about CoG (array or list, m X 3 floats)
		A_ref	:	Reference area
		L_ref	:	Reference length
		p_inf	:	Freestream pressure
		V_inf	:	Freestream velocity

	Returns (as a list):
		C_L		:	Lift coefficient (vertical force)
		C_D		:	Drag coefficient (longitudinal force)
		C_N		:	Normal coefficient (yawing moment)
		C_A		:	Axial coefficient (rolling moment)
		C_M		:	Moment coefficient (pitching moment)
	"""

	import numpy as np

	# Dynamic pressure
	q = p_dyn(rho=rho_inf, V=V_inf)

	# Sum of forces
	F_x = np.sum(F[:, 0])
	F_y = np.sum(F[:, 1])
	F_z = np.sum(F[:, 2])

	# Sum of moments (in a given direction, NOT about an axis)
	# NB/ Moment components in [x, y, z] direction act about the [z&y, x&z, x&y] axes.
	M_x = np.sum(M[:, 0])
	M_y = np.sum(M[:, 1])
	M_z = np.sum(M[:, 2])

	M_pitch = M_y
	M_roll = M_x
	M_yaw = M_z
	F_drag = F_x
	F_lift = F_z
	F_lat = F_y

	C_L = F_lift / (q * A_ref)
	C_D = F_drag / (q * A_ref)
	C_S = F_lat / (q * A_ref)
	C_N = M_yaw / (q * A_ref * L_ref)
	C_A = M_roll / (q * A_ref * L_ref)
	C_M = M_pitch / (q * A_ref * L_ref)

#	C_x = F_x / (q * A_ref)
#	C_y = F_y / (q * A_ref)
#	C_z = F_z / (q * A_ref)
#	C_N = M_yaw / (q * A_ref * L_ref)
#	C_A = M_roll / (q * A_ref * L_ref)
#	C_M = M_pitch / (q * A_ref * L_ref)
#
#	C_D = (C_z * np.cos(alpha) * np.sin(beta)) + \
#		(C_x * np.cos(beta) * np.sin(alpha)) - \
#		(C_y * np.sin(beta))

	return [C_L, C_D, C_S, C_N, C_A, C_M]