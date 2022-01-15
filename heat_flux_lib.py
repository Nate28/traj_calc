# -*- coding: utf-8 -*-
"""
=== HEAT FLUX LIB ===
Heat flux correlations for atmospheric entry calculations.

Created on Tue Jul 21 16:15:10 2015
@author: Nathan Donaldson
"""

from __future__ import print_function
import numpy as np

__author__ = 'Nathan Donaldson'
__email__ = 'nathan.donaldson@eng.ox.ac.uk'
__status__ = 'Development'
__version__ = '0.3'
__license__ = 'MIT'

def detra_hidalgo(U, rho, R_n):
	### COMPLETE
	"""
	R. W. Detra and H. Hidalgo, “Generalized heat transfer formulas and graphs
	for nose cone re-entry into the atmosphere,” ARS Journal, vol. 31,
	pp. 318–321, 1961.
	"""
#	q = 5.16E-5 * np.sqrt((rho / R_n) * (U**3.15))
#	q *= 1E6
	q = 9.81775E6 * ((((0.6096 * rho) / (2 * R_n)))**0.5) * ((U / 3048)**3.15)
	#q *= 1E4

#For detra and hidalgo I have in the traj manual
#
#q[W/cm**2] = 9.81775E6 * ((0.6096*rho[kg/m**3] / 2R[m]))**0.5 * ((U[m/s]/3048)**3.15)

	return q

def brandis_johnston(U, rho, R_n, mode='conv'):
	### COMPLETE
	"""
	A. M. Brandis and C. O. Johnston, “Characterization of stagnation-point
	heat flux for earth entry,” in AIAA Aviation, Atlanta, Georgia, USA: AIAA,
	Jun. 2014
	"""

	if mode == 'conv':
		# Convective heating correlation
#		if (U < 9.5E3) and (U >= 3E3):
#			q = 4.502E-9 * (rho**0.4704) * (U**3.147) * (R_n**-0.5038)
#		elif (U >= 9.5E3) and (U <= 17E3):
#			q = 1.270E-6 * (rho**0.4678) * (U**2.524) * (R_n**-0.5397)
#		elif (U < 3E3) or (U > 17):
#			q = 0

		if (U < 9.5E3):
			q = 4.502E-9 * (rho**0.4704) * (U**3.147) * (R_n**-0.5038)
		elif (U >= 9.5E3):
			q = 1.270E-6 * (rho**0.4678) * (U**2.524) * (R_n**-0.5397)

	elif mode == 'rad':
		# Radiative heating correlation
		C = 3.416E4
		b = 1.261
		f = -53.26 + (6555 / (1 + ((16000 / U)**8.25)))

		if (R_n >= 0) and (R_n <= 0.5):
			a_max = 0.61
		elif (R_n > 0.5) and (R_n <= 2):
			a_max = 1.23
		elif (R_n > 2) and (R_n <= 10):
			a_max = 0.49

		a = np.min(np.array([a_max, (3.175E6 * (U**-1.80) * (rho**-0.1575))]))
		q = C * (R_n**a) * (rho**b) * f

	if q != 0:
		# Convert from W/cm**2 to W/m**2
		q *= 1E4

	return q

def smith(U, rho, R_n, mode='conv', planet='Earth'):
	### AGARD Report 808
	"""
	Smith, A., "Heat Transfer for Perfect Gas and Chemically Reacting Flows",
	von Karman Institute of Fluid Dynamics, Belgium, AGARD Report 808,
	pg.3.1 - 3.14, March 1995.  Available:
	https://www.cso.nato.int/Pubs/rdp.asp?RDP=AGARD-R-808
	"""
	conv_params = {'Venus':	[195.13E6, 0.5, -0.5, 3.04, 0, 16E3],
				  'Earth':	[206.68E6, 0.5, -0.5, 3.15, 0, 8E3],
				  'Titan':	[206.68E6, 0.5, -0.5, 3.15, 0, 8E3],
				  'Mars':	[195.13E6, 0.5, -0.5, 3.04, 0, 16E3]}

	rad_params = {'Venus' :	[7.7E9, 0.52, 0.48, 9.0, 0, 11E3],
				 'Earth'	:	[6.54E10, 1.6, 1.0, 8.5, 0, 8E3],
				 'Mars'	:	[3.84E4, 1.16, 0.56, 21.5, 0, 7E3],
				 'Titan'	:	[8.83E12, 1.65, 1.0, 5.6, 4E3, 7E3]}

	if mode == 'conv':
		k, a, b, c, range_i, range_f = conv_params[planet]
	elif mode == 'rad':
		k, a, b, c, range_i, range_f = rad_params[planet]

	q = k * (rho**a) * (R_n**b) * ((U / 1E4)**c)

	# Convert from W/cm**2 to W/m**2
	#q *= 1E4

	return q

class FayRiddellHelper:
	def __init__(self, gas_file='air.xml', d=4e-10, thetav=5500.0):
		import flow_calc_lib as fcl
		import cantera as ct
		
		# Generate gas property calculation object
		gas = ct.Solution(gas_file)
		
		self.gas = gas
		self.gas_file = gas_file
		self.d = d
		self.thetav = thetav
		
		return None
		
	def calculate(self, MaInf, pInf, TInf, Tw, Rn, geom='sph', chem='equil'):
		"""
		Fay-Riddell heat flux calculator function.  Computes heat flux at the 
		stagnation point of a sphere or cylinder in supersonic/hypersonic flow.
		
		Inputs:
			MaInf	: Freestream Mach number
			pInf 	: Freestream static pressure
			TInf 	: Fresstream static temperature
			Tw 		: Wall temperature
			Rn 		: Nose radius
			geom 	: Geometry mode.  Adjusts the factor by which predictions are
				multiplied before being returned.  Must be one of 'sph' (for a 
				sphere), or 'cyl' (for a cylinder).  The default is 'sph'.
			chem 	: Wall chemical reaction mode.  Must be one of 'equil' (for a
				boundary layer at chemical equilibrium), 'catFr' (for a frozen
				boundary layer with a fully catalytic wall), or 'nonCatFr' (for a
				frozen boundary layer with a non-catalytic wall).  The default is 
				'equil'.
				
		Output:
			q 		: Predicted heat flux at the stagnation point of a sphere
		"""
		
# 		import flow_calc_lib as fcl
# 		import cantera as ct
		
		gas = self.gas
		d = self.d
		thetav = self.thetav
		
		# Freestream gas properties
		gas.TP = TInf, pInf
		gamma = gas.cp / gas.cv
		R = gas.cp - gas.cv
		
		# Freestream total quantities
		p0 = fcl.p_stag(Ma=MaInf, gamma_var=gamma, p=pInf)
		T0 = fcl.T_stag(Ma=MaInf, gamma_var=gamma, T=TInf)
		
		# Post-shock conditions (taken as wall conditions except for T)
		pr, Tr, p0r, T0r, rhor, Ma_2, _ = fcl.normal_shock_ratios(MaInf, gamma)
		p2 = pInf * pr
		p02 = p0 * p0r
		T02 = T0
		
		# External flow conditions (evaluated at stagnation conditions)
		gas.TP = T02, p02
		rho_e = gas.density
		mu_e = gas.viscosity
		gamma_e = gas.cp / gas.cv
	#	rho_e = p02 / (R * T02)
	#	mu_e = fcl.viscosity(mode='Sutherland', gas='air', T=T0)
		h_0 = fcl.enthalpy(gamma_var=gamma_e, R=R, T=T0)
		
		# Wall conditions
		gas.TP = Tw, p02
		rho_w = gas.density
		mu_w = gas.viscosity
		gamma_w = gas.cp / gas.cv
	#	rho_w = p02 / (R * Tw)
	#	mu_w = fcl.viscosity(mode='Sutherland', gas='air', T=Tw)
		h_wall = fcl.enthalpy(gamma_var=gamma_w, theta_v=thetaV, R=R, T=Tw)
		
		# Chemical properties of gas at wall
		#air_w.TP = (T_inf, p_inf)
		k_w = gas.thermal_conductivity
		C_p_w = gas.cp
		mu_w = gas.viscosity
		D_im = np.mean(gas.mix_diff_coeffs_mass)
		#m_air = (air_w.mean_molecular_weight) * 1E-3
		#N_A = ct.avogadro
		
		# Standard enthalpy of formation (mass fraction of each species multiplied
		# by respective change in enthalpy for reactions)
		# NB/ Returned in J/kmol, so conversion to J/kg is necessary
		h_D_kmol = np.sum(gas.Y * gas.delta_enthalpy) # J/kmol
		molW = np.sum(gas.Y * gas.molecular_weights) # g/mol = kg/kmol
		h_D = h_D_kmol / molW
		
		# Non-dimensional numbers (NB/ Lewis number has a HUGE impact on final Q_dot)
		# Assumption of Sc = 0.5 is from Nathan Joiner, FGE
		Pr_w = fcl.Prandtl(C_p=C_p_w, k=k_w, mu=mu_w)
	#	Sc = 0.5
		Sc = fcl.Schmidt(mu=mu_w, rho=rho_w, D=D_im)
		Le = fcl.Lewis(Pr=Pr_w, Sc=Sc)
	#	Le = fcl.Lewis(k=k_w, rho=rho_w, D_im=D_im, Cp=C_p_w)
		
		# Velocity gradient
		du_dx = fcl.vel_gradient(p_inf=pInf, p_0=p02, R_n=Rn, rho=rho_e)
	
		# Original Fay-Riddell
		if chem == 'nonCatFr':
			q = fay_riddell_nonCatFr(Pr_w, rho_e, rho_w, mu_e, mu_w, du_dx, 
				h_0, h_wall, h_D, mode=geom)
		elif chem == 'catFr':
			q = fay_riddell_catFr(Pr_w, rho_e, rho_w, mu_e, mu_w, du_dx, h_0, 
				h_wall, Le, h_D, mode=geom)
		elif chem == 'equil':
			q = fay_riddell_equil(Pr_w, rho_e, rho_w, mu_e, mu_w, du_dx, h_0, 
				h_wall, Le, h_D, mode=geom)
		else:
			print('ERROR: INCORRECT CHEMISTRY MODE DEFINED')
	
	#	# Zuppardi polynomials
	#	q = fay_riddell_zuppardi(h_0, h_wall, mode=chem)
		
		return q
		
def fay_riddell_helper(MaInf, pInf, TInf, Tw, Rn, geom='sph', chem='equil'):
	"""
	Fay-Riddell heat flux calculator function.  Computes heat flux at the 
	stagnation point of a sphere or cylinder in supersonic/hypersonic flow.
	
	Inputs:
		MaInf	: Freestream Mach number
		pInf 	: Freestream static pressure
		TInf 	: Fresstream static temperature
		Tw 		: Wall temperature
		Rn 		: Nose radius
		geom 	: Geometry mode.  Adjusts the factor by which predictions are
			multiplied before being returned.  Must be one of 'sph' (for a 
			sphere), or 'cyl' (for a cylinder).  The default is 'sph'.
		chem 	: Wall chemical reaction mode.  Must be one of 'equil' (for a
			boundary layer at chemical equilibrium), 'catFr' (for a frozen
			boundary layer with a fully catalytic wall), or 'nonCatFr' (for a
			frozen boundary layer with a non-catalytic wall).  The default is 
			'equil'.
			
	Output:
		q 		: Predicted heat flux at the stagnation point of a sphere
	"""
	
	import flow_calc_lib as fcl
	import cantera as ct
	
	# Generate gas property calculation object
	gas = ct.Solution('air.xml')
	
	d = 4E-10
	thetaV = 5500.0
#	thetaV = TInf
	
	# Freestream gas properties
	gas.TP = TInf, pInf
	gamma = gas.cp / gas.cv
	R = gas.cp - gas.cv
	
	# Freestream total quantities
	p0 = fcl.p_stag(Ma=MaInf, gamma_var=gamma, p=pInf)
	T0 = fcl.T_stag(Ma=MaInf, gamma_var=gamma, T=TInf)
	
	# Post-shock conditions (taken as wall conditions except for T)
	pr, Tr, p0r, T0r, rhor, Ma_2, _ = fcl.normal_shock_ratios(MaInf, gamma)
	p2 = pInf * pr
	p02 = p0 * p0r
	T02 = T0
	
	# External flow conditions (evaluated at stagnation conditions)
	gas.TP = T02, p02
	rho_e = gas.density
	mu_e = gas.viscosity
	gamma_e = gas.cp / gas.cv
#	rho_e = p02 / (R * T02)
#	mu_e = fcl.viscosity(mode='Sutherland', gas='air', T=T0)
	h_0 = fcl.enthalpy(gamma_var=gamma_e, R=R, T=T0)
	
	# Wall conditions
	gas.TP = Tw, p02
	rho_w = gas.density
	mu_w = gas.viscosity
	gamma_w = gas.cp / gas.cv
#	rho_w = p02 / (R * Tw)
#	mu_w = fcl.viscosity(mode='Sutherland', gas='air', T=Tw)
	h_wall = fcl.enthalpy(gamma_var=gamma_w, theta_v=thetaV, R=R, T=Tw)
	
	# Chemical properties of gas at wall
	#air_w.TP = (T_inf, p_inf)
	k_w = gas.thermal_conductivity
	C_p_w = gas.cp
	mu_w = gas.viscosity
	D_im = np.mean(gas.mix_diff_coeffs_mass)
	#m_air = (air_w.mean_molecular_weight) * 1E-3
	#N_A = ct.avogadro
	
	# Standard enthalpy of formation (mass fraction of each species multiplied
	# by respective change in enthalpy for reactions)
	# NB/ Returned in J/kmol, so conversion to J/kg is necessary
	h_D_kmol = np.sum(gas.Y * gas.delta_enthalpy) # J/kmol
	molW = np.sum(gas.Y * gas.molecular_weights) # g/mol = kg/kmol
	h_D = h_D_kmol / molW
	
	# Non-dimensional numbers (NB/ Lewis number has a HUGE impact on final Q_dot)
	# Assumption of Sc = 0.5 is from Nathan Joiner, FGE
	Pr_w = fcl.Prandtl(C_p=C_p_w, k=k_w, mu=mu_w)
#	Sc = 0.5
	Sc = fcl.Schmidt(mu=mu_w, rho=rho_w, D=D_im)
	Le = fcl.Lewis(Pr=Pr_w, Sc=Sc)
#	Le = fcl.Lewis(k=k_w, rho=rho_w, D_im=D_im, Cp=C_p_w)
	
	# Velocity gradient
	du_dx = fcl.vel_gradient(p_inf=pInf, p_0=p02, R_n=Rn, rho=rho_e)

	# Original Fay-Riddell
	if chem == 'nonCatFr':
		q = fay_riddell_nonCatFr(Pr_w, rho_e, rho_w, mu_e, mu_w, du_dx, 
			h_0, h_wall, h_D, mode=geom)
	elif chem == 'catFr':
		q = fay_riddell_catFr(Pr_w, rho_e, rho_w, mu_e, mu_w, du_dx, h_0, 
			h_wall, Le, h_D, mode=geom)
	elif chem == 'equil':
		q = fay_riddell_equil(Pr_w, rho_e, rho_w, mu_e, mu_w, du_dx, h_0, 
			h_wall, Le, h_D, mode=geom)
	else:
		print('ERROR: INCORRECT CHEMISTRY MODE DEFINED')

#	# Zuppardi polynomials
#	q = fay_riddell_zuppardi(h_0, h_wall, mode=chem)
	
	return q

def fay_riddell_adams(Pr_w, rho_e, rho_w, mu_e, mu_w, du_dx, h_0, h_wall, 
	Le, h_D, mode='sph'):
	"""
	J. A. Fay and F. R. Riddell, “Theory of stagnation point heat transfer in
	dissociated air,” Journal of the Aeronautical Sciences, vol. 25, no. 2,
	pp. 73–85, Feb. 1958. [Online]. Available:
	http://pdf.aiaa.org/downloads/TOCPDFs/36_373-386.pdf.
	
	Calculation copied from EVBSPBLC spreadsheet by John C. Adams
	
	NB/ All quantities must be in Imperial units
	"""
	
	if mode == 'sph':
		A = 0.76
	elif mode == 'cyl':
		A = 0.53
	
	# Equilibrium boundary layer
	q = A * \
	(Pr_w**(-0.6)) * \
	((rho_e * mu_e)**0.4) * \
	((rho_w * mu_w)**0.1) * \
	np.sqrt(du_dx) * \
	(h_0 - h_wall) * \
	778.17

	return q

def fay_riddell_equil(Pr_w, rho_e, rho_w, mu_e, mu_w, du_dx, h_0, h_wall, 
	Le, h_D, mode='sph'):
	"""
	J. A. Fay and F. R. Riddell, “Theory of stagnation point heat transfer in
	dissociated air,” Journal of the Aeronautical Sciences, vol. 25, no. 2,
	pp. 73–85, Feb. 1958. [Online]. Available:
	http://pdf.aiaa.org/downloads/TOCPDFs/36_373-386.pdf.
	"""
	
	if mode == 'sph':
		A = 0.76
	elif mode == 'cyl':
		A = 0.53

	# Equilibrium boundary layer
	q = A * \
	(Pr_w**(-0.6)) * \
	((rho_e * mu_e)**0.4) * \
	((rho_w * mu_w)**0.1) * \
	np.sqrt(du_dx) * \
	(h_0 - h_wall) * \
	(1 + (((Le**0.52) - 1) * (h_D / h_0)))

	return q

def fay_riddell_catFr(Pr_w, rho_e, rho_w, mu_e, mu_w, du_dx, h_0, h_wall, 
	Le, h_D, mode='sph'):
	"""
	J. A. Fay and F. R. Riddell, “Theory of stagnation point heat transfer in
	dissociated air,” Journal of the Aeronautical Sciences, vol. 25, no. 2,
	pp. 73–85, Feb. 1958. [Online]. Available:
	http://pdf.aiaa.org/downloads/TOCPDFs/36_373-386.pdf.
	"""

	if mode == 'sph':
		A = 0.76
	elif mode == 'cyl':
		A = 0.53
	
	# Equilibrium boundary layer, catalytic wall
	q = A * \
	(Pr_w**(-0.6)) * \
	((rho_e * mu_e)**0.4) * \
	((rho_w * mu_w)**0.1) * \
	np.sqrt(du_dx) * \
	(h_0 - h_wall) * \
	(1 + (((Le**0.63) - 1) * (h_D / h_0)))

	return q

def fay_riddell_nonCatFr(Pr_w, rho_e, rho_w, mu_e, mu_w, du_dx, h_0, h_wall,
	h_D, mode='sph'):
	"""
	J. A. Fay and F. R. Riddell, “Theory of stagnation point heat transfer in
	dissociated air,” Journal of the Aeronautical Sciences, vol. 25, no. 2,
	pp. 73–85, Feb. 1958. [Online]. Available:
	http://pdf.aiaa.org/downloads/TOCPDFs/36_373-386.pdf.
	"""

	if mode == 'sph':
		A = 0.76
	elif mode == 'cyl':
		A = 0.53
	
	# Non-catalytic wall, frozen boundary layer
	q = A * \
	(Pr_w**(-0.6)) * \
	((rho_e * mu_e)**0.4) * \
	((rho_w * mu_w)**0.1) * \
	np.sqrt(du_dx) * \
	(h_0 - h_wall) * \
	(1 - (h_D / h_0))

	return q

def fay_riddell_zuppardi(h_0, h_w, mode='equil'):
	"""
	Polynomial fits of the Fay-Riddell equation.  Input parameters are
	freestream total enthalpy, h_0, flow enthalpy at the wall, h_w, and
	the chemistry mode, ('equil', 'catFr', or 'nonCatFr').
	
	Zuppardi, G., and Esposito, A., "Recasting the Fay-Riddell formulae for 
	computing the stagnation point heat flux," Proceedings of the Institution 
	of Mechanical Engineers, Vol. 214, 2000, pp. 115-120.
	"""
	
	if mode == 'nonCatFr':
		q = ((3.992 / (h_0**0.8076)) + \
			(3.458E-6 * (h_0**0.1924)) + \
			(1.276E-14 * (h_0**1.1924))) * \
			(h_0 - h_w)
	elif mode == 'catFr':
		q = ((-18.614 / (h_0**0.8924)) + \
			(2.818E-5 * (h_0**0.1076)) - \
			(5.966E-14 * (h_0**1.1076))) * \
			(h_0 - h_w)		
	elif mode == 'equil':
		q = ((-20.555 / (h_0**0.8932)) + \
			(2.902E-5 * (h_0**0.1068)) - \
			(6.588E-14 * (h_0**1.1068))) * \
			(h_0 - h_w)
	else:
		print('ERROR: INCORRECT MODE SELECTED')
	
	return q

def detra_kemp_ridell(U, rho, R_n, Cp, T_0):
	"""
	Detra, R.W., Kemp, N.H., and Riddell, F.R., “Addendum to
	Heat Transfer to Satellite Vehicles Reentering the Atmosphere,”
	Jet Propulsion, Vol. 27, 1957, pp. 1256-1257.
	"""
	q = 5.7E-7 * ((((1 + j) * (rho/1000)) / R_n)**0.5) * ((U / 100)**3.25) * \
		(1 - ((0.035 * Cp * T_0) / (U**2)))

#	q = (199.87E6 / (T_0 - 300)) *
#		((0.3048 / R_n)**0.5) * ((rho / 1.225)**0.5) * ((U / 7924.8)**3.15)
	return q

def van_driest(rho_s, mu_s, h_s, h_w, du_dx):
	q = 0.94 * ((rho_s * mu_s)**0.5) * (h_s - h_w) * ((du_dx)**0.5)
	return q

def fenstera(U, rho, R_n, rho_0):
	q = 0.635E-69 * ((1 / R_n)**0.5) * ((rho_0 / rho)**0.5) * (U**2.862)
	return q

def sutton_graves(U, rho, R_n):
	### COMPLETE
	"""
	K. Sutton and R. Graves, "A general stagnation-point convective heating
	equation for arbitrary gas mixtures," Techncical Report NASA TR R-376,
	1971.
	"""
	#q = 18.8 * np.sqrt(rho / R_n) * (U/1000)**2
	q = 1.83E-4 * np.sqrt(rho / R_n) * ((U)**3)
	#q /= 1E4
	return q

def tauber_sutton(U, rho, R_n, rho_ratio):
	### COMPLETE
	"""
	M. Tauber and K. Sutton, "Stagnation-Point radiative Heating Relations for
	Earth and Mars Entries," Journal of Spacecraft and Rockets, Vol. 28,
	No. 1, 1991, pp. 40-42.

	S. Surzhikov and M. Shuvalov, "Estimation of Radiation-Convection Heating
	of Four Types of Reentry Spacecrafts," Physical-Chemical Kinetics in Gas
	Dynamics, vol. 15, Issue 4, Lomonosov Moscow State University, 2014.
	"""

	#if mode == 'conv':
	C = 4.736E4
	b = 1.22

	if (U <= 7620):
		g1 = 372.6
		g2 = 8.5
		g3 = 1.6

		q = g1 * R_n * ((3.28084E-4 * U)**g2) * ((1 / rho_ratio)**g3)

	elif (U <= 9000) and (U > 7620):
		g1 = 25.34
		g2 = 12.5
		g3 = 1.78

		q = g1 * R_n * ((3.28084E-4 * U)**g2) * ((1 / rho_ratio)**g3)

	elif (U <= 11500) and (U > 9000):
#	if (U <= 11500):
		f_fit = np.array([ -3.93206793e-12,   1.61370008e-07,  -2.43598601e-03,
			1.61078691e+01,  -3.94948753e+04])

		f = np.polyval(f_fit, U)

		if (R_n >= 0) and (R_n <= 1):
			a_max = 1
		elif (R_n > 1) and (R_n <= 2):
			a_max = 0.6
		elif (R_n > 2):
			a_max = 0.5

		a = np.min(np.array([a_max, (1.072E6 * (U**-1.88) * (rho**-0.325))]))

		q = C * (R_n**a) * (rho**b) * f

	elif (U > 11500):
		f_fit = np.array([-1.00233100E-12, 4.89774670E-8, -8.42982517E-4,
			6.25525796, -17168.3333])

		f = np.polyval(f_fit, U)

		if (R_n >= 0) and (R_n <= 1):
			a_max = 1
		elif (R_n > 1) and (R_n <= 2):
			a_max = 0.6
		elif (R_n > 2):
			a_max = 0.5

		a = np.min(np.array([a_max, (1.072E6 * (U**-1.88) * (rho**-0.325))]))

		q = C * (R_n**a) * (rho**b) * f


#Tauber-Sutton checks out with the traj manual. For nose radii less than 1m, 'a' is implemented as the lesser of 1.072e6*v^-1.88*rho^-0.325 and 0.6, other wise the lesser of the expression and 0.5.

#	f_fit = np.array([ -4.62082721e-13,   2.24407878e-08,  -3.57511472e-04,
#         2.33651065e+00,  -5.39748383e+03])

#	if U < 9000:
#		f = 1.5
#	else:
#		f = np.polyval(f_fit, U)
#
#	if (R_n >= 0) and (R_n <= 1):
#		a_max = 1
#	elif (R_n > 1) and (R_n <= 2):
#		a_max = 0.6
##	elif (R_n > 2) and (R_n <= 3):
#	elif (R_n > 2):
#		a_max = 0.5
#
#	a = np.min(np.array([a_max, (1.072E6 * (U**-1.88) * (rho**-0.325))]))
#
#	q = C * (R_n**a) * (rho**b) * f

	q *= 1E4

	return q

def stefan_boltzmann(epsilon, T):
	theta = 5.670373E-8
	q_rad = epsilon * theta * (T**4)
	return q_rad

def thermal_balance(q_in, T_init, m, Cp, epsilon):
	"""
	Models bulk thermal balance in an object by using a simple forward Euler
	scheme and the Stefan-Boltzmann blackbody radiation equation to find net
	heat transfer and hence bulk temperature as a function of time.
	"""
	length = len(q_in)

	T = np.zeros(length)
	q_out = np.zeros(length)
	#q_in = np.zeros(length)
	q_net = np.zeros(length)

	T[0] = T_init
	q_out[0] = stefan_boltzmann(epsilon, T[0])
	q_net[0] = q_in[0] - q_out[0]

	for i in range(1, length):
		T[i] = T[i-1] + (q_net[i-1] / (m * Cp))
		q_out[i] = stefan_boltzmann(epsilon, T[i])
		q_net[i] = q_in[i] - q_out[i]

	return [T, q_in, q_out, q_net]

def thermal_balance_2(q_in, epsilon):
	"""
	Models bulk thermal balance in an object by using a simple forward Euler
	scheme and the Stefan-Boltzmann blackbody radiation equation to find net
	heat transfer and hence bulk temperature as a function of time.
	"""
	theta = 5.670373E-8

	T = (q_in / (epsilon * theta))**(1.0/4.0)

	return [T]
