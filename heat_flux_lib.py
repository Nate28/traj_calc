# -*- coding: utf-8 -*-
"""
=== HEAT FLUX LIB ===
Heat flux correlations for atmospheric entry calculations.

Created on Tue Jul 21 16:15:10 2015
@author: Nathan Donaldson
"""

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
	q = 5.16E-5 * np.sqrt((rho / R_n) * (U**3.15))
#	q *= 1E6
#	q = 9.81775E6 * ((((0.6096 * rho) / (2 * R_n)))**0.5) * ((U / 3048)**3.15)
	q *= 1E4

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
	conv_params = {'Earth':	[195.13E6, 0.5, -0.5, 3.04, 0, 16E3],
				  'Mars'	:	[206.68E6, 0.5, -0.5, 3.15, 0, 8E3],
				  'Venus':	[206.68E6, 0.5, -0.5, 3.15, 0, 8E3],
				  'Titan':	[195.13E6, 0.5, -0.5, 3.04, 0, 16E3]}

	rad_params = {'Venus' :	[7.7E9, 0.52, 0.48, 9.0, 0, 11E3],
				 'Earth'	:	[65.4E9, 1.6, 1.0, 8.5, 0, 8E3],
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

def fay_riddell(Pr, rho_e, rho_w, mu_e, mu_w, du_dx, h_0, h_wall, Le, h_D):
	### TODO: Input Fay-Riddell correlation from flow_calc_lib
	"""
	J. A. Fay and F. R. Riddell, “Theory of stagnation point heat transfer in
	dissociated air,” Journal of the Aeronautical Sciences, vol. 25, no. 2,
	pp. 73–85, Feb. 1958. [Online]. Available:
	http://pdf.aiaa.org/downloads/TOCPDFs/36_373-386.pdf.
	"""
	q = 0.76 * \
	(Pr**(-0.6)) * \
	((rho_e * mu_e)**0.4) * \
	((rho_w * mu_w)**0.1) * \
	np.sqrt(du_dx) * \
	(h_0 - h_wall) * \
	(1 + (((Le**0.52) - 1) * (h_D / h_0)))

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
	q = 18.8 * np.sqrt(rho / R_n) * (U/1000)**2
	q *= 1E4
	return q

def tauber_sutton(U, rho, R_n, mode='conv'):
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

	if (U <= 11500):
		f_fit = np.array([ -3.93206793e-12,   1.61370008e-07,  -2.43598601e-03,
			1.61078691e+01,  -3.94948753e+04])
	elif (U > 11500):
		f_fit = np.array([-1.00233100E-12, 4.89774670E-8, -8.42982517E-4,
			6.25525796, -17168.3333])

#	f_fit = np.array([ -4.62082721e-13,   2.24407878e-08,  -3.57511472e-04,
#         2.33651065e+00,  -5.39748383e+03])

	f = np.polyval(f_fit, U)
	if (R_n >= 0) and (R_n <= 1):
		a_max = 1
	elif (R_n > 1) and (R_n <= 2):
		a_max = 0.6
#	elif (R_n > 2) and (R_n <= 3):
	elif (R_n > 2):
		a_max = 0.5
	a = np.min(np.array([a_max, (1.072E6 * (U**-1.88) * (rho**-0.325))]))

	q = C * (R_n**a) * (rho**b) * f
	#q *= 1E4

	return q