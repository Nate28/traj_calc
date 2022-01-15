# -*- coding: utf-8 -*-
"""
Created on Sat Sep 26 22:11:55 2015

@author: Nathan
"""

from __future__ import print_function
import numpy as np

try:
	from numba import autojit
except:
	def autojit(a):
		return a

@autojit
def EulerX(beta, mode='rad', clockwise=False):
	# Return the Euler matrix for rotation about the X axis
	if mode == 'deg':
		beta = np.deg2rad(beta)
		
	if not clockwise:
		x_rot_mat = np.array([	[1, 	0, 				0			],
								[0, 	np.cos(beta), 	np.sin(beta)],
								[0, 	-np.sin(beta), 	np.cos(beta)]	])
	elif clockwise:
		x_rot_mat = np.array([	[1, 	0, 				0			],
								[0, 	np.cos(beta),	-np.sin(beta)],
								[0, 	np.sin(beta), 	np.cos(beta)]	])
		
	return x_rot_mat

@autojit
def EulerY(theta, mode='rad', clockwise=False):
	# Return the Euler matrix for rotation about the Y axis
	if mode == 'deg':
		theta = np.deg2rad(theta)
		
	if not clockwise:
		y_rot_mat = np.array([	[np.cos(theta), 0, 	-np.sin(theta)	],
								[0, 			1,	0				],
								[np.sin(theta), 0, 	np.cos(theta)	]	])
	elif clockwise:
		y_rot_mat = np.array([	[np.cos(theta), 0, 	np.sin(theta)	],
								[0, 			1,	0				],
								[-np.sin(theta), 0, 	np.cos(theta)]	])
		
	return y_rot_mat

@autojit
def EulerZ(alpha, mode='rad', clockwise=False):
	# Return the Euler matrix for rotation about the Z axis
	if mode == 'deg':
		theta = np.deg2rad(alpha)
		
	if not clockwise:
		z_rot_mat = np.array([	[np.cos(alpha),		np.sin(alpha),	0],
								[-np.sin(alpha),	np.cos(alpha),	0],
								[0,					0,				1]	])
	elif clockwise:
		z_rot_mat = np.array([	[np.cos(alpha),		-np.sin(alpha),	0],
								[np.sin(alpha),		np.cos(alpha),	0],
								[0,					0,				1]	])
		
	return z_rot_mat

@autojit
def rotx(points, beta, mode='deg'):
	"""
	Rotate points about the global X axis by the angles beta (roll).  Rotation
	matrices are sourced from http://mathworld.wolfram.com/RotationMatrix.html
	"""



	if mode == 'rad':
		pass
	elif mode == 'deg':
		beta	=	np.deg2rad(beta)
	else:
		print('ERROR: Incorrect angle type specified.  Assuming degrees.')

	# Rotation about X axis
	x_rot_mat = np.array([	[1, 	0, 				0			],
							[0, 	np.cos(beta), 	np.sin(beta)],
							[0, 	-np.sin(beta), 	np.cos(beta)]	])

	# Sequentially rotate input points about X, Y and then Z axes
	if np.size(points) == 3:
		rot_x 	= np.dot(points, x_rot_mat)
	else:
		rot_x 	= np.dot(points, x_rot_mat)

	return rot_x

@autojit
def roty(points, theta, mode='deg'):
	"""
	Rotate points about the global Y axis by the angles theta (yaw).  Rotation
	matrices are sourced from http://mathworld.wolfram.com/RotationMatrix.html
	"""

	if mode == 'rad':
		pass
	elif mode == 'deg':
		theta	=	np.deg2rad(theta)
	else:
		print('ERROR: Incorrect angle type specified.  Assuming degrees.')

	# Rotation about Y axis
	y_rot_mat = np.array([	[np.cos(theta), 0, 	-np.sin(theta)	],
							[0, 			1,	0				],
							[np.sin(theta), 0, 	np.cos(theta)	]	])

	# Sequentially rotate input points about X, Y and then Z axes
	if np.size(points) == 3:
		rot_y 	= np.dot(points, y_rot_mat)
	else:
		rot_y 	= np.dot(points, y_rot_mat)

	return rot_y

@autojit
def rotz(points, alpha, mode='deg'):
	"""
	Rotate points about the global Z axis by the angles theta (pitch).  Rotation
	matrices are sourced from http://mathworld.wolfram.com/RotationMatrix.html
	"""



	if mode == 'rad':
		pass
	elif mode == 'deg':
		alpha	=	np.deg2rad(alpha)
	else:
		print('ERROR: Incorrect angle type specified.  Assuming degrees.')

	# Rotation about Z axis
	z_rot_mat = np.array([	[np.cos(alpha),		np.sin(alpha),	0],
							[-np.sin(alpha),	np.cos(alpha),	0],
							[0,					0,				1]	])

	# Sequentially rotate input points about X, Y and then Z axes
	if np.size(points) == 3:
		rot_z 	= np.dot(points, z_rot_mat)
	else:
		rot_z 	= np.dot(points, z_rot_mat)

	return rot_z

@autojit
def rotxyz(points, beta, theta, alpha, mode='deg', clockwise=False):
	"""
	=== triPy.rotxyz ===
	Rotates points about the global X, Y and Z axes by the angles beta, theta,
	and alpha (roll, yaw, and pitch, respectively) in that order.  Rotation
	matrices are sourced from http://mathworld.wolfram.com/RotationMatrix.html

	=== Inputs ===
	'points'	XYZ coordinates of the points to be rotated
	'beta'		Angle of rotation about the X axis (roll)
	'theta'		Angle of rotation about the Y axis (yaw)
	'alpha'		Angle of rotation about the Z axis (pitch)
	'mode'		Defines whether angles are expressed in radians or degrees
				(degrees are the default)
	'clockwise' Defines direction of rotation about axes
				(may be either True or False - the default is False: anticlockwise)

	=== Usage ===
	import triPy
	rot_x, rot_y, rot_z = rotMatrix(x, y, z, beta, theta, alpha)
	"""



	if mode == 'rad':
		pass
	elif mode == 'deg':
		beta		=	np.deg2rad(beta)
		theta	=	np.deg2rad(theta)
		alpha	=	np.deg2rad(alpha)
	else:
		print('ERROR: Incorrect angle type specified.  Assuming degrees.')

	if clockwise == False:
		# Rotation about X axis
		x_rot_mat = np.array([	[1, 	0, 				0			],
								[0, 	np.cos(beta), 	np.sin(beta)],
								[0, 	-np.sin(beta), 	np.cos(beta)]	])

		# Rotation about Y axis
		y_rot_mat = np.array([	[np.cos(theta), 0, 	-np.sin(theta)	],
								[0, 			1,	0				],
								[np.sin(theta), 0, 	np.cos(theta)	]	])

		# Rotation about Z axis
		z_rot_mat = np.array([	[np.cos(alpha),		np.sin(alpha),	0],
								[-np.sin(alpha),		np.cos(alpha),	0],
								[0,					0,				1]	])

	else:
		# Rotation about X axis
		x_rot_mat = np.array([	[1, 	0, 				0			],
								[0, 	np.cos(beta),	-np.sin(beta)],
								[0, 	np.sin(beta), 	np.cos(beta)]	])

		# Rotation about Y axis
		y_rot_mat = np.array([	[np.cos(theta), 0, 	np.sin(theta)	],
								[0, 			1,	0				],
								[-np.sin(theta), 0, 	np.cos(theta)	]	])

		# Rotation about Z axis
		z_rot_mat = np.array([	[np.cos(alpha),		-np.sin(alpha),	0],
								[np.sin(alpha),	np.cos(alpha),	0],
								[0,					0,				1]	])


	# Sequentially rotate input points about X, Y and then Z axes
	if np.size(points) == 3:
		rot_x 	= np.dot(points, x_rot_mat)
		rot_xy 	= np.dot(rot_x, y_rot_mat)
		rot_xyz = np.dot(rot_xy, z_rot_mat)
	else:
		rot_x 	= np.dot(points, x_rot_mat)
		rot_xy 	= np.dot(rot_x, y_rot_mat)
		rot_xyz = np.dot(rot_xy, z_rot_mat)

	return rot_xyz

@autojit
def rotzyx(points, beta, theta, alpha, mode='deg', clockwise=False):
	"""
	=== triPy.rotxyz ===
	Rotates points about the global X, Y and Z axes by the angles beta, theta,
	and alpha (roll, yaw, and pitch, respectively) in reverse order.  Rotation
	matrices are sourced from http://mathworld.wolfram.com/RotationMatrix.html

	=== Inputs ===
	'points'	XYZ coordinates of the points to be rotated
	'beta'		Angle of rotation about the X axis (roll)
	'theta'		Angle of rotation about the Y axis (pitch)
	'alpha'		Angle of rotation about the Z axis (yaw)
	'mode'		Defines whether angles are expressed in radians or degrees
				(degrees are the default)
	'clockwise' Defines direction of rotation about axes
				(may be either True or False - the default is False: anticlockwise)

	=== Usage ===
	import triPy
	rot_x, rot_y, rot_z = rotMatrix(x, y, z, beta, theta, alpha)
	"""

	if mode == 'rad':
		pass
	elif mode == 'deg':
		beta		=	np.deg2rad(beta)
		theta	=	np.deg2rad(theta)
		alpha	=	np.deg2rad(alpha)
	else:
		print('ERROR: Incorrect angle type specified.  Assuming degrees.')

	if clockwise == False:
		# Rotation about X axis
		x_rot_mat = np.array([	[1, 	0, 				0			],
								[0, 	np.cos(beta), 	np.sin(beta)],
								[0, 	-np.sin(beta), 	np.cos(beta)]	])

		# Rotation about Y axis
		y_rot_mat = np.array([	[np.cos(theta), 0, 	-np.sin(theta)	],
								[0, 			1,	0				],
								[np.sin(theta), 0, 	np.cos(theta)	]	])

		# Rotation about Z axis
		z_rot_mat = np.array([	[np.cos(alpha),		np.sin(alpha),	0],
								[-np.sin(alpha),	np.cos(alpha),	0],
								[0,					0,				1]	])

	else:
		# Rotation about X axis
		x_rot_mat = np.array([	[1, 	0, 				0			],
								[0, 	np.cos(beta),	-np.sin(beta)],
								[0, 	np.sin(beta), 	np.cos(beta)]	])

		# Rotation about Y axis
		y_rot_mat = np.array([	[np.cos(theta), 0, 	np.sin(theta)	],
								[0, 			1,	0				],
								[-np.sin(theta), 0, 	np.cos(theta)	]	])

		# Rotation about Z axis
		z_rot_mat = np.array([	[np.cos(alpha),		-np.sin(alpha),	0],
								[np.sin(alpha),	np.cos(alpha),	0],
								[0,					0,				1]	])


	# Sequentially rotate input points about X, Y and then Z axes
	if np.size(points) == 3:
		rot_z 	= np.dot(points, z_rot_mat)
		rot_zy 	= np.dot(rot_z, y_rot_mat)
		rot_zyx = np.dot(rot_zy, x_rot_mat)
	else:
		rot_z 	= np.dot(points, z_rot_mat)
		rot_zy 	= np.dot(rot_z, y_rot_mat)
		rot_zyx = np.dot(rot_zy, x_rot_mat)

	return rot_zyx

@autojit
def eulerToVector(alpha, theta):
	"""
	Convert Euler angles (pitch and yaw; alpha and theta) to vector components
	in XYZ reference frame.
	"""

	x = np.sin(alpha) * np.cos(theta)
	y = np.sin(alpha) * np.sin(theta)
	z = np.cos(alpha)

	return [x, y, z]

@autojit
def vectorToEuler(x, y, z):
	"""
	Convert XYZ vector components to Euler angles (pitch and yaw; alpha and
	theta)
	"""

	# Trig relations
	r = np.linalg.norm([x, y, z])
	theta = np.arctan(y / x)
	alpha = np.arccos(z / r)

#	1st quad: 0
#	2nd quad: 180-
#	3rd quad: +180
#	4th quad: 360-

	# Identify quadrants in XY plane and adjust returned angle accordingly
	# Rotation about Z axis (yaw, theta)
	if (x > 0) & (y > 0):
		# 1st quadrant
		pass
	elif (x < 0) & (y > 0):
		# 2nd quadrant
		theta = np.pi - np.abs(theta)
	elif (x < 0) & (y < 0):
		# 3rd quadrant
		theta = np.pi + np.abs(theta)
	elif (x > 0) & (y < 0):
		# 4th quadrant
		theta = (2 * np.pi) - np.abs(theta)
	else:
		# Quadrant borders i.e. x = 0 or y = 0
		pass

	# Identify quadrants in XZ plane and adjust returned angle accordingly
	# Rotation about Y axis (pitch, alpha)
	if (x > 0) & (z > 0):
		# 1st quadrant
		pass
	elif (x < 0) & (z > 0):
		# 2nd quadrant
		alpha = np.pi - np.abs(alpha)
	elif (x < 0) & (z < 0):
		# 3rd quadrant
		alpha = np.pi + np.abs(alpha)
	elif (x > 0) & (z < 0):
		# 4th quadrant
		alpha = (2 * np.pi) - np.abs(alpha)
	else:
		# Quadrant borders i.e. x = 0 or y = 0
		pass

	return [alpha, theta]

#@autojit
def localPitchRoll(pitch, yaw, roll, clockwise=False):
	"""
	Calculate pitch and roll of an object in its local vetical plane
	NB: pitch, yaw, and roll denote rotations about the Y, Z, and X axes,
	respectively.
	"""

	# Define initial pointing vectors
	vecX = np.array([1, 0, 0])
	vecZ = np.array([0, 0, 1])

	# Transform vectors using provided Euler angles
	vecX1 = rotxyz(vecX, roll, pitch, yaw, mode='rad', clockwise=clockwise)
	vecZ1 = rotxyz(vecZ, roll, pitch, yaw, mode='rad', clockwise=clockwise)

	# Project vectors onto local YZ plane
	vecX1_proj = np.array([0, vecX1[1], vecX1[2]])
	vecZ_proj = np.array([0, 0, np.linalg.norm(vecX1_proj)])
	
	# Calculate angle to roll vectors back into XY plane
#	planeRoll = np.arccos(np.dot(vecX1, vecZ_proj) / (np.linalg.norm(vecX1_proj)**2))
	planeRoll = vectorAngle(vecX1_proj, vecZ_proj)

	# Calculate local pitch angle (about local Y axis)
	vecZ2 = rotxyz(vecZ1, planeRoll, 0, 0, mode='rad', clockwise=clockwise)
#	localPitch = np.arccos(np.dot(vecX1, [1, 0, 0]) / (np.linalg.norm(vecX1)))
	localPitch = vectorAngle(vecX1, vecX)

	# Transform normal vector into ZY plane (pitch) to find local roll
	vecZ3 = rotxyz(vecZ2, 0, -localPitch, 0, mode='rad', clockwise=clockwise)
#	localRoll = np.arccos(np.dot(vecZ3, [0, 0, 1]) / (np.linalg.norm(vecZ3)))
	localRoll = vectorAngle(vecZ3, vecZ)

	return localPitch, localRoll, planeRoll

@autojit
def vectorAngle(a, b):
	"""
	Return angle between two vectors
	"""
	return np.arccos(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
