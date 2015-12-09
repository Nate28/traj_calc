# -*- coding: utf-8 -*-
"""
Created on Sat Sep 26 22:11:55 2015

@author: Nathan
"""

import numpy as np

def rotx(points, beta, mode='deg'):
	"""
	Rotate points about the global X axis by the angles beta (roll).  Rotation 
	matrices are sourced from http://mathworld.wolfram.com/RotationMatrix.html
	"""
	
	import numpy as np
	
	if mode == 'rad':
		pass
	elif mode == 'deg':
		beta	=	np.deg2rad(beta)
	else:
		print 'ERROR: Incorrect angle type specified.  Assuming degrees.'
	
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
	
def roty(points, theta, mode='deg'):
	"""
	Rotate points about the global Y axis by the angles theta (yaw).  Rotation 
	matrices are sourced from http://mathworld.wolfram.com/RotationMatrix.html
	"""
	
	import numpy as np
	
	if mode == 'rad':
		pass
	elif mode == 'deg':
		theta	=	np.deg2rad(theta)
	else:
		print 'ERROR: Incorrect angle type specified.  Assuming degrees.'
	
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
	
def rotz(points, alpha, mode='deg'):
	"""
	Rotate points about the global Z axis by the angles theta (pitch).  Rotation 
	matrices are sourced from http://mathworld.wolfram.com/RotationMatrix.html
	"""
	
	import numpy as np
	
	if mode == 'rad':
		pass
	elif mode == 'deg':
		alpha	=	np.deg2rad(alpha)
	else:
		print 'ERROR: Incorrect angle type specified.  Assuming degrees.'
	
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
	
def euler_to_vector(alpha, theta):
	"""
	Convert Euler angles (pitch and yaw; alpha and theta) to vector components 
	in XYZ reference frame.
	"""	
	
	x = np.sin(alpha) * np.cos(theta)
	y = 	np.sin(alpha) * np.sin(theta)
	z = np.cos(alpha)
	
	return [x, y, z]
	
def vector_to_euler(x, y, z):
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