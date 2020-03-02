""" CS4277/CS5477 Lab 2: Camera Calibration.
See accompanying Jupyter notebook (lab2.ipynb) for instructions.

Name: Tseng Yu-Ting
Email: E0503474@u.nus.edu
NUSNET ID: E0503474
"""

import cv2
import numpy as np
from scipy.optimize import least_squares

_COLOR_RED   = (255, 0, 0)
_COLOR_GREEN = (0, 255, 0)
_COLOR_BLUE  = (0, 0, 255)


### Helper Function 1 - Convert a 3x3 matrix into a rotation matrix
def convt2rotation(Q):
	u, _, v = np.linalg.svd(Q)
	return np.dot(u, v)

### Helper Function 2 - Convert the vector to matrix
def skew(a):
	s = np.array([[0, -a[2, 0], a[1, 0]], [a[2, 0], 0, -a[0, 0]], [-a[1, 0], a[0, 0], 0]])
	return s

def vector2matrix(S):
	S = np.expand_dims(S, axis = 1)
	
	den =  1 + np.dot(S.T, S)
	num = (1 - np.dot(S.T, S)) * (np.eye(3)) + 2 * skew(S) + 2 * np.dot(S, S.T)
	R = num / den
	
	R = np.hstack((R, np.zeros((3, 1), dtype = np.float32)))
	return R


### Helper Function 3 - Convert the matrix to vector
def matrix2quaternion(r):
	r = r[:3, :3]
	diff = r - r.T

	t = np.zeros(3)
	t[0] = -diff[1, 2]
	t[1] =  diff[0, 2]
	t[2] = -diff[0, 1]
	t_0  = np.divide(t, np.linalg.norm(t) + np.finfo(np.float32).eps)

	sin   =  np.linalg.norm(t) / 2
	cos   = (np.trace(r) - 1)  / 2
	theta = np.arctan2(sin, cos)

	q = np.zeros(4)
	q[0]  = np.cos(theta / 2)
	q[1:] = t_0 * np.sin(theta / 2)
	return q

def matrix2vector(R):
	Q = matrix2quaternion(R)
	return Q[1:] / Q[0]


### Problem 2 - Find the intrinsic parameters matK
def intrinsic(matHs):
	num = len(matHs)
	matHs = np.array(matHs)

	# Generate constraints of matV
	v_ij = {}
	v_ij[(0, 0)] = np.zeros((num, 6))
	v_ij[(0, 1)] = np.zeros((num, 6))
	v_ij[(1, 1)] = np.zeros((num, 6))

	for (i, j) in v_ij:
		v_ij[(i, j)][:, 0] = matHs[:, 0, i] * matHs[:, 0, j]
		v_ij[(i, j)][:, 1] = matHs[:, 0, i] * matHs[:, 1, j] + matHs[:, 1, i] * matHs[:, 0, j]
		v_ij[(i, j)][:, 2] = matHs[:, 1, i] * matHs[:, 1, j]
		v_ij[(i, j)][:, 3] = matHs[:, 2, i] * matHs[:, 0, j] + matHs[:, 0, i] * matHs[:, 2, j]
		v_ij[(i, j)][:, 4] = matHs[:, 2, i] * matHs[:, 1, j] + matHs[:, 1, i] * matHs[:, 2, j]
		v_ij[(i, j)][:, 5] = matHs[:, 2, i] * matHs[:, 2, j]
	
	matV = np.concatenate((v_ij[(0, 1)], v_ij[(0, 0)] - v_ij[(1, 1)]), axis = 0)

	# Use SVD decompostion and find b
	_, _, b = np.linalg.svd(matV)
	matB = np.array([[b[5][0], b[5][1], b[5][3]],
					 [b[5][1], b[5][2], b[5][4]],
					 [b[5][3], b[5][4], b[5][5]]])

	# Compute intrinsic matK
	py = (matB[0][1] * matB[0][2] - matB[0][0] * matB[1][2]) / (matB[0][0] * matB[1][1] - matB[0][1] * matB[0][1])
	_lambda = matB[2][2] - (matB[0][2] * matB[0][2] + py * (matB[0][1] * matB[0][2] - matB[0][0] * matB[1][2])) / matB[0,0]
	fx = np.sqrt(_lambda / matB[0][0])
	fy = np.sqrt(_lambda * matB[0][0] / (matB[0][0] * matB[1][1] - matB[0][1] * matB[0][1]))
	s = -matB[0][1] * fx * fx * fy / _lambda
	px = s * py / fy - matB[0][2] * fx * fx / _lambda

	matK = np.array([[fx,  s, px],
					 [0., fy, py],
					 [0., 0., 1.]])
	return matK

### Problem 3 - Find the extrinsic parameters matR and matT
def extrinsic(matH, matK):
	# Copmute the inverse matrix of matK
	inv = np.linalg.inv(matK)
	s = 1. / np.linalg.norm(np.dot(inv, matH[:, 0]))

	# Compute extrinsic matR and matT
	matR = np.zeros((3, 3))
	matT = np.zeros((3, 1))
	matR[:, 0] = s * np.matmul(inv, matH[:, 0])
	matR[:, 1] = s * np.matmul(inv, matH[:, 1])
	matR[:, 2] = np.cross(matR[:, 0], matR[:, 1])
	matT       = s * np.matmul(inv, matH[:, 2])
	matR = convt2rotation(matR)
	return matR, matT

### Problem 2 & 3 - Estimate the intrisics and extrinsics of cameras
def init_param(ptWorld, ptImage):
	""" YOUR CODE STARTS HERE """
	# Compute the homography matrix matH
	num = len(ptImage)
	matHs = []
	for k in range(num):
		ptSrc = np.transpose(ptWorld)
		ptDst = np.transpose(ptImage[k])

		matH, _ = cv2.findHomography(ptSrc, ptDst)
		matHs.append(matH)
	
	# Compute the intrinsic matrix matK
	# Compute the extrinsic matrices matR and matT
	matK = intrinsic(matHs)
	matRs = []
	matTs = []
	for k in range(num):
		matR, matT = extrinsic(matHs[k], matK)
		matRs.append(matR)
		matTs.append(matT)

	matK = np.array([matK[0][0], matK[0][1], matK[0][2], matK[1][1], matK[1][2]])
	""" YOUR CODE ENDS HERE """
	return matRs, matTs, matK


### Problem 4 - Write the error function for least_squares
def error_fun(param, ptWorld, ptImage):
	# Find matA and matD
	matK = param[0:5]
	matD = param[5:10]
	matA = np.array([matK[0], matK[1], matK[2], 0, matK[3], matK[4], 0, 0, 1]).reshape([3, 3])
	
	lst = []
	num = np.array(ptWorld).shape[1]
	dim = np.array(ptImage).shape[0]
	src = np.concatenate((np.transpose(ptWorld), np.ones((num, 1))), axis = 1)
	for k in range(dim):

		# Find matR and matT
		matR = param[10 + k * 6: 13 + k * 6]
		matT = param[13 + k * 6: 16 + k * 6]
		matR = vector2matrix(matR)

		# Compute the position after extrinsic transformation
		matE = np.array([matR[:, 0], matR[:, 1], matT])
		tmp = np.matmul(src, matE)

		# Turn back to cartesian coordinate
		tmp[:,0] = tmp[:,0] / tmp[:,2]
		tmp[:,1] = tmp[:,1] / tmp[:,2]
		lst.append(tmp[:,:2])
	lst = np.array(lst).reshape(num * 3, 2)

	""" YOUR CODE STARTS HERE """
	sqrx = np.square(lst[:,0]).reshape(num * 3, 1)
	sqry = np.square(lst[:,1]).reshape(num * 3, 1)

	# Compute for the radial distortion
	rad  = (sqrx + sqry).reshape(num * 3, 1)
	dis1 = lst * (1 + matD[0] * rad + matD[1] * rad**2 + matD[4] * rad**3)

	# Compute for tangential distortion
	tan  = (lst[:,0] * lst[:,1]).reshape(num * 3, 1)
	dis2 = np.array([2 * matD[2] * tan + matD[3] * (rad + 2 * sqrx), 
					 2 * matD[3] * tan + matD[2] * (rad + 2 * sqry)])
	dis2 = np.swapaxes(dis2, 0, 1)[:,:,-1]
	
	dis = dis1 + dis2
	""" YOUR CODE ENDS HERE """

	# Compute the position after intrinsic transformation
	dis = np.concatenate((dis, np.ones((num * 3, 1))), axis = 1)
	res = np.matmul(dis, np.transpose(matA))
	
	# Turn back to cartesian coordinate
	res[:,0] = res[:,0] / res[:,2]
	res[:,1] = res[:,1] / res[:,2]
	res = res[:,:2]

	# Compute the error
	dst = np.swapaxes(ptImage, 1, 2)
	dst = np.array(dst).reshape(num * 3, 2)
	error = np.sum(np.square(res - dst), axis = 1)
	return error

### Problem 4 - Visualize the points after distortion
def visualize_distorted(param, pts_model, pts_2d):
	# Find matA and matD
	matK = param[0:5]
	matD = param[5:10]
	matA = np.array([matK[0], matK[1], matK[2], 0, matK[3], matK[4], 0, 0, 1]).reshape([3, 3])
	
	num = np.array(pts_model).shape[1]
	dim = np.array(pts_2d).shape[0]
	src = np.concatenate((np.transpose(pts_model), np.ones((num, 1))), axis = 1)
	for k in range(dim):

		# Find matR and matT
		matR = param[10 + k * 6: 13 + k * 6]
		matT = param[13 + k * 6: 16 + k * 6]
		matR = vector2matrix(matR)

		# Compute the position after extrinsic transformation
		matE = np.array([matR[:, 0], matR[:, 1], matT])
		tmp = np.matmul(src, matE)

		# Turn back to cartesian coordinate
		tmp[:,0] = tmp[:,0] / tmp[:,2]
		tmp[:,1] = tmp[:,1] / tmp[:,2]
		lst = tmp[:,:2]

		""" YOUR CODE STARTS HERE """
		sqrx = np.square(lst[:,0]).reshape(num, 1)
		sqry = np.square(lst[:,1]).reshape(num, 1)

		# Compute for the radial distortion
		rad  = (sqrx + sqry).reshape(num, 1)
		dis1 = lst * (1 + matD[0] * rad + matD[1] * rad**2 + matD[4] * rad**3)

		# Compute for tangential distortion
		tan  = (lst[:,0] * lst[:,1]).reshape(num, 1)
		dis2 = np.array([2 * matD[2] * tan + matD[3] * (rad + 2 * sqrx), 
						 2 * matD[3] * tan + matD[2] * (rad + 2 * sqry)])
		dis2 = np.swapaxes(dis2, 0, 1)[:,:,-1]
		
		dis = dis1 + dis2
		""" YOUR CODE ENDS HERE """

		# Compute the position after intrinsic transformation
		dis = np.concatenate((dis, np.ones((num, 1))), axis = 1)
		res = np.matmul(dis, np.transpose(matA))
		
		# Turn back to cartesian coordinate
		res[:,0] = res[:,0] / res[:,2]
		res[:,1] = res[:,1] / res[:,2]
		res = res[:,:2]

		dst = np.swapaxes(pts_2d[k], 0, 1)
		img = cv2.imread("./zhang_data/CalibIm{}.tif".format(k + 1))
		for pts in range(num):
			cv2.circle(img, (np.int32(res[pts, 0]), np.int32(res[pts, 1])), 4, _COLOR_BLUE)
			cv2.circle(img, (np.int32(dst[pts, 0]), np.int32(dst[pts, 1])), 3, _COLOR_RED)
		cv2.imwrite("img{}.jpg".format(k + 1), img)