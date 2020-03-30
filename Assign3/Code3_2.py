""" CS4277/CS5477 Lab 3-2: Absolute Pose Estimation.\
See accompanying Jupyter notebook (pnp.ipynb) for instructions.

Name: Tseng Yu-Ting
Email: E0503474@u.nus.edu
NUSNET ID: E0503474
"""

import numpy as np
import cv2
import sympy as sym
import itertools
from sympy.polys import subresultants_qq_zz

### Helper Function - Extract coefficients of a polynomial
def extract_coeff(x1, x2, x3, cos_theta12, cos_theta23, cos_theta13, d12, d23, d13):
	f12 = x1 ** 2 + x2 ** 2 - 2 * x1 * x2 * cos_theta12 - d12
	f23 = x2 ** 2 + x3 ** 2 - 2 * x2 * x3 * cos_theta23 - d23
	f13 = x1 ** 2 + x3 ** 2 - 2 * x1 * x3 * cos_theta13 - d13
	matrix = subresultants_qq_zz.sylvester(f23, f13, x3)

	f12_ = matrix.det()
	f1 = subresultants_qq_zz.sylvester(f12, f12_, x2).det()
	a1 = f1.func(*[term for term in f1.args if not term.free_symbols])
	a2 = f1.coeff(x1 ** 2)
	a3 = f1.coeff(x1 ** 4)
	a4 = f1.coeff(x1 ** 6)
	a5 = f1.coeff(x1 ** 8)
	a = np.array([a1, a2, a3, a4, a5])
	return a

### Helper Function - Estimate the rotation and translation using icp algorithm
def icp(ptWorld, ptImage):
	avgWorld = np.mean(ptWorld, axis = 0, keepdims = True)
	avgImage = np.mean(ptImage, axis = 0, keepdims = True)
	ptWorld_center = ptWorld - avgWorld
	ptImage_center = ptImage - avgImage
	w = np.dot(ptWorld_center.T, ptImage_center)

	u, _, v = np.linalg.svd(w)
	if np.linalg.det(v.T.dot(u.T)) < 0: v[-1, :] *= -1
	if np.linalg.det(v.T.dot(u.T)) > 0: v[-1, :] *=  1

	r = v.T.dot(u.T)
	t = avgImage.T - np.dot(r, avgWorld.T)
	return r, t

### Helper Function - Reconstruct the 3d points from camera-point distance
def reconstruct_3d(matX, matK, ptImage):
	ptReconstruct = []
	for i in range(len(matX)):
		ptReconstruct.append(matX[i] * np.dot(np.linalg.inv(matK), ptImage[i].T))
	ptReconstruct = np.hstack(ptReconstruct)
	return ptReconstruct

def visualize(r, t, points3d, points2d, matK):
	scale = 0.2
	img = cv2.imread("Code3_2_data/img_id4_ud.JPG")
	dim = (int(img.shape[1] * scale), int(img.shape[0] * scale))
	img = cv2.resize(img, dim)

	num = points2d.shape[0]
	trans = np.hstack([r, t])
	points3d = np.hstack([points3d, np.ones((points3d.shape[0], 1))])
	points2d_old = points2d
	points2d_new = np.dot(matK, np.dot(trans, points3d.T))
	points2d_new = np.transpose(points2d_new[:2, :] / points2d_new[2:3, :])
	for k in range(num):
		cv2.circle(img, (int(points2d_old[k, 0] * scale), int(points2d_old[k, 1] * scale)), 3,  (0, 0, 255))
		cv2.circle(img, (int(points2d_new[k, 0] * scale), int(points2d_new[k, 1] * scale)), 4,  (255, 0, 0))
	cv2.imshow("img", img)
	cv2.waitKey(0)


### Problem 1 - Estimate the rotation and translation of camera by using pnp algorithm
def pnp_algo(matK, points2d, points3d):
	_, r_correct, t_correct = cv2.solvePnP(points3d, points2d, matK, np.zeros((5,)))
	r_correct, _ = cv2.Rodrigues(r_correct)
	# print (r_correct)
	# print (t_correct)

	num = points2d.shape[0]
	homo = np.concatenate((points2d, np.ones((num, 1, 1))), axis = 2).reshape(10, 3)
	permutation = list(itertools.permutations(np.arange(num), 2))
	combination = list(itertools.combinations(np.arange(num), 2))

	# Compute the basic information (distance and cosine) between each two points
	matC = np.linalg.inv(np.matmul(matK, matK.T))
	matU = homo @ matC @ homo.T

	cos = np.zeros((num, num))
	dis = np.zeros((num, num))
	for (i, j) in permutation:
		cos[i][j] = matU[i][j] / (np.sqrt(matU[i][i]) * np.sqrt(matU[j][j]))
		dis[i][j] = np.linalg.norm(points3d[i] - points3d[j]) ** 2

	# Compute for matrix r and vector t
	matX = []	
	for k in range(num):

		# Compute the coefficient value within three points
		matA = []
		for (i, j) in combination:
			if k == i: continue
			if k == j: continue

			x1, x2, x3  = sym.symbols("x1, x2, x3")
			coefficient = extract_coeff(x1, x2, x3, cos[k][i], cos[i][j], cos[j][k], dis[k][i], dis[i][j], dis[j][k])
			matA.append(coefficient)
		matA = np.array(matA, dtype = "float")

		# Use SVD decompostion and find v
		_, _, v = np.linalg.svd(matA)
		x = np.average([v[4][1] / v[4][0], v[4][2] / v[4][1], v[4][3] / v[4][2], v[4][4] / v[4][3]])
		matX.append(np.sqrt(x))

	ptImage = np.swapaxes(reconstruct_3d(matX, matK, homo.reshape(num, 1, 3)), 0, 1)
	ptWorld = np.squeeze (points3d)
	r, t = icp(ptWorld, ptImage)
	"""YOUR CODE ENDS HERE"""
	return r, t