""" CS4277/CS5477 Lab 3-1: Eight Point Algorithm.\
See accompanying Jupyter notebook (eight_point.ipynb) for instructions.

Name: Tseng Yu-Ting
Email: E0503474@u.nus.edu
NUSNET ID: E0503474
"""

import numpy as np
import scipy.io as sio
import h5py
import cv2
import matplotlib.pyplot as plt

###Helper function
def compute_right_epipole(F):
	U, S, V = np.linalg.svd(F.T)
	e = V[-1]
	return e / e[2]

### Helper Function - Visualize epipolar lines in the image
def plot_epipolar_line(img1, img2, F, x1, x2, epipole=None, show_epipole=False):
	plt.figure()
	plt.imshow(img1)
	for i in range(x1.shape[1]):
		plt.plot(x1[0, i], x1[1, i], 'bo')
		m, n = img1.shape[:2]
		line1 = np.dot(F.T, x2[:, i])
		t = np.linspace(0, n, 100)
		lt1 = np.array([(line1[2] + line1[0] * tt) / (-line1[1]) for tt in t])
		ndx = (lt1 >= 0) & (lt1 < m)
	plt.plot(t[ndx], lt1[ndx], linewidt = 2)
	plt.figure()
	plt.imshow(img2)

	for i in range(x2.shape[1]):
		plt.plot(x2[0, i], x2[1, i], 'ro')
		if show_epipole:
			if epipole is None:
				epipole = compute_right_epipole(F)
			plt.plot(epipole[0] / epipole[2], epipole[1] / epipole[2], 'r*')

		m, n = img2.shape[:2]
		line2 = np.dot(F, x1[:, i])

		t = np.linspace(0, n, 100)
		lt2 = np.array([(line2[2] + line2[0] * tt) / (-line2[1]) for tt in t])

		ndx = (lt2 >= 0) & (lt2 < m)
		plt.plot(t[ndx], lt2[ndx], linewidth=2)
	plt.show()


### Problem 3 - Compute the essential matrix from point correspondences and intrinsic matrix
def compute_essential(data1, data2, matK):
	"""YOUR CODE STARTS HERE"""
	matE_correct, _ = cv2.cv2.findEssentialMat(data1[:2, :].T, data2[:2, :].T, cameraMatrix = matK)

	# Normalize the coordinate
	num = data1.shape[1]
	inv   = np.linalg.inv(matK)
	data1 = np.matmul(inv, data1)
	data2 = np.matmul(inv, data2)

	# Compute matA
	tmp_data1 = np.tile(data1, (3, 1)).reshape(-1, 1)
	tmp_data2 = np.tile(data2, (1, 3)).reshape(-1, 1)
	matA = np.transpose(np.multiply(tmp_data1, tmp_data2)).reshape(9, -1)
	matA = np.transpose(matA)

	# Use SVD decompostion and find v
	_, _, v = np.linalg.svd(matA)
	matE = np.array([[v[8][0], v[8][1], v[8][2]],
					 [v[8][3], v[8][4], v[8][5]],
					 [v[8][6], v[8][7], v[8][8]]])

	# Enforce singularity constraints
	u, s, v = np.linalg.svd(matE)
	matE = np.matmul(u, np.matmul(np.diag(((s[0] + s[1]) / 2, (s[0] + s[1]) / 2, 0)), v))
	"""YOUR CODE ENDS HERE"""
	return matE

### Problem 4 - Compute the essential matrix from point correspondences and intrinsic matrix
def decompose_e(matE, matK, data1, data2):
	"""YOUR CODE STARTS HERE"""
	_, r, t, _ = cv2.recoverPose(matE, data1[:2, :].T, data2[:2, :].T, matK)
	trans_correct = np.concatenate([r, t], axis = 1)

	# Use SVD decompostion and find u, s, v, respectively
	u, s, v = np.linalg.svd(matE)
	w = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])

	# linear triangular method
	r1, t1 = np.matmul(u, np.matmul(w.T, v)),  u[:, 2].reshape(3, 1)
	r2, t2 = np.matmul(u, np.matmul(w  , v)),  u[:, 2].reshape(3, 1)
	r3, t3 = np.matmul(u, np.matmul(w.T, v)), -u[:, 2].reshape(3, 1)
	r4, t4 = np.matmul(u, np.matmul(w  , v)), -u[:, 2].reshape(3, 1)

	p  = np.matmul(matK, np.concatenate((np.eye(3), np.zeros((3, 1))), axis =1))
	p1 = np.matmul(matK, np.concatenate([r1, t1], axis = 1))
	p2 = np.matmul(matK, np.concatenate([r2, t2], axis = 1))
	p3 = np.matmul(matK, np.concatenate([r3, t3], axis = 1))
	p4 = np.matmul(matK, np.concatenate([r4, t4], axis = 1))

	x1, y1 = data1[0, 0], data1[1, 0]
	x2, y2 = data2[0, 0], data2[1, 0]
	matA1 = np.array([x1 * (p[2, :] - p[0, :]), y1 * (p[2, :] - p[1, :]), x2 * (p1[2, :] - p1[0, :]), y2 * (p1[2, :] - p1[1, :])])
	matA2 = np.array([x1 * (p[2, :] - p[0, :]), y1 * (p[2, :] - p[1, :]), x2 * (p2[2, :] - p2[0, :]), y2 * (p2[2, :] - p2[1, :])])
	matA3 = np.array([x1 * (p[2, :] - p[0, :]), y1 * (p[2, :] - p[1, :]), x2 * (p3[2, :] - p3[0, :]), y2 * (p3[2, :] - p3[1, :])])
	matA4 = np.array([x1 * (p[2, :] - p[0, :]), y1 * (p[2, :] - p[1, :]), x2 * (p4[2, :] - p4[0, :]), y2 * (p4[2, :] - p4[1, :])])

	_, _, v1 = np.linalg.svd(matA1)
	_, _, v2 = np.linalg.svd(matA2)
	_, _, v3 = np.linalg.svd(matA3)
	_, _, v4 = np.linalg.svd(matA4)

	possible1 = v1[3, :] / v1[3, 3]
	possible2 = v2[3, :] / v2[3, 3]
	possible3 = v3[3, :] / v3[3, 3]
	possible4 = v4[3, :] / v4[3, 3]

	flag1 = np.matmul(p, possible1)[2] > 0 and np.matmul(p1, possible1)[2] > 0
	flag2 = np.matmul(p, possible2)[2] > 0 and np.matmul(p2, possible2)[2] > 0
	flag3 = np.matmul(p, possible3)[2] > 0 and np.matmul(p3, possible3)[2] > 0
	flag4 = np.matmul(p, possible4)[2] > 0 and np.matmul(p4, possible4)[2] > 0

	# concate the two matrix
	trans = None
	if flag1 == True: trans = np.concatenate([r1, t1], axis = 1)
	if flag2 == True: trans = np.concatenate([r2, t2], axis = 1)
	if flag3 == True: trans = np.concatenate([r3, t3], axis = 1)
	if flag4 == True: trans = np.concatenate([r4, t4], axis = 1)
	"""YOUR CODE ENDS HERE"""
	return trans


### Problem 2 - Compute the fundamental matrix from point correspondences
def compute_fundamental(data1, data2):
	"""YOUR CODE STARTS HERE"""
	matF_correct, _ = cv2.findFundamentalMat(data1[:2, :].T, data2[:2, :].T, method = cv2.FM_8POINT)
	
	# Normalize the data
	num = data1.shape[1]
	cx_data1 = np.mean(data1[0, :])
	cy_data1 = np.mean(data1[1, :])
	cx_data2 = np.mean(data2[0, :])
	cy_data2 = np.mean(data2[1, :])

	dx_data1 = np.square(data1[0, :] - cx_data1)
	dy_data1 = np.square(data1[1, :] - cy_data1)
	dx_data2 = np.square(data2[0, :] - cx_data2)
	dy_data2 = np.square(data2[1, :] - cy_data2)
	s_data1  = np.sqrt(2) / np.mean(np.sqrt(dx_data1 + dy_data1))
	s_data2  = np.sqrt(2) / np.mean(np.sqrt(dx_data2 + dy_data2))
	matT1 = np.array([[s_data1, 0, -s_data1 * cx_data1],
					  [0, s_data1, -s_data1 * cy_data1],
					  [0,       0, 1]])
	matT2 = np.array([[s_data2, 0, -s_data2 * cx_data2],
					  [0, s_data2, -s_data2 * cy_data2],
					  [0,       0, 1]])
	data1 = np.matmul(matT1, data1)
	data2 = np.matmul(matT2, data2)

	# Compute matA
	tmp_data1 = np.tile(data1, (3, 1)).reshape(-1, 1)
	tmp_data2 = np.tile(data2, (1, 3)).reshape(-1, 1)
	matA = np.transpose(np.multiply(tmp_data1, tmp_data2)).reshape(9, -1)
	matA = np.transpose(matA)

	# Use SVD decompostion and find v
	_, _, v = np.linalg.svd(matA)
	matF = np.array([[v[8][0], v[8][1], v[8][2]],
					 [v[8][3], v[8][4], v[8][5]],
					 [v[8][6], v[8][7], v[8][8]]])

	# Enforce singularity constraints
	u, s, v = np.linalg.svd(matF)
	matF = np.matmul(u, np.matmul(np.diag((s[0], s[1], 0)), v))
	matF = np.matmul(np.transpose(matT2), matF)
	matF = np.matmul(matF, matT1)
	matF = matF / matF[2, 2]
	"""YOUR CODE ENDS HERE"""
	return matF