""" CS4277/CS5477 Lab 1: Fun with Homographies.
See accompanying Jupyter notebook (lab1.ipynb) for instructions.

Name: Tseng Yu-Ting
Email: E0503474@u.nus.edu
Student ID: A0212195L

"""
from math import floor, ceil, sqrt

import cv2
import math
import numpy as np
import matplotlib.pyplot as plt

_COLOR_RED = (255, 0, 0)
_COLOR_GREEN = (0, 255, 0)
_COLOR_BLUE = (0, 0, 255)

### Helper Function 1 - Loads image and converts to RGB format
def load_image(img_path):
	# Read the image and turn to RGB mode
	img = cv2.imread(img_path)
	img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
	return img

### Helper Function 2 - Generates a image line correspondences
def draw_matches(img1, img2, ptImg1, ptImg2, inlier_mask=None):
	# Compute required canvas size
	H1, W1 = img1.shape[0:2]
	H2, W2 = img2.shape[0:2]
	canvas_H = max(H1, H2)
	canvas_W = W1 + W2
	canvas = np.zeros((canvas_H, canvas_W, 3), "uint8")
	canvas[:H1, :W1, :] = img1
	canvas[:H2, W1:, :] = img2

	# Move ptImg1 and ptImg2 to correct position respectively
	ptTmp1 = ptImg1.copy()
	ptTmp2 = ptImg2.copy()
	ptTmp2[:, 0] += W1

	if inlier_mask is None:
		inlier_mask = np.ones(ptImg1.shape[0], dtype=np.bool)

	# Converts all to integer for plotting
	ptTmp1 = ptTmp1.astype(np.int32)
	ptTmp2 = ptTmp2.astype(np.int32)

	# Draw points
	ptAll = np.concatenate([ptTmp1, ptTmp2], axis=0)
	for pt in ptAll:
		cv2.circle(canvas, (pt[0], pt[1]), 4, _COLOR_BLUE, 2)

	# Draw lines
	num = len(ptAll) // 2
	for i in range(num):
		pt1 = tuple(ptTmp1[i, :])
		pt2 = tuple(ptTmp2[i, :])
		color = _COLOR_GREEN if inlier_mask[i] else _COLOR_RED
		cv2.line(canvas, pt1, pt2, color, 2)

	return canvas

### Helper Function 3 - Converts OpenCV's DMatch to point pairs
def matches2pairs(matches, keypoint1, keypoint2):
	# Create two list with keypoints' position values
	pt1, pt2 = [], []
	for m in matches:
		pt1.append(keypoint1[m.queryIdx].pt)
		pt2.append(keypoint2[m.trainIdx].pt)

	# Change the type to numpy
	pt1 = np.stack(pt1, axis=0)
	pt2 = np.stack(pt2, axis=0)
	return pt1, pt2

### Problem 1(a) - Transformation using provided homography matrix
def transform_homography(src, matH):
	""" YOUR CODE STARTS HERE """
	# Turn to homogeneous coordinate
	num = src.shape[0]
	src = np.array(src)
	src = np.concatenate((src, np.ones((num, 1))), axis = 1)

	# Compute the position after transformation
	dst = np.matmul(src, np.transpose(matH))

	# Turn back to cartesian coordinate
	dst[:,0] = dst[:,0] / dst[:,2]
	dst[:,1] = dst[:,1] / dst[:,2]
	""" YOUR CODE ENDS HERE """
	return dst[:,:2]

### Problem 1(b) - Compute Homography Matrix
def compute_homography(src, dst):
	""" YOUR CODE STARTS HERE """
	# Compute the needed matrix A
	num = src.shape[0]
	matA = []
	for k in range(num):
		ptSrc = np.append(src[k], 1)	# src point
		ptDst = np.append(dst[k], 1)	# dst point

		Ax = [-ptDst[2] * ptSrc[0], -ptDst[2] * ptSrc[1], -ptDst[2] * ptSrc[2], 0, 0, 0, \
			   ptDst[0] * ptSrc[0],  ptDst[0] * ptSrc[1],  ptDst[0] * ptSrc[2]]
		Ay = [0, 0, 0, -ptDst[2] * ptSrc[0], -ptDst[2] * ptSrc[1], -ptDst[2] * ptSrc[2], \
						ptDst[1] * ptSrc[0],  ptDst[1] * ptSrc[1],  ptDst[1] * ptSrc[2]]
		matA.append(Ax)
		matA.append(Ay)

	# Use SVD decompostion and find v
	matA = np.array(matA)
	_, _, v = np.linalg.svd(matA)
	matH = np.reshape(v[8], (3, 3))
	matH = (1 / matH[2][2]) * matH
	""" YOUR CODE ENDS HERE """
	return matH

### Problem 2 - Image Warping using Homography
def interpolation(src, mapx, mapy):
	# Find the distance of the position value to the floor and ceiling value
	tmpx = round(mapx - int(mapx), 2)
	tmpy = round(mapy - int(mapy), 2)

	dis  = [(tmpx, tmpy), (tmpx, 1-tmpy), (1-tmpx, tmpy), (1-tmpx, 1-tmpy)]
	pos  = [(mapy, mapx), (mapy, 1+mapx), (1+mapy, mapx), (1+mapy, 1+mapx)]

	# Compute the final possible pixel value
	value = 0
	for k in range(4):
		value += dis[3-k][0] * dis[3-k][1] * src[int(pos[k][0]), int(pos[k][1])]
	return value

def warp_image(src, dst, matH):
	# Copy to avoid overwriting the original image
	tmp = dst.copy()

	""" YOUR CODE STARTS HERE """
	# List all the position
	H, W = dst.shape[0:2]
	ori = np.meshgrid(np.arange(W), np.arange(H))
	ori = np.array(ori).reshape(2, W*H)
	ori = np.transpose(ori)

	# Compute the position after transformation
	new = transform_homography(ori, np.linalg.inv(matH))
	new = np.transpose(new)
	new = new.reshape(2, H, W)

	# Rewrite to the original image
	for i in range(H):
		for j in range(W):
			mapx = new[0][i][j]
			mapy = new[1][i][j]
			if mapx < 1 or mapx >= src.shape[1] - 1: continue
			if mapy < 1 or mapy >= src.shape[0] - 1: continue
			tmp[i][j] = interpolation(src, mapx, mapy)
	""" YOUR CODE ENDS HERE """
	return tmp

def warp_images_all(imgs, matHs):
	assert len(imgs) == len(matHs) and len(imgs) > 0
	num = len(imgs)

	# Compute the corner position after transformation
	corner = []
	for i in range(num):
		h, w = imgs[i].shape[:2]
		oriCorner = np.array([[0., 0.], [w, 0.], [w, h], [0., h]])
		newCorner = transform_homography(oriCorner, matHs[i])
		corner.append(newCorner)
	corner = np.concatenate(corner, axis = 0)

	# Compute required canvas size
	min_x, min_y = np.min(corner, axis = 0)
	max_x, max_y = np.max(corner, axis = 0)
	min_x, min_y = floor(min_x), floor(min_y)
	max_x, max_y = ceil (max_x), ceil (max_y)
	canvas = np.zeros((max_y-min_y, max_x-min_x, 3), imgs[0].dtype)

	# Adjust matrix H and warp the image
	for i in range(num):
		tmp = np.array([[1.0, 0.0, -min_x],
						[0.0, 1.0, -min_y],
						[0.0, 0.0, 1.0]], matHs[i].dtype)
		newH = tmp @ matHs[i]
		canvas = warp_image(imgs[i], canvas, newH)
	return canvas

### Problem 3 - Robust Homography Estimation using RANSAC
def compute_homography_error(src, dst, matH):
	""" YOUR CODE STARTS HERE """
	# Compute the bidirectional errors and sum up
	err = np.zeros(src.shape[0], np.float64)
	fir = transform_homography(src, matH)
	sec = transform_homography(dst, np.linalg.inv(matH))
	fir = np.sum((fir - dst) ** 2, axis = 1)
	sec = np.sum((sec - src) ** 2, axis = 1)
	err = fir + sec
	""" YOUR CODE ENDS HERE """
	return err

def compute_homography_ransac(src, dst, thresh=16.0, num_tries=200):
	""" YOUR CODE STARTS HERE """
	# Initialize the value
	num = src.shape[0]
	best_lines = -1
	best_model = np.eye(3, dtype = np.float64)
	best_masks = np.ones(num, dtype = np.bool)

	# Find the best solution through iteration
	for k in range(num_tries):
		index = np.random.randint(0, num, size = 4)
		ptSrc = np.array([src[index]]).reshape(4, 2)
		ptDst = np.array([dst[index]]).reshape(4, 2)

		model = compute_homography(ptSrc, ptDst)
		error = compute_homography_error(src, dst, model)
		lines = error[np.where(error < thresh)].shape[0]

		if lines >= best_lines:
			best_lines = lines
			best_masks = error < thresh
			best_model = compute_homography(src[np.where(error < thresh)], dst[np.where(error < thresh)])
	""" YOUR CODE ENDS HERE """
	return best_model, best_masks

### Problem 4 - Adding more images
def concatenate_homographies(ori_matH, ref):
	num = len(ori_matH) + 1
	assert ref < num

	""" YOUR CODE STARTS HERE """
	# Compute the matrix H of those before "ref"
	new_matH = []
	new_model = np.eye(3, dtype = np.float64)
	for k in range(ref - 1, -1, -1):
		new_model = np.matmul(new_model, ori_matH[k])
		new_matH.append(new_model)
	new_matH.reverse()

	# Compute the matrix H of those after "ref"
	new_model = np.eye(3, dtype = np.float64)
	new_matH.append(new_model)
	for k in range(ref, num - 1,  1):
		new_model = np.matmul(new_model, np.linalg.inv(ori_matH[k]))
		new_matH.append(new_model)
	""" YOUR CODE ENDS HERE """
	return new_matH