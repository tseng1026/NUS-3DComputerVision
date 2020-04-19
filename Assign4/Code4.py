""" CS4277/CS5477 Lab 4: Plane Sweep Stereo
See accompanying Jupyter notebook (lab4.ipynb) for instructions.

Name: Tseng Yu-Ting
Email: E0503474@u.nus.edu
NUSNET ID: E0503474
"""
import json
import os

import cv2
import numpy as np
from scipy.spatial.transform import Rotation
import scipy.ndimage

### Helper - Image Class (image, pose_mat)
class Image(object):
	def __init__(self, qvec, tvec, name, root_folder=''):
		self.qvec = qvec
		self.tvec = tvec
		self.name = name  # image filename
		self._image = self.load_image(os.path.join(root_folder, name))

		# Extrinsic matrix: Transforms from world to camera frame
		self.pose_mat = self.make_extrinsic(qvec, tvec)

	def __repr__(self):
		return '{}: qvec={}\n tvec={}'.format(
			self.name, self.qvec, self.tvec
		)

	@property
	def image(self):
		return self._image.copy()

	@staticmethod
	### Helper - Loads image and converts it to float64
	def load_image(path):
		im = cv2.imread(path)
		im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
		return im.astype(np.float64) / 255.0

	@staticmethod
	### Helper - Make 3x4 camera extrinsic matrix from colmap pose
	def make_extrinsic(qvec, tvec):
		rotation = Rotation.from_quat(np.roll(qvec, -1))
		return np.concatenate([rotation.as_dcm(), tvec[:, None]], axis=1)

### Helper - Write metadata to json file
def write_json(outfile, images, intrinsic_matrix, img_hw):
	img_height, img_width = img_hw

	images_meta = []
	for im in images:
		images_meta.append({
			'name': im.name,
			'qvec': im.qvec.tolist(),
			'tvec': im.tvec.tolist(),
		})

	data = {
		'img_height': img_height,
		'img_width': img_width,
		'K': intrinsic_matrix.tolist(),
		'images': images_meta
	}
	with open(outfile, 'w') as fid:
		json.dump(data, fid, indent=2)

### Helper - Loads dataset
def load_data(root_folder):
	print('Loading data from {}...'.format(root_folder))
	with open(os.path.join(root_folder, 'metadata.json')) as fid:
		metadata = json.load(fid)

	images = []
	for im in metadata['images']:
		images.append(Image(np.array(im['qvec']), np.array(im['tvec']),
							im['name'], root_folder=root_folder))
	img_hw = (metadata['img_height'], metadata['img_width'])
	K = np.array(metadata['K'])

	print('Loaded data containing {} images.'.format(len(images)))
	return images, K, img_hw

### Helper - Converts color representation into hexadecimal representation for K3D
def rgb2hex(rgb):
	rgb_uint = (rgb * 255).astype(np.uint8)
	hex = np.sum(rgb_uint * np.array([[256 ** 2, 256, 1]]),
				 axis=1).astype(np.uint32)
	return hex


### Problem 1 - Compute relative pose between two cameras
def compute_relative_pose(cam_pose, ref_pose):
	""" YOUR CODE STARTS HERE """
	camR = cam_pose[:, :3].reshape(3, 3)
	camT = cam_pose[:,  3].reshape(3, 1)
	refR = ref_pose[:, :3].reshape(3, 3)
	refT = ref_pose[:,  3].reshape(3, 1)

	matR = camR @ refR.T
	matT = camR @ refR.T @ refT * -1 + camT
	relative_pose = np.concatenate((matR, matT), axis = 1)
	""" YOUR CODE ENDS HERE """
	return relative_pose

### Problem 1 - Compute plane sweep homographies, assuming fronto parallel planes
def get_plane_sweep_homographies(matK, relative_pose, inv_depths):
	""" YOUR CODE STARTS HERE """
	matR = relative_pose[:, :3].reshape(3, 3)
	matT = relative_pose[:,  3].reshape(3, 1)
	matN = np.array([0, 0, 1]).reshape(1, 3)

	homography = []
	for d in range(len(inv_depths)):
		homography.append(matK @ (matR + matT @ matN * inv_depths[d]) @ np.linalg.inv(matK))
	homography = np.array(homography)
	""" YOUR CODE ENDS HERE """

	return homography

### Problem 2 - Compute plane sweep volume, by warping all images to the reference camera fronto-parallel planes
def compute_plane_sweep_volume(images, ref_pose, matK, inv_depths, img_size):
	H, W = img_size
	num = len(images)
	dth = len(inv_depths)
	extras = None

	""" YOUR CODE STARTS HERE """
	white = np.ones((H, W))
	sqr = np.zeros((dth, H, W, 3), dtype = np.float64)
	tot = np.zeros((dth, H, W, 3), dtype = np.float64)
	ps_volume   = np.zeros((dth, H, W), dtype = np.float64)
	accum_count = np.zeros((dth, H, W), dtype = np.float64)

	for k in range(num):
		pose = compute_relative_pose(ref_pose, images[k].pose_mat)
		matH = get_plane_sweep_homographies(matK, pose, inv_depths)

		for d in range(dth):
			warp = cv2.warpPerspective(images[k].image, matH[d], (W, H), cv2.INTER_LINEAR, cv2.BORDER_CONSTANT)
			mask = cv2.warpPerspective(white          , matH[d], (W, H), cv2.INTER_LINEAR, cv2.BORDER_CONSTANT)
			sqr[d] += warp * warp
			tot[d] += warp
			accum_count[d] += mask

	for k in range(3):
		tot[:,:,:,k] = tot[:,:,:,k] / accum_count
		sqr[:,:,:,k] = sqr[:,:,:,k] / accum_count
	ps_volume = np.mean(sqr - tot ** 2, axis = 3)

	""" YOUR CODE ENDS HERE """
	return ps_volume, accum_count.astype(int), extras

### Problem 2 - Computes inverse depth map from plane sweep volume as the argmin over plane sweep volume variances
def compute_depths(ps_volume, inv_depths):
	""" YOUR CODE STARTS HERE """
	H = ps_volume.shape[1]
	W = ps_volume.shape[2]

	inv_depth_image = np.zeros((H, W), dtype = np.float64)
	for i in range(H):
		for j in range(W):
			argmin = np.argmin(ps_volume[:,i,j])
			inv_depth_image[i][j] = inv_depths[argmin]
	""" YOUR CODE ENDS HERE """
	return inv_depth_image


### Problem 3 - Converts the depth map into points by unprojecting depth map into 3D
def unproject_depth_map(image, inv_depth_image, matK, mask = None):
	"""YOUR CODE STARTS HERE """
	H = image.shape[0]
	W = image.shape[1]

	xy = np.meshgrid (np.arange(W), np.arange(H))
	xy = np.transpose(np.array(xy), (1, 2, 0))
	xyz = np.concatenate((xy, np.ones((H, W, 1))), axis = 2)
	
	xyz = xyz @ np.linalg.inv(matK).T
	xyz = xyz / np.expand_dims(inv_depth_image, axis = 2)
	xyz = xyz.reshape(-1, 3)

	if type(mask) != type(None): image = image * np.expand_dims(mask, axis = 2)
	rgb = image.reshape(-1, 3)
	""" YOUR CODE ENDS HERE """
	return xyz, rgb


### Problem 4 - Post processes the plane sweep volume and compute a mask to indicate which pixels have confident estimates of the depth
def post_process(ps_volume, inv_depths, accum_count, extras):
	""" YOUR CODE STARTS HERE """
	dth = ps_volume.shape[0]
	H = ps_volume.shape[1]
	W = ps_volume.shape[2]

	new_ps_volume = np.zeros((dth, H, W), dtype = np.float64)
	for d in range(dth):
		maximum = np.max(ps_volume[d])
		minimum = np.min(ps_volume[d])
		original = (ps_volume[d] - minimum) / (maximum - minimum)

		gaussian = cv2.GaussianBlur(original, (5, 5), 0)
		new_ps_volume[d] = gaussian * (maximum - minimum) + minimum

	inv_depth_filtered = compute_depths(new_ps_volume, inv_depths)

	depth = np.zeros((H, W), dtype = np.uint32)
	for d in range(dth):
		depth = np.where(inv_depth_filtered == inv_depths[d], d, depth)
	mask = np.where(depth >= 64, True, False)

	inv_depth_filtered = cv2.GaussianBlur(inv_depth_filtered, (3, 3), 0)
	""" YOUR CODE ENDS HERE """
	return inv_depth_filtered, mask

