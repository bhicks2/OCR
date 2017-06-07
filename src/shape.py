import numpy as np
from matplotlib import pyplot as plt
from scipy.ndimage import imread
import canny
from PIL import Image

def calc_bin(val, minL, maxL, nbins):
	if val == minL: 
		return 0
	elif val == maxL: 
		return nbins - 1
	return int(val/(maxL/nbins))

def bin_mag_and_ang(p1, p2, rbins, abins, max_dist):
	magnitude = np.log(np.linalg.norm(p2-p1))
	angle = np.arctan2(p2[0]-p1[0], p2[1]-p1[1])
	if angle < 0: angle + 2*np.pi
	mag_bin = calc_bin(magnitude, 0, np.log(max_dist), rbins)
	ang_bin = calc_bin(angle, 0, 2*np.pi, abins)
	return mag_bin, ang_bin

def build_histograms(edge_pts, rbins, abins, max_dist):
	total_histograms = []
	for p1 in edge_pts:
		rest = np.array([p2 for p2 in edge_pts if np.linalg.norm(p2-p1) != 0])
		histogram = np.zeros((rbins, abins))
		for p2 in rest:
			mag_bin, ang_bin = bin_mag_and_ang(p1, p2, rbins, abins, max_dist)
			histogram[mag_bin, ang_bin] += 1
		total_histograms.append(np.ndarray.flatten(histogram))
	return np.array(total_histograms)

def plot_hist(hist_arr):
	hist_arr = np.reshape(hist_arr, (5, 12))
	plt.imshow(hist_arr, cmap='gray')
	plt.xticks([])
	plt.yticks([])
	plt.xlabel('angle', fontsize=30)
	plt.ylabel('log(r)', fontsize=30)
	plt.show()

def plot_fig(img):
	plt.imshow(img, cmap='gray')
	plt.xticks([])
	plt.yticks([])
	for spine in plt.gca().spines.values():
		spine.set_visible(False)
	plt.show()

def plot_edge(img1, edgepts):
	data = 255*np.ones(img1.shape)
	data[edgepts[:,0], edgepts[:,1]] = 0
	edge_img = Image.fromarray(data)
	plt.imshow(edge_img, cmap='gray')
	plt.xticks([])
	plt.yticks([])
	for spine in plt.gca().spines.values():
		spine.set_visible(False)
	plt.show()

# Main function
if __name__ == '__main__':
	filenames = ['a1.png']
	for file in filenames:
		img = imread(file, flatten=True)
		edges = canny.canny(img, sigma=0.8, low_thresh=0.01, high_thresh=0.1)
		edges_sample = edges[np.sort(np.random.choice(edges.shape[0], (200,), replace=False))]
		# plot_fig(img)
		# plot_edge(img, edges)
		# plot_edge(img, edges_sample)
		shape_hist = build_histograms(edges_sample, 5, 12, np.sqrt(img.shape[0]**2 + img.shape[1]**2))