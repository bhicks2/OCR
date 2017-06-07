import numpy as np
from scipy import signal as sg
from matplotlib import pyplot as plt
from scipy.ndimage import imread
from scipy.ndimage import filters
from PIL import Image

# Creates a Gaussian kernel for smoothing the image
def gaussian_kernel(k, sigma):
    x, y = np.mgrid[-k:k+1, -k:k+1]
    g = np.exp(-(x**2 + y**2)/(2.0 * sigma**2))
    return g / (2.0 * np.pi * sigma**2)

# Creates a Sobel filter to calculate gradients and angles
def sobel_filter(img):
	Hx = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
	Hy = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
	Gx = sg.convolve2d(img, Hx, mode='same', boundary='symm')
	Gy = sg.convolve2d(img, Hy, mode='same', boundary='symm')
	gradients = np.sqrt(np.square(Gx) + np.square(Gy))
	angles = np.arctan2(Gy, Gx) * 180/np.pi
	neg_angles = angles < 0
	angles[neg_angles] = angles[neg_angles] + 180
	return gradients, angles

# Non-maximum suppression to filter out edges (first pass)
def non_max_suppress(img, grad, angles):
    img_proc = np.zeros(img.shape)
    for i in range(1, img.shape[0]-1):
        for j in range(1, img.shape[1]-1):
            angle = angles[i][j]
            fill = grad[i,j]
            if angle < 22.5 or angle >= 157.5:
                if grad[i,j] >= grad[i,j-1] and grad[i,j] > grad[i,j+1]:
                    img_proc[i,j] = fill
            elif angle >= 22.5 and angle < 67.5:
                if grad[i,j] >= grad[i-1,j-1] and grad[i,j] > grad[i+1,j+1]:
                    img_proc[i,j] = fill
            elif angle >= 67.5 and angle < 112.5:
                if grad[i,j] >= grad[i-1,j] and grad[i,j] > grad[i+1,j]:
                    img_proc[i,j] = fill
            elif angle >= 112.5 and angle < 157.5:
                if grad[i,j] >= grad[i-1,j+1] and grad[i,j] > grad[i+1,j-1]:
                    img_proc[i,j] = fill
    return img_proc

# Find edges based on threshold (second pass)
def hysteresis(img, low, high):
	edges = []
	high_thresh = high * np.max(img)
	low_thresh = low * np.max(img)
	for i in range(img.shape[0]):
		for j in range(img.shape[1]):
			if img[i,j] > high_thresh:
				edges.append([i, j])
			elif img[i,j] > low_thresh and img[i,j] <= high_thresh:
				for p in range(i-1, i+2):
					for q in range(j-1, j+2):
						if img[p,q] > high_thresh:
							edges.append([i, j])
	return np.array(edges)

# Our canny edge detection method
def canny(image, sigma, low_thresh, high_thresh):
    (y, x) = image.shape
    image = image.astype(float)
    img = 255*np.divide(image - np.tile(np.min(image), (y,x)), np.tile(np.max(image), (y,x)) - np.tile(np.min(image), (y,x)))
    H = gaussian_kernel(2, 2*sigma)
    img_filtered = sg.convolve2d(img, H, mode='same', boundary='symm')
    gradients, angles = sobel_filter(img_filtered)
    max_supp = non_max_suppress(img_filtered, gradients, angles)
    edges = hysteresis(max_supp, low_thresh, high_thresh)
    return edges

# Helper method for plotting
def plot_fig(img1, edges):
	plt.subplot(121)
	plt.imshow(img1, cmap='gray')
	plt.title('Original Image')
	plt.xticks([])
	plt.yticks([])

	plt.subplot(122)
	data = np.zeros(img1.shape)
	data[edges[:,0], edges[:,1]] = 255
	edge_img = Image.fromarray(data)
	plt.imshow(edge_img, cmap='gray')
	plt.title('Edge Image')
	plt.xticks([])
	plt.yticks([])

	plt.show()

# Main function
if __name__ == '__main__':
	filename = '.png'
	img = imread(filename, flatten=True)
	edges = canny(img, sigma=0.8, low_thresh=0.01, high_thresh=0.25)
	# np.savetxt('gradients.txt', edges, fmt='%d', delimiter='\t')
	plot_fig(img, edges)
