import numpy as np
from scipy import misc
import matplotlib.pyplot as plt


RED_INDEX = 0
GREEN_INDEX = 1
BLUE_INDEX = 2

RED_LUMA = 0.299
GREEN_LUMA = 0.587
BLUE_LUMA = 0.114

class Image(object):

    def __init__(self, image):
        if isinstance(image, np.ndarray):
            self.image = image
        elif isinstance(image, Image):
            self.image = np.array(image.image, copy = True)
        elif isinstance(image, str):
            self.image = misc.imread(image)[:, :, RED_INDEX : BLUE_INDEX + 1]

    @property
    def height(self):
        return self.image.shape[0]

    @property
    def width(self):
        return self.image.shape[1]

    @property
    def size(self):
        return (self.height, self.width)

    def resize(self, new_width, new_height, in_place = False):
        new_image = misc.imresize(self.image, (height, width))
        if not in_place:
            return Image(new_image)
        else:
            self.image = new_image

    def crop(self, row, col, new_width, new_height, in_place = False):
        new_image = self.image[row : row + new_height, col : col + new_width]

        if not in_place:
            return Image(new_image)
        else:
            self.image = new_image

    def add_noise(self, intensity, monochromatic = True, in_place = False):
        if monochromatic:
            third_dimension = 1
        else:
            third_dimension = 3

        noise_layer = 2 * intensity * np.random.rand(self.height, self.width, third_dimension) - intensity
        print noise_layer
        new_image = self.image + noise_layer
        print new_image
        new_image = np.clip(new_image, 0, 255)

        if not in_place:
            return Image(new_image)
        else:
            self.image = new_image

    def convert_to_grayscale(self, in_place = False):
        weight_matrix = np.array([RED_LUMA, GREEN_LUMA, BLUE_LUMA])
        weight_matrix = np.reshape(weight_matrix, [1, 1, weight_matrix.shape[0]])

        weighted_image = self.image * weight_matrix
        grayscale_image = np.sum(weighted_image, axis = 2)

        new_image = np.dstack((grayscale_image, grayscale_image, grayscale_image))

        if not in_place:
            return Image(new_image)
        else:
            self.image = image

    # region is assumed to be a 4-tuple of
    # the form (row_min, col_min, row_max, col_max)
    #
    # Note: the region is the region of the hog,
    # not the region that is required for the gradient
    # however, the region cannot include the extremes
    # in any direction
    def calculate_hog(self, region, nbins):
        row_min, col_min, row_max, col_max = region

        if row_min < 1 or row_max > self.height - 1:
            raise IndexError()
        if col_min < 1 or col_max > self.width - 1:
            raise IndexError()


        # convert the actual region to the region of the image we need
        # in order to calculate the gradient.
        gradient_region = (row_min - 1, col_min - 1, row_max + 1, col_max + 1)
        # get gradients for the region
        angles, magnitudes = self.__compute_gradients__(gradient_region)

        # determine the histogram of gradients
        histogram = np.zeros([nbins])

        histogram = np.zeros([nbins], dtype = float)

        width = angles.shape[1]
        height = angles.shape[0]

        bin_width = float(180)/nbins
        midpoint = float(bin_width) / 2

        for i in range(height):
            for j in range(width):

                angle = angles[i, j]
                magnitude = magnitudes[i, j]

                if angle < midpoint:
                    bin_above = 0
                    bin_below = nbins - 1

                    angle_above = midpoint
                    angle_below = -midpoint

                    histogram[bin_below] += magnitude * np.abs(angle - angle_above) / (180 / nbins)
                    histogram[bin_above] += magnitude * np.abs(angle - angle_below) / (180 / nbins)
                    break

                if angle >= 180 - midpoint:
                    bin_above = 0
                    bin_below = nbins - 1

                    angle_above = 180 + midpoint
                    angle_below = 180 - midpoint

                    histogram[bin_below] += magnitude * np.abs(angle - angle_above) / (180 / nbins)
                    histogram[bin_above] += magnitude * np.abs(angle - angle_below) / (180 / nbins)
                    break

                for k in range(nbins - 1):
                    bin_below = k
                    bin_above = k + 1

                    angle_below = bin_below * bin_width + midpoint
                    angle_above = bin_above * bin_width + midpoint

                    if angle_below <= angle and angle_above >= angle:
                        histogram[bin_below] += magnitude * np.abs(angle - angle_above) / (180 / nbins)
                        histogram[bin_above] += magnitude * np.abs(angle - angle_below) / (180 / nbins)
                        break

        return histogram


    def save_image(self, filename):
        image = misc.toimage(self.image, cmin = 0, cmax = 255, mode = "RGB", channel_axis = 2)
        image.save(filename)

    def display_image(self):
        image = misc.toimage(self.image, cmin = 0, cmax = 255, mode = "RGB", channel_axis = 2)
        plt.imshow(image)
        plt.show()

    # region is assumed to be passed in as a
    # 4-tuple of the form (row_min, col_min, row_max, col_max)
    def __compute_gradients__(self, region):
        row_min, col_min, row_max, col_max = region
        if row_min < 0 or row_max > self.height:
            raise IndexError()
        if col_min < 0 or col_max > self.width:
            raise IndexError()

        height = row_max - row_min
        width = col_max - col_min

        if height < 3:
            raise IndexError()
        if width < 3:
            raise IndexError()

        angles = np.zeros([height - 2, width - 2], dtype=float)
        magnitudes = np.zeros([height - 2, width - 2], dtype=float)

        for row in range(row_min + 1, row_max - 1):
            for col in range(col_min + 1, col_max - 1):
                above = self.image[row - 1, col, 0]
                below = self.image[row + 1, col, 0]

                left = self.image[row, col - 1, 0]
                right = self.image[row, col + 1, 0]

                dx = right - left
                dy = below - above

                if dx == 0:
                    angle = 90
                else:
                    angle = np.arctan(dy / dx) * 180 / np.pi

                magnitude = np.sqrt(dx ** 2 + dy ** 2)

                angles[row - 1, col - 1] = angle
                magnitudes[row - 1, col - 1] = magnitude

        return angles, magnitudes

    def __setitem__(self, args, new_value):
        row = None
        col = None
        color = None

        if len(args) == 2:
            row, col = args
        else:
            row, col, color = args

        if row < 0 or row >= self.height:
            raise IndexError()
        if col < 0 or col >= self.width:
            raise IndexError()
        if color is not None and (color < 0 or color > BLUE_INDEX):
            raise IndexError()

        if color is None:
            self.image[row, col] = new_value
        else:
            self.image[row, col, color] = new_value

    def __getitem__(self, args):
        row = None
        col = None
        color = None

        if len(args) == 2:
            row, col = args
        else:
            row, col, color = args

        if row < 0 or row >= self.height:
            raise IndexError()
        if col < 0 or col >= self.width:
            raise IndexError()
        if color is not None and (color < 0 or color > BLUE_INDEX):
            raise IndexError()

        if color is None:
            return self.image[row, col, :]
        else:
            return self.image[row, col, color]

        if row < 0 or row >= self.height:
            raise IndexError()
        if col < 0 or y >= self.width:
            raise IndexError()
        if color is not None and (color < 0 or color > BLUE_INDEX):
            raise IndexError()

        if color is None:
            return self.image[row, col, :]
        else:
            return self.image[row, col, color]
