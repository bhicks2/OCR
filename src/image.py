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
            self.image = image.astype(float)
        elif isinstance(image, Image):
            self.image = np.array(image.image, copy = True)
        elif isinstance(image, str):
            self.image = misc.imread(image)[:, :, RED_INDEX : BLUE_INDEX + 1].astype(float)

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
            self.image = new_image.astype(float)

    def crop(self, row, col, new_height, new_width, in_place = False):
        new_image = self.image[row : row + new_height, col : col + new_width]

        if not in_place:
            return Image(new_image)
        else:
            self.image = new_image.astype(float)

    def add_noise(self, intensity, monochromatic = True, in_place = False):
        if monochromatic:
            third_dimension = 1
        else:
            third_dimension = 3

        noise_layer = 2 * intensity * np.random.rand(self.height, self.width, third_dimension) - intensity
        new_image = self.image + noise_layer
        new_image = np.clip(new_image, 0, 255)

        if not in_place:
            return Image(new_image)
        else:
            self.image = new_image.astype(float)

    # certainty is a representation of the certainty we
    # have about the image's color. That is, if we believe
    # it is completely accurate, then we might have an
    # intensity of 100, which means that the pixel will
    # be determined to be itself. If it is less than 100
    # then we will split the remaining weight among the
    # surrounding pixels evenly.
    def denoise(self, certainty, in_place = False):
        center_weight = certainty / 100.0
        neighbor_weight = (1.0 - center_weight) / 8.0

        new_image = np.array(self.image, copy = True)

        for color in range(3):
            for i in range(1, self.height - 1):
                for j in range(1, self.width - 1):
                    center = self.image[i, j, color]

                    above = self.image[i - 1, j, color]
                    below = self.image[i + 1, j, color]

                    left = self.image[i, j - 1, color]
                    right = self.image[i, j + 1, color]

                    above_left = self.image[i - 1, j - 1, color]
                    above_right = self.image[i - 1, j + 1, color]

                    below_left = self.image[i + 1, j - 1, color]
                    below_right = self.image[i + 1, j + 1, color]

                    new_color = center_weight * center

                    new_color += neighbor_weight * above
                    new_color += neighbor_weight * below
                    new_color += neighbor_weight * left
                    new_color += neighbor_weight * right

                    new_color += neighbor_weight * above_left
                    new_color += neighbor_weight * above_right
                    new_color += neighbor_weight * below_left
                    new_color += neighbor_weight * below_right

                    new_image[i, j, color] = new_color

        if not in_place:
            return Image(new_image)
        else:
            self.image = new_image.astype(float)

    def normalize(self, in_place = False):
        maximum = np.max(np.max(self.image, axis = 1), axis = 0)
        maximum = np.reshape(maximum, [1, 1, 3])

        minimum = np.min(np.min(self.image, axis = 1), axis = 0)
        minimum = np.reshape(minimum, [1, 1, 3])

        new_image = 255 * ((self.image - minimum) / (maximum - minimum))

        if not in_place:
            return Image(new_image)
        else:
            self.image = new_image.astype(float)

    def binarize(self, threshold = None, in_place = False):
        if type(threshold) == int:
            threshold = np.array([threshold, threshold, threshold])

        if threshold is None:
            threshold = np.mean(self.image, axis = (0, 1))

        print threshold
        threshold = np.reshape(threshold, [1, 1, 3])

        new_image = 255 * (self.image > threshold)

        if not in_place:
            return Image(new_image)
        else:
            self.image = new_image.astype(float)


    def convert_to_grayscale(self, in_place = False):
        weight_matrix = np.array([RED_LUMA, GREEN_LUMA, BLUE_LUMA])
        weight_matrix = np.reshape(weight_matrix, [1, 1, weight_matrix.shape[0]])

        weighted_image = self.image * weight_matrix
        grayscale_image = np.sum(weighted_image, axis = 2)

        new_image = np.dstack((grayscale_image, grayscale_image, grayscale_image))

        if not in_place:
            return Image(new_image)
        else:
            self.image = new_image.astype(float)

    def darken(self, multiplier, in_place = False):
        if multiplier > 100 or multiplier < 0:
            raise IndexError("Multipler must fall in the closed interval [0, 100]")

        multiplier /= 100.0
        multiplier = 1 - multiplier

        new_image = multiplier * self.image

        if not in_place:
            return Image(new_image)
        else:
            self.image = new_image.astype(float)

    def brighten(self, multiplier, in_place = False):
        if multiplier > 100 or multiplier < 0:
            raise IndexError("Multipler must fall in the closed interval [0, 100]")

        multiplier = (100 - multiplier) / 100.0

        new_image = 255 - self.image
        new_image = multiplier * new_image
        new_image = 255 - new_image

        if not in_place:
            return Image(new_image)
        else:
            self.image = new_image

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

        # get gradients for the region
        angles, magnitudes = self.__compute_gradients__(region)

        # determine the histogram of gradients
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
        if row_min < 1 or row_max > self.height - 1:
            raise IndexError()
        if col_min < 1 or col_max > self.width - 1:
            raise IndexError()

        height = row_max - row_min
        width = col_max - col_min

        if height < 1:
            raise IndexError()
        if width < 1:
            raise IndexError()

        angles = np.zeros([height, width], dtype=float)
        magnitudes = np.zeros([height, width], dtype=float)

        for row in range(row_min, row_max):
            for col in range(col_min, col_max):
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


                angles[row - row_min, col - col_min] = angle
                magnitudes[row - row_min, col - col_min] = magnitude

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
