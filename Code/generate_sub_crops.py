from os import listdir
import numpy as np
import scipy.io as sio

input_folder = "data/processed/"
output_folder = "data/completed/"
mean_matrix_path = "per_pixel_mean.mat"

mean_matrix = sio.loadmat(mean_matrix_path)["matrix"]

files = listdir(input_folder)

for f in files:
    image = sio.loadmat(input_folder + f)
    name = image["name"][0]
    matrix = image["matrix"]
    adjusted = np.subtract(matrix, mean_matrix)

    width = 256
    height = 256
    half_width = width / 2
    half_height = height / 2

    # Generate sub crops
    upper_left = adjusted[0:224, 0:224]
    sio.savemat(output_folder + name + "_ul.mat", {'matrix':upper_left})
    upper_right = adjusted[width-224:width, 0:224]
    sio.savemat(output_folder + name + "_ur.mat", {'matrix':upper_right})
    lower_left = adjusted[0:224, height-224:height]
    sio.savemat(output_folder + name + "_ll.mat", {'matrix':lower_left})
    lower_right = adjusted[width-224:width, height-224:height]
    sio.savemat(output_folder + name + "_lr.mat", {'matrix':lower_right})
    centre = adjusted[half_width-112:half_width+112, half_height-112:half_height+112]
    sio.savemat(output_folder + name + "_c.mat", {'matrix':centre})

    flipped = np.fliplr(adjusted)
    upper_left = flipped[0:224, 0:224]
    sio.savemat(output_folder + name + "_f_ul.mat", {'matrix':upper_left})
    upper_right = flipped[width-224:width, 0:224]
    sio.savemat(output_folder + name + "_f_ur.mat", {'matrix':upper_right})
    lower_left = flipped[0:224, height-224:height]
    sio.savemat(output_folder + name + "_f_ll.mat", {'matrix':lower_left})
    lower_right = flipped[width-224:width, height-224:height]
    sio.savemat(output_folder + name + "_f_lr.mat", {'matrix':lower_right})
    centre = flipped[half_width-112:half_width+112, half_height-112:half_height+112]
    sio.savemat(output_folder + name + "_f_c.mat", {'matrix':centre})
