from os import listdir
import numpy as np
import scipy.io as sio
from PIL import Image

input_data_path = "data/original/"
output_data_path = "data/processed/"

files = listdir(input_data_path)
images = []
sum_matrix = np.zeros((256, 256, 3), dtype=np.int8)

for f in files:
    name = f.split('.')[0]

    # Read image
    image = Image.open(input_data_path + f)

    # Crop image
    width, height = image.size
    if width < height:
        image.thumbnail((256, height), Image.ANTIALIAS)
    else:
        image.thumbnail((width, 256), Image.ANTIALIAS)

    width, height = image.size
    half_width = width / 2
    half_height = height / 2
    cropped_image = image.crop((half_width - 128, half_height - 128, half_width + 128, half_height + 128))
    width, height = cropped_image.size

    # Calculate per pixel mean
    rgb_image = cropped_image.convert('RGB')
    for i in range(width):
        for j in range(height):
            r, g, b = rgb_image.getpixel((i, j))
            sum_matrix[i][j][0] += r
            sum_matrix[i][j][1] += g
            sum_matrix[i][j][2] += b

    matrix = np.array(rgb_image)
    images.append((name, matrix))

mean_matrix = np.divide(sum_matrix, len(files))

for image in images:
    name = image[0]
    matrix = image[1]
    adjusted = np.subtract(matrix, mean_matrix)

    width = adjusted.shape[0]
    height = adjusted.shape[1]
    half_width = width / 2
    half_height = height / 2

    # Generate sub crops
    upper_left = adjusted[0:224,0:224]
    sio.savemat(output_data_path + name + "_ul.mat", {'matrix':upper_left})
    upper_right = adjusted[width-224:width,0:224]
    sio.savemat(output_data_path + name + "_ur.mat", {'matrix':upper_right})
    lower_left = adjusted[0:224,height-224:height]
    sio.savemat(output_data_path + name + "_ll.mat", {'matrix':lower_left})
    lower_right = adjusted[width-224:width,height-224:height]
    sio.savemat(output_data_path + name + "_lr.mat", {'matrix':lower_right})
    centre = adjusted[half_width-112:half_width+112,half_height-112:half_height+112]
    sio.savemat(output_data_path + name + "_c.mat", {'matrix':centre})

    flipped = np.fliplr(adjusted)
    upper_left = flipped[0:224,0:224]
    sio.savemat(output_data_path + name + "_f_ul.mat", {'matrix':upper_left})
    upper_right = flipped[width-224:width,0:224]
    sio.savemat(output_data_path + name + "_f_ur.mat", {'matrix':upper_right})
    lower_left = flipped[0:224,height-224:height]
    sio.savemat(output_data_path + name + "_f_ll.mat", {'matrix':lower_left})
    lower_right = flipped[width-224:width,height-224:height]
    sio.savemat(output_data_path + name + "_f_lr.mat", {'matrix':lower_right})
    centre = flipped[half_width-112:half_width+112,half_height-112:half_height+112]
    sio.savemat(output_data_path + name + "_f_c.mat", {'matrix':centre})
