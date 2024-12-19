import os
import numpy as np
import cv2
import time
import sys


def process_batches(image_folder, batch_folder, output_folder, id, start_index, end_index):
    sum_matrix = np.zeros((256, 256, 3))

    for batch in range(start_index, end_index + 1):
        print("Batch %i" % batch)

        batch_output_folder = output_folder + "batch_" + str(batch) + "\\"
        if not os.path.isdir(batch_output_folder):
            os.mkdir(batch_output_folder)

        batch_file = batch_folder + "batch_" + str(batch) + ".txt"

        f = open(batch_file, 'r')
        lines = f.readlines()

        start = time.time()
        for line in lines:
            split = line.split(" ")
            image_path = split[0]
            image_name = image_path.split("\\")[1]

            # Read image
            image = cv2.imread(image_folder + image_path + ".JPEG")

            # Crop image
            shape = image.shape
            height = shape[0]
            width = shape[1]
            if width < 256 and height < 256:
                image = cv2.resize(image, (256, 256))
                height = 256
                width = 256
            elif height < width:
                image = cv2.resize(image, (width, 256))
                height = 256
            else:
                image = cv2.resize(image, (256, height))
                width = 256

            half_height = int(height / 2)
            half_width = int(width / 2)
            cropped_image = image[(half_height - 128):(half_height + 128), (half_width - 128):(half_width + 128)]
            sum_matrix = sum_matrix + cropped_image

            cv2.imwrite(batch_output_folder + image_name + ".JPEG", cropped_image)
        end = time.time()
        print(end - start)

    np.save("sum_matrix_" + str(id) + ".npy", sum_matrix)

if __name__ == "__main__":
    image_folder = "D:\\ILSVRC2012_img_train\\"
    batch_folder = "batches\\"
    output_folder = "D:\\batch_images\\"

    args = sys.argv
    id = int(args[1])
    start_index = int(args[2])
    end_index = int(args[3])
    process_batches(image_folder, batch_folder, output_folder, id, start_index, end_index)
