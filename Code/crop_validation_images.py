from os import listdir
import cv2

input_folder = "C:\\Users\\oddca\\Downloads\\ILSVRC2012_img_val\\"
output_folder = "validation_images\\"

files = listdir(input_folder)

for f in files:
    image = cv2.imread(input_folder + f)

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

    cv2.imwrite(output_folder + f, cropped_image)