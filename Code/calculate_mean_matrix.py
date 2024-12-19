from os import listdir
import numpy as np
import cv2

NR_IMAGES = 1300000

def test():
    test = np.arange(1000).reshape((100, 10))
    t = test[10:20,...]
    print(t)


def load_wnids():
    wnids = []
    f = open("Z:\Masteroppgave\VisUndConvNet\data\id_wnid.csv", 'r')
    for line in f.readlines():
        split = line.split(',')
        wnid = split[1].rstrip().replace('\'', '')
        wnids.append(wnid)
    return wnids

def load_labels():
    wnid_label = {}
    directories = listdir("Z:\Masteroppgave\VisUndConvNet\data\images")
    directories.sort()
    i = 0
    for dir in directories:
        wnid_label[dir] = i
        i += 1
    return wnid_label


def read_images():
    image_folder = "data/batch_images/"
    folders = listdir(image_folder)
    nr_images = 0
    sum_matrix = np.zeros((256, 256, 3), dtype='int32')
    for folder in folders:
        images = listdir(image_folder + folder)
        for f in images:
            image = cv2.imread(image_folder + folder + "\\" + f)
            sum_matrix = sum_matrix + image
            nr_images += 1
            if nr_images % 1000 == 0:
                print(nr_images)


print(load_wnids())
print(load_labels())