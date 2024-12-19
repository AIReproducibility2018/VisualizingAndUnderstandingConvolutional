from os import listdir
import random

random.seed(1)

input_folder = "D:\\ILSVRC2012_img_train\\"
output_folder = "batches/"

file_names = []

folders = listdir(input_folder)

i = 0
for folder in folders:
    print("Folder %i" % i)
    files = listdir(input_folder + folder)

    for f in files:
        name = f.split('.')[0]
        file_names.append((folder + "\\" + name, str(i)))
    i += 1

n = len(file_names)
print(n)

random.shuffle(file_names)

j = 0
k = 0
f = open(output_folder + "batch_" + str(k) + ".txt", 'w')
for i in range(n):
    f.write(file_names[i][0] + " " + file_names[i][1] + "\n")
    j += 1
    if j == 128:
        j = 0
        k += 1
        f.close()
        f = open(output_folder + "batch_" + str(k) + ".txt", 'w')

f.close()
