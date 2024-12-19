import os
import keras
from keras.models import Model
import numpy as np
import cv2

BATCH_SIZE = 128
VALIDATION_IMAGE_FOLDER = "data/validation_images/"
VALIDATION_GROUND_TRUTH_FILE = "data/ILSVRC2012_validation_ground_truth.txt"
WNID_FILE = "data/id_wnid.csv"
DATA_FOLDER = "data/images"


def load_wnids():
    wnids = []
    f = open(WNID_FILE, 'r')
    for line in f.readlines():
        split = line.split(',')
        wnid = split[1].rstrip().replace('\'', '')
        wnids.append(wnid)
    return wnids


def load_labels():
    wnid_label = {}
    directories = os.listdir(DATA_FOLDER)
    directories.sort()
    i = 0
    for dir in directories:
        wnid_label[dir] = i
        i += 1
    return wnid_label


def load_validation_data(wnids, labels):
    X_val = []
    y_val = []
    files = os.listdir(VALIDATION_IMAGE_FOLDER)
    for f in files:
        image = cv2.imread(VALIDATION_IMAGE_FOLDER + f)
        X_val.append(image)
    ground_truth_file = open(VALIDATION_GROUND_TRUTH_FILE, 'r')
    for line in ground_truth_file.readlines():
        id = int(line)-1
        wnid = wnids[id]
        label = labels[wnid]
        y_val.append(label)
    return X_val, np.asarray(y_val)


def create_validation_batches(inputs, targets, batch_size):
    n = len(inputs)
    if n % batch_size == 0:
        augmentation = 0
    else:
        augmentation = batch_size - (n % batch_size)
    targets = np.append(targets, np.full(augmentation, -1))
    for i in range(0, n + augmentation, batch_size):
        subcrops = np.zeros(shape=(batch_size * 10, 224, 224, 3))
        classification = targets[i:i + 128]
        for j in range(0, batch_size):
            # If necessary, pad batch with dummy data to get complete batch
            if i + j >= n:
                image = np.zeros(shape=(256, 256, 3))
            else:
                image = inputs[i + j]
            k = j * 10
            subcrops[k + 0, 0:224, 0:224, 0:3] = image[0:224, 0:224]
            subcrops[k + 1, 0:224, 0:224, 0:3] = image[256 - 224:256, 0:224]
            subcrops[k + 2, 0:224, 0:224, 0:3] = image[0:224, 256 - 224:256]
            subcrops[k + 3, 0:224, 0:224, 0:3] = image[256 - 224:256, 256 - 224:256]
            subcrops[k + 4, 0:224, 0:224, 0:3] = image[128 - 112:128 + 112, 128 - 112:128 + 112]
        yield subcrops, classification


# Predictions are grouped in groups of 10. Average prediction over group and compare with target value
def evaluate_validation_predictions(predictions, targets):
    acc = 0.0
    shape = predictions.shape
    j = 0
    for i in range(0, shape[0], 10):
        # Exclude dummy data
        if targets[j] != -1:
            average = np.average(predictions[i:i + 10, ...], axis=1)
            predicted_class = np.argmax(average)
            if predicted_class == targets[j]:
                acc += 1.0
        j += 1
    return acc / j


class CalculateValidationScore(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.validation_accs = []
        wnids = load_wnids()
        labels = load_labels()
        X_val, y_val = load_validation_data(wnids, labels)
        self.X_val = X_val
        self.y_val = y_val
        print("Loaded validation data...")

    def on_epoch_end(self, epoch, logs={}):
        val_acc = 0.0
        val_batches = 0
        for inputs, targets in create_validation_batches(self.X_val, self.y_val, BATCH_SIZE):
            predictions = self.model.predict(inputs, batch_size=BATCH_SIZE)
            val_acc += evaluate_validation_predictions(predictions, targets)
            val_batches += 1
        validation_acc = val_acc / val_batches

        if len(self.validation_accs) > 0:
            prev_val_acc = self.validation_accs[-1]
            if abs(validation_acc - prev_val_acc) < 0.0001:
                previous_val_acc = validation_acc
                learning_rate = self.model.optimizer.lr / 10
                self.model.optimizer.lr = learning_rate

        print("Validation accuracy: \t\t{:.4f} %".format(validation_acc))
        self.validation_accs.append(validation_acc)
