import os

os.environ["THEANO_FLAGS"] = "mode=FAST_RUN,device=cuda,floatX=float32,dnn.enabled=False"

from keras import optimizers
from keras import initializers
from keras.models import Model, load_model
from keras.layers import Input, Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint
import numpy as np
import cv2
import random
from training_callback import CalculateValidationScore

RANDOM_SEED = 1
EPOCHS = 70
BATCH_SIZE = 128
NR_CLASSES = 1000
NR_TRAINING_BATCHES = 10010

BATCH_FOLDER = "data/batches/"
BATCH_IMAGE_FOLDER = "data/batch_images/"
VALIDATION_IMAGE_FOLDER = "data/validation_images/"
VALIDATION_GROUND_TRUTH_FILE = "data/ILSVRC2012_validation_ground_truth.txt"
MODEL_FILE = ""
DATA_FOLDER = "data/images"

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


def get_model(input_shape, nr_classes):
    inputs = Input(batch_shape=input_shape)

    conv_1 = Conv2D(filters=96, kernel_size=(7, 7), strides=2, data_format='channels_last', activation='relu',
                    kernel_initializer=initializers.Constant(0.01), bias_initializer='zeros')(inputs)
    pool_1 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(conv_1)
    normalization_1 = BatchNormalization()(pool_1)

    conv_2 = Conv2D(filters=256, kernel_size=(5, 5), strides=2, activation='relu',
                    kernel_initializer=initializers.Constant(0.01), bias_initializer='zeros')(normalization_1)
    pool_2 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(conv_2)
    normalization_2 = BatchNormalization()(pool_2)

    conv_3 = Conv2D(filters=384, kernel_size=(3, 3), strides=1, activation='relu',
                    kernel_initializer=initializers.Constant(0.01), bias_initializer='zeros')(normalization_2)

    conv_4 = Conv2D(filters=384, kernel_size=(3, 3), strides=1, activation='relu',
                    kernel_initializer=initializers.Constant(0.01), bias_initializer='zeros')(conv_3)

    conv_5 = Conv2D(filters=256, kernel_size=(3, 3), strides=1, activation='relu',
                    kernel_initializer=initializers.Constant(0.01), bias_initializer='zeros')(conv_4)
    pool_5 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(conv_5)

    flat_6 = Flatten(name="flatten")(pool_5)
    dense_6 = Dense(4096, activation='relu',
                    kernel_initializer=initializers.Constant(0.01), bias_initializer='zeros')(flat_6)
    drop_6 = Dropout(0.5)(dense_6)

    dense_7 = Dense(4096, activation='relu',
                    kernel_initializer=initializers.Constant(0.01), bias_initializer='zeros')(drop_6)
    drop_7 = Dropout(0.5)(dense_7)

    dense_8 = Dense(nr_classes, kernel_initializer=initializers.Constant(0.01), bias_initializer='zeros')(drop_7)

    prediction = Activation(activation='softmax')(dense_8)

    return Model(input=inputs, output=prediction)


def load_mean_matrix():
    return np.load("mean_matrix.npy")


def load_training_data(nr_classes):
    X_train = []
    y_train = []
    for i in range(NR_TRAINING_BATCHES):
        f = open(BATCH_FOLDER + "batch_" + str(i) + ".txt", 'r')
        for l in f.readlines():
            line = l.split(" ")
            name = line[0].split("\\")[1]
            label = int(line[1])
            classification = np.zeros(nr_classes)
            classification[label] = 1
            image_file = "batch_" + str(i) + "/" + name + ".JPEG"
            X_train.append(image_file)
            y_train.append(classification)
    return np.asarray(X_train), np.asarray(y_train)


def load_validation_data():
    X_val = []
    y_val = []
    files = os.listdir(VALIDATION_IMAGE_FOLDER)
    for f in files:
        image = cv2.imread(VALIDATION_IMAGE_FOLDER + f)
        X_val.append(image)
    ground_truth_file = open(VALIDATION_GROUND_TRUTH_FILE, 'r')
    for line in ground_truth_file.readlines():
        # Zero index class
        y_val.append(int(line) - 1)
    return X_val, np.asarray(y_val)


def create_training_batches(inputs, targets, batch_size, mean_matrix):
    n = len(inputs)
    print("Number of training samples ", n)
    indices = np.arange(n)
    np.random.shuffle(indices)
    for start in range(0, n - batch_size + 1, batch_size):
        batch_indices = indices[start:start + batch_size]
        batch = inputs[batch_indices]

        batch_images = []
        for image_file in batch:
            image = cv2.imread(BATCH_IMAGE_FOLDER + image_file) - mean_matrix
            batch_images.append(image)

        yield np.asarray(generate_training_subcrops(batch_images)), targets[batch_indices]


def generate_training_subcrops(inputs):
    subcrops = []
    for image in inputs:
        r = random.uniform(0, 1)
        if r <= 0.2:
            crop = image[0:224, 0:224]
        elif r <= 0.4:
            crop = image[256 - 224:256, 0:224]
        elif r <= 0.6:
            crop = image[0:224, 256 - 224:256]
        elif r <= 0.8:
            crop = image[256 - 224:256, 256 - 224:256]
        else:
            crop = image[128 - 112:128 + 112, 128 - 112:128 + 112]
        r = random.uniform(0, 1)
        if r < 0.5:
            crop = np.fliplr(crop)
        subcrops.append(crop)
    return subcrops


def generate_subcrop(input, mean_matrix):
    image = input - mean_matrix
    r = random.uniform(0, 1)
    if r <= 0.2:
        crop = image[0:224, 0:224]
    elif r <= 0.4:
        crop = image[256 - 224:256, 0:224]
    elif r <= 0.6:
        crop = image[0:224, 256 - 224:256]
    elif r <= 0.8:
        crop = image[256 - 224:256, 256 - 224:256]
    else:
        crop = image[128 - 112:128 + 112, 128 - 112:128 + 112]
    r = random.uniform(0, 1)
    if r < 0.5:
        crop = np.fliplr(crop)
    return crop


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
            flipped = np.fliplr(image)
            subcrops[k + 5, 0:224, 0:224, 0:3] = flipped[0:224, 0:224]
            subcrops[k + 6, 0:224, 0:224, 0:3] = flipped[256 - 224:256, 0:224]
            subcrops[k + 7, 0:224, 0:224, 0:3] = flipped[0:224, 256 - 224:256]
            subcrops[k + 8, 0:224, 0:224, 0:3] = flipped[256 - 224:256, 256 - 224:256]
            subcrops[k + 9, 0:224, 0:224, 0:3] = flipped[128 - 112:128 + 112, 128 - 112:128 + 112]
        yield subcrops, classification


def generate_validation_subcrops(inputs, batch_size):
    n = len(inputs)
    if n % batch_size == 0:
        augmentation = 0
    else:
        augmentation = batch_size - (n % batch_size)
    subcrops = np.zeros(shape=(n + augmentation, 224, 224, 3))
    i = 0
    for image in inputs:
        subcrops[i + 0, 0:224, 0:224, 0:3] = image[0:224, 0:224]
        subcrops[i + 1, 0:224, 0:224, 0:3] = image[256 - 224:256, 0:224]
        subcrops[i + 2, 0:224, 0:224, 0:3] = image[0:224, 256 - 224:256]
        subcrops[i + 3, 0:224, 0:224, 0:3] = image[256 - 224:256, 256 - 224:256]
        subcrops[i + 4, 0:224, 0:224, 0:3] = image[128 - 112:128 + 112, 128 - 112:128 + 112]
        flipped = np.fliplr(image)
        subcrops[i + 5, 0:224, 0:224, 0:3] = flipped[0:224, 0:224]
        subcrops[i + 6, 0:224, 0:224, 0:3] = flipped[256 - 224:256, 0:224]
        subcrops[i + 7, 0:224, 0:224, 0:3] = flipped[0:224, 256 - 224:256]
        subcrops[i + 8, 0:224, 0:224, 0:3] = flipped[256 - 224:256, 256 - 224:256]
        subcrops[i + 9, 0:224, 0:224, 0:3] = flipped[128 - 112:128 + 112, 128 - 112:128 + 112]
        i += 1
    return subcrops


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


def crop_generator(batches, mean_matrix):
    while True:
        X, y = next(batches)
        batch_size = X.shape[0]
        crops = np.zeros((batch_size, 224, 224, 3))
        for i in range(batch_size):
            crops[i] = generate_subcrop(X[i], mean_matrix)
        yield (crops, y)


def train_model():
    learning_rate = 0.01
    if not MODEL_FILE:
        model = get_model((BATCH_SIZE, 224, 224, 3), NR_CLASSES)
        model.compile(optimizer=optimizers.SGD(lr=learning_rate, momentum=0.9),
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])
        print("Compiled model...")
    else:
        model = load_model(MODEL_FILE)
        model.compile(optimizer=optimizers.SGD(lr=learning_rate, momentum=0.9),
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])
        print("Loaded model... ", MODEL_FILE)

    mean_matrix = load_mean_matrix()
    training_datagen = ImageDataGenerator()
    training_generator = training_datagen.flow_from_directory(DATA_FOLDER, batch_size=128, seed=RANDOM_SEED)
    training_crop_generator = crop_generator(training_generator, mean_matrix)

    model_checkpoint = ModelCheckpoint("models/model.{epoch:02d}.hdf5")
    validation_accuracy = CalculateValidationScore()
    callbacks = [model_checkpoint, validation_accuracy]

    model.fit_generator(training_crop_generator, steps_per_epoch=10009, epochs=EPOCHS, verbose=2, callbacks=callbacks)


train_model()
