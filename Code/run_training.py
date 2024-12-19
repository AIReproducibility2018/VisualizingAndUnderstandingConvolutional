import os

os.environ["THEANO_FLAGS"] = "mode=FAST_RUN,device=cuda,floatX=float32"
import lasagne
import theano
import numpy as np
import theano.tensor as T
import time

BATCH_SIZE = 128
EPOCHS = 70
INITIAL_LEARNING_RATE = 0.001
MOMENTUM = 0.9
NR_TRAINING_BATCHES = 10010
NR_VALIDATION_BATCHES = 100

input_var = T.Tensor4('inputs')
target_var = T.ivector('targets')

net = {}
# Input layer
net['input'] = lasagne.layers.InputLayer((BATCH_SIZE, 3, 224, 224), input_var=input_var)

# Layer 1
net['conv1'] = lasagne.layers.Conv2DLayer(net['input'], 1, (7, 7), 2, nonlinearity=lasagne.nonlinearities.rectify)
net['pool1'] = lasagne.layers.MaxPool2DLayer(net['conv1'], (3, 3), 2)
net['norm1'] = lasagne.layers.LocalResponseNormalization2DLayer(net['pool1'])

# Layer 2
net['conv2'] = lasagne.layers.Conv2DLayer(net['norm1'], 1, (5, 5), 2, nonlinearity=lasagne.nonlinearities.rectify)
net['pool2'] = lasagne.layers.MaxPool2DLayer(net['conv2'], (3, 3), 2)
net['norm2'] = lasagne.layers.LocalResponseNormalization2DLayer(net['pool2'])

# Layer 3
net['conv3'] = lasagne.layers.Conv2DLayer(net['norm2'], 1, (3, 3), 1, nonlinearity=lasagne.nonlinearities.rectify)

# Layer 4
net['conv4'] = lasagne.layers.Conv2DLayer(net['conv3'], 1, (3, 3), 1, nonlinearity=lasagne.nonlinearities.rectify)
net['pool4'] = lasagne.layers.MaxPool2DLayer(net['conv4'], (3, 3), 2)

# Layer 5
net['dense5'] = lasagne.layers.DenseLayer(net['pool4'], 4096)
net['dropout5'] = lasagne.layers.DropoutLayer(net['dense5'], p=0.5)

# Layer 6
net['dense6'] = lasagne.layers.DenseLayer(net['dropout5'], 4096)
net['dropout6'] = lasagne.layers.DropoutLayer(net['dense6'], p=0.5)

# Output layer
net['output'] = lasagne.layers.NonlinearityLayer(net['dropout6'], nonlinearity=lasagne.nonlinearities.softmax)

# Define training function
prediction = lasagne.layers.get_output(net['output'])
loss = lasagne.objectives.categorical_crossentropy(prediction, target_var)
loss = loss.mean()

params = lasagne.layers.get_all_params(net['output'], trainable=True)
updates = lasagne.updates.nesterov_momentum(loss, params, learning_rate=INITIAL_LEARNING_RATE, momentum=MOMENTUM)

training_function = theano.function([input_var, target_var], loss, updates=updates)

# Define validation function
validation_prediction = lasagne.layers.get_output(net['output'], deterministic=True)
validation_loss = lasagne.objectives.categorical_crossentropy(validation_prediction, target_var)
validation_loss = validation_loss.mean()
validation_acc = T.mean(T.eq(T.argmax(validation_prediction, axis=1), target_var), dtype=theano.config.floatX)

validation_function = theano.function([input_var, target_var], [validation_loss, validation_acc])


# Create minibatches
def load_batch(batch_nr, type):
    if type == 'training':
        pass
        # Load batch file
        # Load each image
        # Subtract mean
        # Generate random subimage
    elif type == 'validation':
        pass


for epoch in range(EPOCHS):
    # Train on training set
    train_err = 0
    train_batches = 0
    start_time = time.time()
    for batch in range(NR_TRAINING_BATCHES):
        inputs, targets = load_batch(batch, 'training')
        train_err += training_function(inputs, targets)
        train_batches += 1

    # Test on validation set
    validation_err = 0
    validation_acc = 0
    val_batches = 0
    for batch in range(NR_VALIDATION_BATCHES):
        inputs, targets = load_batch(batch, 'validation')
        err, acc = validation_function(inputs, targets)
        validation_err += err
        validation_acc += acc
        val_batches += 1

    print("Epoch {} of {} took {:.3f}s".format(epoch, EPOCHS, time.time() - start_time))
    print("  training loss:\t\t{:.6f}".format(train_err / train_batches))
    print("  validation loss:\t\t{:.6f}".format(validation_err / val_batches))
    print("  validation accuracy:\t\t{:.2f} %".format(validation_acc / val_batches * 100))
