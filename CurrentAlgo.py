import numpy as np
import theano
import theano.tensor as T
import os
import lasagne
import time
import h5py
import csv
from scipy import misc
from keras.preprocessing.image import ImageDataGenerator
from DataAugmentation import buffered_gen_mp, buffered_gen_threaded

def residual_block(incoming, num_filters=64, filter_size=(3, 3), nonlinearity=lasagne.nonlinearities.rectify):

    conv = lasagne.layers.Conv2DLayer(incoming, num_filters=num_filters/2, filter_size=(1, 1), pad='same',
                                      W=lasagne.init.HeNormal(gain='relu'), flip_filters=False,
                                      nonlinearity=None)
    conv = lasagne.layers.BatchNormLayer(conv)
    conv = lasagne.layers.NonlinearityLayer(conv, nonlinearity=nonlinearity)

    conv = lasagne.layers.Conv2DLayer(conv, num_filters=num_filters/2, filter_size=filter_size, pad='same',
                                      W=lasagne.init.HeNormal(gain='relu'), flip_filters=False,
                                      nonlinearity=None)
    conv = lasagne.layers.BatchNormLayer(conv)
    conv = lasagne.layers.NonlinearityLayer(conv, nonlinearity=nonlinearity)

    conv = lasagne.layers.Conv2DLayer(conv, num_filters=num_filters, filter_size=(1, 1), pad='same',
                                      W=lasagne.init.HeNormal(gain='relu'), flip_filters=False,
                                      nonlinearity=None)
    conv = lasagne.layers.BatchNormLayer(conv)

    conv = lasagne.layers.ElemwiseSumLayer([conv, incoming])
    conv = lasagne.layers.ParametricRectifierLayer(conv)
    return conv


def bottleneck_block(incoming, num_filters=64, filter_size=(3, 3), bottleneck_size=None,
                     nonlinearity=lasagne.nonlinearities.rectify):
    if bottleneck_size is None:
        bottleneck_size = num_filters / 4

    conv = lasagne.layers.Conv2DLayer(incoming, num_filters=bottleneck_size, filter_size=(1, 1), pad='same',
                                      nonlinearity=lasagne.nonlinearities.linear)
    conv = lasagne.layers.BatchNormLayer(conv)
    conv = lasagne.layers.NonlinearityLayer(conv, nonlinearity=nonlinearity)

    conv = lasagne.layers.Conv2DLayer(conv, num_filters=bottleneck_size, filter_size=filter_size, pad='same',
                                      nonlinearity=lasagne.nonlinearities.linear)
    conv = lasagne.layers.BatchNormLayer(conv)
    conv = lasagne.layers.NonlinearityLayer(conv, nonlinearity=nonlinearity)

    conv = lasagne.layers.Conv2DLayer(conv, num_filters=num_filters, filter_size=(1, 1), pad='same',
                                      nonlinearity=lasagne.nonlinearities.linear)
    conv = lasagne.layers.BatchNormLayer(conv)
    conv = lasagne.layers.ElemwiseSumLayer([conv, incoming])
    conv = lasagne.layers.ParametricRectifierLayer(conv)
    return conv


def build_cnn(input_var=None):
    # Input layer:
    network = lasagne.layers.InputLayer(shape=(None, 3, 144, 192),
                                        input_var=input_var)


    network = lasagne.layers.Conv2DLayer(network, num_filters=64, filter_size=(11, 11), stride=(3, 3),
                                      W=lasagne.init.HeNormal(gain='relu'), flip_filters=False,
                                      nonlinearity=None)
    network = lasagne.layers.BatchNormLayer(network)
    network = lasagne.layers.NonlinearityLayer(network, nonlinearity=lasagne.nonlinearities.rectify)

    # Res Blocks of filters = 64
    network = residual_block(network, num_filters=64, filter_size=(3, 3), nonlinearity=lasagne.nonlinearities.rectify)
    network = residual_block(network, num_filters=64, filter_size=(3, 3), nonlinearity=lasagne.nonlinearities.rectify)
    network = residual_block(network, num_filters=64, filter_size=(3, 3), nonlinearity=lasagne.nonlinearities.rectify)
    network = residual_block(network, num_filters=64, filter_size=(3, 3), nonlinearity=lasagne.nonlinearities.rectify)
    network = residual_block(network, num_filters=64, filter_size=(3, 3), nonlinearity=lasagne.nonlinearities.rectify)
    network = lasagne.layers.MaxPool2DLayer(network, pool_size=(2, 2), stride=(2, 2))

    network = lasagne.layers.Conv2DLayer(network, num_filters=128, filter_size=(3, 3), pad='same',
                                         W=lasagne.init.HeNormal(gain='relu'), flip_filters=False,
                                         nonlinearity=None)
    network = lasagne.layers.BatchNormLayer(network)
    network = lasagne.layers.NonlinearityLayer(network, nonlinearity=lasagne.nonlinearities.rectify)
    # Res Blocks of filters = 128
    network = residual_block(network, num_filters=128, filter_size=(3, 3), nonlinearity=lasagne.nonlinearities.rectify)
    network = residual_block(network, num_filters=128, filter_size=(3, 3), nonlinearity=lasagne.nonlinearities.rectify)
    network = residual_block(network, num_filters=128, filter_size=(3, 3), nonlinearity=lasagne.nonlinearities.rectify)
    network = residual_block(network, num_filters=128, filter_size=(3, 3), nonlinearity=lasagne.nonlinearities.rectify)
    network = residual_block(network, num_filters=128, filter_size=(3, 3), nonlinearity=lasagne.nonlinearities.rectify)
    network = lasagne.layers.MaxPool2DLayer(network, pool_size=(2, 2), stride=(2, 2))

    network = lasagne.layers.Conv2DLayer(network, num_filters=256, filter_size=(3, 3), pad='same',
                                         W=lasagne.init.HeNormal(gain='relu'), flip_filters=False,
                                         nonlinearity=None)
    network = lasagne.layers.BatchNormLayer(network)
    network = lasagne.layers.NonlinearityLayer(network, nonlinearity=lasagne.nonlinearities.rectify)
    # Res Blocks of filters = 256
    network = residual_block(network, num_filters=256, filter_size=(3, 3), nonlinearity=lasagne.nonlinearities.rectify)
    network = residual_block(network, num_filters=256, filter_size=(3, 3), nonlinearity=lasagne.nonlinearities.rectify)
    network = residual_block(network, num_filters=256, filter_size=(3, 3), nonlinearity=lasagne.nonlinearities.rectify)
    network = residual_block(network, num_filters=256, filter_size=(3, 3), nonlinearity=lasagne.nonlinearities.rectify)
    network = residual_block(network, num_filters=256, filter_size=(3, 3), nonlinearity=lasagne.nonlinearities.rectify)
    #network = lasagne.layers.MaxPool2DLayer(network, pool_size=(2, 2), stride=(2, 2))

    network = lasagne.layers.Conv2DLayer(network, num_filters=512, filter_size=(3, 3), pad='same',
                                         W=lasagne.init.HeNormal(gain='relu'), flip_filters=False,
                                         nonlinearity=None)
    network = lasagne.layers.BatchNormLayer(network)
    network = lasagne.layers.NonlinearityLayer(network, nonlinearity=lasagne.nonlinearities.rectify)
    # Res Blocks of filters = 512
    network = residual_block(network, num_filters=512, filter_size=(3, 3), nonlinearity=lasagne.nonlinearities.rectify)
    network = residual_block(network, num_filters=512, filter_size=(3, 3), nonlinearity=lasagne.nonlinearities.rectify)
    network = residual_block(network, num_filters=512, filter_size=(3, 3), nonlinearity=lasagne.nonlinearities.rectify)
    network = residual_block(network, num_filters=512, filter_size=(3, 3), nonlinearity=lasagne.nonlinearities.rectify)
    network = residual_block(network, num_filters=512, filter_size=(3, 3), nonlinearity=lasagne.nonlinearities.rectify)

    network = lasagne.layers.GlobalPoolLayer(network)

    network = lasagne.layers.flatten(network)

    # And, finally, the 10-unit output layer with 50% dropout on its inputs:
    network = lasagne.layers.DenseLayer(lasagne.layers.dropout(network, p=.65), num_units=10,
                                        nonlinearity=lasagne.nonlinearities.softmax)

    return network


def iterate_minibatches(inputs, targets, batchsize, length, shuffle=False):
    if shuffle:
        indices = np.arange(len(inputs))
        np.random.shuffle(indices)
    for start_idx in range(0, length - batchsize + 1, batchsize):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize]
        else:
            excerpt = slice(start_idx, start_idx + batchsize)
        yield inputs[excerpt], targets[excerpt]


def iterate_minibatches_train(inputs, targets, batchsize, start, length, shuffle=False, augment=False):
    if shuffle:
        indices = np.arange(len(inputs))
        np.random.shuffle(indices)
    for start_idx in range(0, length - batchsize + 1, batchsize):
        if shuffle:
            excerpt = indices[start_idx + start:start_idx + start + batchsize]
        else:
            excerpt = slice(start_idx + start, start_idx + start + batchsize)

        if augment:
            padded = np.pad(inputs[excerpt], ((0, 0), (0, 0), (4, 4), (4, 4)), mode='constant')
            random_cropped = np.zeros(inputs[excerpt].shape, dtype=np.float32)
            crops = np.random.random_integers(0, high=8, size=(batchsize, 2))
            for r in range(batchsize):
                random_cropped[r, :, :, :] = padded[r, :, crops[r, 0]:(crops[r, 0] + 144),
                                             crops[r, 1]:(crops[r, 1] + 192)]
            inp_exc = random_cropped
        else:
            inp_exc = inputs[excerpt]

        yield inp_exc, targets[excerpt]


def main(num_epochs=100):

    datagen = ImageDataGenerator(
        rotation_range=30,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        vertical_flip=True,
        fill_mode='nearest')

    path = 'C:\\Users\\nikhil\\PycharmProjects\\imgs\\train'
    test_path = 'C:\\Users\\nikhil\\PycharmProjects\\imgs\\test'
    # Prepare Theano variables for inputs and targets
    input_var = T.tensor4('inputs')
    target_var = T.ivector('targets')
    network = build_cnn(input_var)
    prediction = lasagne.layers.get_output(network)
    loss = lasagne.objectives.categorical_crossentropy(prediction, target_var)
    loss = loss.mean()
    all_layers = lasagne.layers.get_all_layers(network)
    l2_penalty = lasagne.regularization.regularize_layer_params(all_layers, lasagne.regularization.l2) * 0.0001
    loss += l2_penalty

    # load:
    f = np.load("network.npz")
    params = [f["param%d" % i] for i in range(len(f.files))]
    f.close()
    lasagne.layers.set_all_param_values(network, params)

    params = lasagne.layers.get_all_params(network, trainable=True)
    updates = lasagne.updates.adadelta(loss, params)

    test_prediction = lasagne.layers.get_output(network, deterministic=True)
    test_loss = lasagne.objectives.categorical_crossentropy(test_prediction, target_var)
    test_loss = test_loss.mean()

    test_acc = T.mean(T.eq(T.argmax(test_prediction, axis=1), target_var), dtype=theano.config.floatX)

    train_fn = theano.function([input_var, target_var], loss, updates=updates)
    predict_fn = theano.function([input_var], test_prediction)

    val_fn = theano.function([input_var, target_var], [test_loss, test_acc])

    X = []
    Y = []
    with h5py.File('data_color.h5', 'r') as hf:
        X = np.array(hf.get('trainX'))
        Y = np.array(hf.get('trainY'))
    np.random.seed(31)
    for epoch in range(num_epochs):
        train_err = 0
        train_batches = 0
        start_time = time.time()
        prob = np.random.randint(0, 10)
        if prob > 7:
            for batch in buffered_gen_threaded(
                    iterate_minibatches_train(X, Y, 15, 0, 5000, shuffle=True, augment=True),
                    buffer_size=4):
                inputs, targets = batch
                inputs = inputs.astype(np.float32)
                train_err += train_fn(inputs, targets)
                train_batches += 1
        else:
            for batch in buffered_gen_threaded(
                    datagen.flow(X, Y, batch_size=15, shuffle=True, seed=epoch + prob), buffer_size=4):
                inputs, targets = batch
                inputs = inputs.astype(np.float32)
                train_err += train_fn(inputs, targets)
                train_batches += 1
                if train_batches >= 700:
                    break

        Xvalidate = X[12000:12600]
        Yvalidate = Y[12000:12600]
        valid_err = 0
        valid_batches = 0
        for batch in iterate_minibatches(Xvalidate, Yvalidate, 15, Xvalidate.shape[0], shuffle=False):
            inputs, targets = batch
            inputs = inputs.astype(np.float32)
            valid_err += val_fn(inputs, targets)[0]
            valid_batches += 1
        print("Epoch {} of {} took {:.3f}s".format(epoch + 1, num_epochs, time.time() - start_time))
        print("  Training loss:\t\t{:.6f}".format(train_err/train_batches))
        print("  Validation loss: " + str(valid_err/valid_batches))

    values = lasagne.layers.get_all_param_values(network)
    # save:
    np.savez("network.npz", **{"param%d" % i: param for i, param in enumerate(values)})

    '''
    with open('result_file.csv', 'wb') as outcsv:
        writer = csv.writer(outcsv)
        row_headers = []
        row_headers.append("img")
        for i in range(10):
            row_headers.append("c" + str(i))
        writer.writerow(row_headers)

        for f in os.listdir(test_path):
            Xtest = []
            curr_path = test_path + '\\' + f
            data = misc.imread(curr_path)
            data = misc.imresize(data, 0.3)
            data = np.transpose(data, (2, 0, 1))
            data = data.astype(np.float32)
            Xtest.append(data)
            Xtest = np.array(Xtest)
            preds = predict_fn(Xtest)
            row_headers = []
            row_headers.append(str(f))
            for i in range(preds.shape[1]):
                row_headers.append(preds[0][i])
            writer.writerow(row_headers)
    '''
main(20)
