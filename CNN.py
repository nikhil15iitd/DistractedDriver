import numpy as np
import theano
import theano.tensor as T
import os
import lasagne
import time
import h5py
import gzip
import pickle
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D, AveragePooling2D
from keras.optimizers import SGD, Adadelta
from keras.utils import np_utils
from keras.models import model_from_json
from keras.callbacks import History
import csv
from sklearn.cross_validation import train_test_split
from sklearn.cross_validation import KFold
from scipy import misc
from keras.preprocessing.image import ImageDataGenerator
from DataAugmentation import buffered_gen_mp, buffered_gen_threaded
import sys

sys.setrecursionlimit(10000)


def up_block(incoming, num_filters=64, filter_size=(3, 3), nonlinearity=lasagne.nonlinearities.rectify):
    # left branch
    projection = lasagne.layers.Conv2DLayer(incoming, num_filters=num_filters, filter_size=(1, 1), flip_filters=False,
                                            b=None,
                                            nonlinearity=None)
    projection = lasagne.layers.BatchNormLayer(projection)

    # right branch
    conv = lasagne.layers.Conv2DLayer(incoming, num_filters=num_filters / 2, filter_size=(1, 1), flip_filters=False,
                                      W=lasagne.init.HeNormal(gain='relu'),
                                      nonlinearity=None)
    conv = lasagne.layers.BatchNormLayer(conv)
    conv = lasagne.layers.ParametricRectifierLayer(conv)

    conv = lasagne.layers.Conv2DLayer(conv, num_filters=num_filters / 2, filter_size=filter_size, pad='same',
                                      flip_filters=False,
                                      W=lasagne.init.HeNormal(gain='relu'),
                                      nonlinearity=None)
    conv = lasagne.layers.BatchNormLayer(conv)
    conv = lasagne.layers.ParametricRectifierLayer(conv)

    conv = lasagne.layers.Conv2DLayer(conv, num_filters=num_filters, filter_size=(1, 1), flip_filters=False,
                                      W=lasagne.init.HeNormal(gain='relu'),
                                      nonlinearity=None)
    conv = lasagne.layers.BatchNormLayer(conv)
    conv = lasagne.layers.ElemwiseSumLayer([conv, projection])
    conv = lasagne.layers.ParametricRectifierLayer(conv)
    return conv


def residual_block(incoming, num_filters=64, filter_size=(3, 3), nonlinearity=lasagne.nonlinearities.rectify):
    conv = lasagne.layers.Conv2DLayer(incoming, num_filters=num_filters / 2, filter_size=(1, 1), flip_filters=False,
                                      W=lasagne.init.HeNormal(gain='relu'),
                                      nonlinearity=None)
    conv = lasagne.layers.BatchNormLayer(conv)
    conv = lasagne.layers.ParametricRectifierLayer(conv)

    conv = lasagne.layers.Conv2DLayer(conv, num_filters=num_filters / 2, filter_size=filter_size, pad='same',
                                      flip_filters=False,
                                      W=lasagne.init.HeNormal(gain='relu'),
                                      nonlinearity=None)
    conv = lasagne.layers.BatchNormLayer(conv)
    conv = lasagne.layers.ParametricRectifierLayer(conv)

    conv = lasagne.layers.Conv2DLayer(conv, num_filters=num_filters, filter_size=(1, 1), flip_filters=False,
                                      W=lasagne.init.HeNormal(gain='relu'),
                                      nonlinearity=None)
    conv = lasagne.layers.BatchNormLayer(conv)
    conv = lasagne.layers.ElemwiseSumLayer([conv, incoming])
    conv = lasagne.layers.ParametricRectifierLayer(conv)
    return conv


def build_cnn(input_var=None):
    network = lasagne.layers.InputLayer(shape=(None, 3, 144, 192), input_var=input_var)

    network = lasagne.layers.Conv2DLayer(network, num_filters=128, filter_size=(7, 7), stride=(3, 3),
                                         flip_filters=False,
                                         W=lasagne.init.HeNormal(gain='relu'), nonlinearity=None)
    network = lasagne.layers.BatchNormLayer(network)
    network = lasagne.layers.FeaturePoolLayer(network, pool_size=2)

    for i in range(5):
        network = residual_block(network, num_filters=64, filter_size=(3, 3),
                                 nonlinearity=lasagne.nonlinearities.rectify)
    network = lasagne.layers.MaxPool2DLayer(network, pool_size=(2, 2), stride=(2, 2))

    network = up_block(network, num_filters=128, filter_size=(3, 3), nonlinearity=lasagne.nonlinearities.rectify)
    for i in range(5):
        network = residual_block(network, num_filters=128, filter_size=(3, 3),
                                 nonlinearity=lasagne.nonlinearities.rectify)
    network = lasagne.layers.MaxPool2DLayer(network, pool_size=(2, 2), stride=(2, 2))

    network = up_block(network, num_filters=256, filter_size=(3, 3), nonlinearity=lasagne.nonlinearities.rectify)
    for i in range(4):
        network = residual_block(network, num_filters=256, filter_size=(3, 3),
                                 nonlinearity=lasagne.nonlinearities.rectify)
    network = lasagne.layers.MaxPool2DLayer(network, pool_size=(2, 2), stride=(2, 2))

    network = up_block(network, num_filters=512, filter_size=(3, 3), nonlinearity=lasagne.nonlinearities.rectify)
    for i in range(3):
        network = residual_block(network, num_filters=512, filter_size=(3, 3),
                                 nonlinearity=lasagne.nonlinearities.rectify)

    network = lasagne.layers.Conv2DLayer(network, num_filters=1024, filter_size=(3, 3), pad='same',
                                         flip_filters=False,
                                         W=lasagne.init.GlorotUniform(),
                                         nonlinearity=None)
    network = lasagne.layers.BatchNormLayer(network)
    network = lasagne.layers.FeaturePoolLayer(network, pool_size=4)

    network = lasagne.layers.GlobalPoolLayer(network)

    network = lasagne.layers.flatten(network)

    # Output
    network = lasagne.layers.DenseLayer(lasagne.layers.dropout(network, p=.6), num_units=10,
                                        nonlinearity=lasagne.nonlinearities.softmax)

    return network


def vgg_std16_model():
    model = Sequential()
    model.add(ZeroPadding2D((1, 1), input_shape=(3, 224, 224)))
    model.add(Convolution2D(64, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(64, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(128, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(128, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))

    # Add another conv layer with ReLU + GAP
    model.add(Convolution2D(1024, 3, 3, activation='relu', border_mode="same"))
    model.add(AveragePooling2D((14, 14)))
    model.add(Flatten())

    model.add(Dense(10, activation='softmax'))
    model.name = "VGGCAM"
    return model


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
            padded = np.pad(inputs[excerpt], ((0, 0), (0, 0), (8, 8), (8, 8)), mode='constant')
            random_cropped = np.zeros(inputs[excerpt].shape, dtype=np.float32)
            crops = np.random.random_integers(0, high=16, size=(batchsize, 2))
            for r in range(batchsize):
                random_cropped[r, :, :, :] = padded[r, :, crops[r, 0]:(crops[r, 0] + 144),
                                             crops[r, 1]:(crops[r, 1] + 192)]
            inp_exc = random_cropped
        else:
            inp_exc = inputs[excerpt]

        yield inp_exc, targets[excerpt]


def create_submission(K):
    path = 'C:\\Users\\nikhil\\PycharmProjects\\imgs\\test'
    input_var = T.tensor4('inputs')
    network = build_cnn(input_var)
    # load:
    f = np.load("resnet100.npz")
    params = [f["param%d" % i] for i in range(len(f.files))]
    f.close()
    lasagne.layers.set_all_param_values(network, params)

    test_prediction = lasagne.layers.get_output(network, deterministic=True)
    predict_fn = theano.function([input_var], test_prediction)

    with open('result_file.csv', 'wb') as outcsv:
        writer = csv.writer(outcsv)
        row_headers = []
        row_headers.append("img")
        for i in range(10):
            row_headers.append("c" + str(i))
        writer.writerow(row_headers)

        for f in os.listdir(path):
            Xtest = []
            curr_path = path + '\\' + f
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


def zero_out(X):
    tx = np.random.randint(0, high=180)
    ty = np.random.randint(0, high=180)
    X[:, :, tx:tx + 32, ty:ty + 32] = np.zeros((32, 32))
    return X


def fit(Xtrain, Ytrain, Xvalidate, Yvalidate, fold_num, num_epochs):
    datagen = ImageDataGenerator(
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest')
    meanR = 80.158
    meanG = 97.0143
    meanB = 95.1707

    input_var = T.tensor4('inputs')
    target_var = T.ivector('targets')

    network = build_cnn(input_var)

    prediction = lasagne.layers.get_output(network)
    loss = lasagne.objectives.categorical_crossentropy(prediction, target_var)
    loss = loss.mean()
    all_layers = lasagne.layers.get_all_layers(network)
    l2_penalty = lasagne.regularization.regularize_layer_params(all_layers, lasagne.regularization.l2) * 0.0001
    loss += l2_penalty
    '''
    # load:
    model_filename = 'resnet' + str(fold_num) + '.npz'
    if os.path.isfile(model_filename):
        f = np.load(model_filename)
        params = [f["param%d" % i] for i in range(len(f.files))]
        f.close()
        lasagne.layers.set_all_param_values(network, params)
    '''
    f = gzip.open('wide_resnet_n2_k4.pklz', 'rb')
    values = pickle.load(f)
    params = values
    f.close()
    lasagne.layers.set_all_param_values(network, params)

    params = lasagne.layers.get_all_params(network, trainable=True)
    updates = lasagne.updates.adadelta(loss, params)

    test_prediction = lasagne.layers.get_output(network, deterministic=True)
    test_loss = lasagne.objectives.categorical_crossentropy(test_prediction, target_var)
    test_loss = test_loss.mean()

    train_fn = theano.function([input_var, target_var], loss, updates=updates)
    val_fn = theano.function([input_var, target_var], test_loss)
    np.random.seed(1234)
    for epoch in range(num_epochs):
        train_err = 0
        train_batches = 0
        start_time = time.time()
        prob = np.random.randint(0, 10)
        if epoch % 2 == 0:
            for batch in buffered_gen_threaded(
                    iterate_minibatches_train(Xtrain, Ytrain, 15, 0, 4000, shuffle=True, augment=True),
                    buffer_size=4):
                inputs, targets = batch
                inputs = inputs.astype(np.float32)
                inputs[:, 0, :, :] = inputs[:, 0, :, :] - meanR
                inputs[:, 1, :, :] = inputs[:, 1, :, :] - meanG
                inputs[:, 2, :, :] = inputs[:, 2, :, :] - meanB
                if prob > 6:
                    inputs = zero_out(inputs)
                train_err += train_fn(inputs, targets)
                train_batches += 1
        else:
            for batch in buffered_gen_threaded(
                    datagen.flow(Xtrain, Ytrain, batch_size=15, shuffle=True, seed=epoch + 4321), buffer_size=4):
                inputs, targets = batch
                inputs = inputs.astype(np.float32)
                inputs[:, 0, :, :] = inputs[:, 0, :, :] - meanR
                inputs[:, 1, :, :] = inputs[:, 1, :, :] - meanG
                inputs[:, 2, :, :] = inputs[:, 2, :, :] - meanB
                if prob > 5:
                    inputs = zero_out(inputs)
                train_err += train_fn(inputs, targets)
                train_batches += 1
                if train_batches >= 500:
                    break

        valid_err = 0
        valid_batches = 0
        for batch in iterate_minibatches(Xvalidate, Yvalidate, 15, Xvalidate.shape[0], shuffle=False):
            inputs, targets = batch
            inputs = inputs.astype(np.float32)
            inputs[:, 0, :, :] = inputs[:, 0, :, :] - meanR
            inputs[:, 1, :, :] = inputs[:, 1, :, :] - meanG
            inputs[:, 2, :, :] = inputs[:, 2, :, :] - meanB
            valid_err += val_fn(inputs, targets)
            valid_batches += 1
        print("Epoch {} of {} took {:.3f}s".format(epoch + 1, num_epochs, time.time() - start_time))
        print("  Training loss:\t\t{:.6f}".format(train_err / train_batches))
        print("  Validation loss: " + str(valid_err / valid_batches))

    values = lasagne.layers.get_all_param_values(network)
    # np.savez(model_filename, **{"param%d" % i: param for i, param in enumerate(values)})


def copy_selected_drivers(train_data, train_target, driver_id, driver_list):
    data = []
    target = []
    for i in range(driver_id.shape[0]):
        if driver_id[i] in driver_list:
            data.append(train_data[i])
            target.append(train_target[i])
    data = np.array(data)
    target = np.array(target)
    return data, target


def main(nfolds):
    random_state = 20
    with h5py.File('data_color.h5', 'r') as hf:
        X = np.array(hf.get('trainX'))
        Y = np.array(hf.get('trainY'))
        driver_id = np.array(hf.get('trainZ'))
        unique_drivers = list(hf.get('unique_drivers'))
    num_fold = 0
    kf = KFold(len(unique_drivers), n_folds=nfolds, shuffle=True, random_state=random_state)
    for train_drivers, test_drivers in kf:
        unique_list_train = [unique_drivers[i] for i in train_drivers]
        X_train, Y_train = copy_selected_drivers(X, Y, driver_id, unique_list_train)
        unique_list_valid = [unique_drivers[i] for i in test_drivers]
        X_valid, Y_valid = copy_selected_drivers(X, Y, driver_id, unique_list_valid)

        num_fold += 1
        print('Start KFold number {} from {}'.format(num_fold, nfolds))
        print('Split train: ', len(X_train), len(Y_train))
        print('Split valid: ', len(X_valid), len(Y_valid))
        print('Train drivers: ', unique_list_train)
        print('Test drivers: ', unique_list_valid)
        fit(X_train, Y_train, X_valid, Y_valid, num_fold, 30)


def single():
    with h5py.File('data_color.h5', 'r') as hf:
        X = np.array(hf.get('trainX'))
        Y = np.array(hf.get('trainY'))
        driver_id = np.array(hf.get('trainZ'))
        unique_drivers = list(hf.get('unique_drivers'))
    unique_list_train = unique_drivers[:-2]
    X_train, Y_train = copy_selected_drivers(X, Y, driver_id, unique_list_train)
    unique_list_valid = unique_drivers[-2:]
    X_valid, Y_valid = copy_selected_drivers(X, Y, driver_id, unique_list_valid)
    X = []
    Y = []
    print('Split train: ', len(X_train), len(Y_train))
    print('Split valid: ', len(X_valid), len(Y_valid))
    print('Train drivers: ', unique_list_train)
    print('Test drivers: ', unique_list_valid)
    fit(X_train, Y_train, X_valid, Y_valid, 0, 20)


def save_model(model, index, cross=''):
    json_string = model.to_json()
    if not os.path.isdir('cache'):
        os.mkdir('cache')
    json_name = 'architecture' + str(index) + cross + '.json'
    weight_name = 'model_weights' + str(index) + cross + '.h5'
    open(os.path.join('cache', json_name), 'w').write(json_string)
    model.save_weights(os.path.join('cache', weight_name), overwrite=True)


def read_model(index, cross=''):
    json_name = 'architecture' + str(index) + cross + '.json'
    weight_name = 'model_weights' + str(index) + cross + '.h5'
    model = model_from_json(open(os.path.join('cache', json_name)).read())
    model.load_weights(os.path.join('cache', weight_name))
    return model


def VGG_predict(number, modelname):
    MEAN_VALUE = np.array([103.939, 116.779, 123.68])
    model = vgg_std16_model()
    path = 'C:\\Users\\nikhil\\PycharmProjects\\imgs\\test'
    model1 = read_model(number, modelname)
    model2 = read_model(11, modelname)
    test_labels = {}
    with open('result_file.csv', 'wb') as outcsv:
        writer = csv.writer(outcsv)
        row_headers = []
        row_headers.append("img")
        for i in range(10):
            row_headers.append("c" + str(i))
        writer.writerow(row_headers)

        for f in os.listdir(path):
            Xtest = []
            curr_path = path + '\\' + f
            data = misc.imread(curr_path)
            data = misc.imresize(data, (224, 224))
            data = np.transpose(data, (2, 0, 1))
            data = data.astype(np.float32)
            data = data[::-1, :, :]
            data[0, :, :] = data[0, :, :] - MEAN_VALUE[0]
            data[1, :, :] = data[1, :, :] - MEAN_VALUE[1]
            data[2, :, :] = data[2, :, :] - MEAN_VALUE[2]
            Xtest.append(data)
            Xtest = np.array(Xtest)
            preds1 = model1.predict(Xtest, batch_size=1, verbose=1)
            preds2 = model2.predict(Xtest, batch_size=1, verbose=1)
            # get predicted labels for semi supervised learning
            label1 = np.argmax(preds1[0])
            label2 = np.argmax(preds2[0])
            mean_value = (preds1[0][label1] + preds2[0][label2]) / 2
            if mean_value < 0.9:
                if np.random.randint(0, high=10) > 4:
                    test_labels[f] = label1
                else:
                    test_labels[f] = label2
            row_headers = []
            row_headers.append(str(f))
            for i in range(preds1.shape[1]):
                mu = (preds1[0][i] + preds2[0][i]) / 2
                row_headers.append(mu)
            writer.writerow(row_headers)

    with open('test_labels.p', 'wb') as fp:
        pickle.dump(test_labels, fp)


def VGG_single(num_epochs, modelnumber, modelname):
    datagen = ImageDataGenerator(
        rotation_range=35,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest')
    model = vgg_std16_model()

    model = read_model(modelnumber, modelname)
    '''
    # Load weights for pretrained VGG16 model
    with h5py.File('vgg16_weights.h5', 'r') as hf:
        for k in range(hf.attrs['nb_layers']):
            g = hf['layer_{}'.format(k)]
            weights = [g['param_{}'.format(p)] for p in range(g.attrs['nb_params'])]
            model.layers[k].set_weights(weights)
            if model.layers[k].name == "convolution2d_13":
                break
        print('Model loaded.')
    '''
    sgd = SGD(lr=1e-4, decay=4e-4, momentum=0.7, nesterov=True)
    model.compile(optimizer=sgd, loss='categorical_crossentropy')

    with h5py.File('data224.h5', 'r') as hf:
        trainY = np.array(hf.get('trainY'))
        driver_id = np.array(hf.get('trainZ'))
        unique_drivers = list(hf.get('unique_drivers'))
    MEAN_VALUE = np.array([103.939, 116.779, 123.68])

    # read from memmap file
    train = np.memmap('data224.dat', dtype='uint8', mode='r', shape=(22424, 3, 224, 224))
    train_indices = [i for i in range(train.shape[0])]

    unique_list_train = unique_drivers
    unique_list_valid = unique_drivers[-1]
    print('Train drivers: ', unique_list_train)
    print('Test drivers: ', unique_list_valid)

    for epoch in range(num_epochs):
        print('Epoch: ' + str(epoch + 1))
        np.random.seed(3 + epoch)
        for i in range(0, train.shape[0], 1024):
            if i + 1024 > train.shape[0] and i < train.shape[0] - 1:
                idxs = train_indices[i:]
            else:
                idxs = train_indices[i:i + 1024]
            X = train[idxs]
            Y = trainY[idxs]
            driverids = driver_id[idxs]

            X_train, Y_train = copy_selected_drivers(X, Y, driverids, unique_list_train)
            X_valid, Y_valid = copy_selected_drivers(X, Y, driverids, unique_list_valid)
            X = []
            Y = []
            # print('Split train: ', len(X_train), len(Y_train))
            # print('Split valid: ', len(X_valid), len(Y_valid))
            Y_valid = np_utils.to_categorical(Y_valid)
            Y_train = np_utils.to_categorical(Y_train)
            X_valid = X_valid[:, ::-1, :, :]
            X_valid = X_valid.astype(np.float32)
            X_valid[:, 0, :, :] = X_valid[:, 0, :, :] - MEAN_VALUE[0]
            X_valid[:, 1, :, :] = X_valid[:, 1, :, :] - MEAN_VALUE[1]
            X_valid[:, 2, :, :] = X_valid[:, 2, :, :] - MEAN_VALUE[2]

            # train method 2
            if np.random.randint(0, 10) > 4 and epoch > 3:
                train_batches = 0
                for batch in datagen.flow(X_train, Y_train, batch_size=X_train.shape[0], shuffle=True, seed=epoch + 7):
                    inputs, targets = batch
                    inputs = inputs.astype(np.float32)
                    # convert to BGR
                    inputs = inputs[:, ::-1, :, :]
                    if np.random.randint(0, high=10) > 6:
                        inputs = zero_out(inputs)
                    inputs[:, 0, :, :] = inputs[:, 0, :, :] - MEAN_VALUE[0]
                    inputs[:, 1, :, :] = inputs[:, 1, :, :] - MEAN_VALUE[1]
                    inputs[:, 2, :, :] = inputs[:, 2, :, :] - MEAN_VALUE[2]
                    train_batches += 1
                    model.fit(inputs, targets, batch_size=16, nb_epoch=1, verbose=1, shuffle=True)
                    if train_batches > 0:
                        break

            # train method 1
            # convert to BGR
            X_train = X_train[:, ::-1, :, :]
            X_train = X_train.astype(np.float32)
            # augment data
            if np.random.randint(0, 10) > 4:
                padded = np.pad(X_train, ((0, 0), (0, 0), (32, 32), (32, 32)), mode='constant')
                random_cropped = np.zeros(X_train.shape, dtype=np.float32)
                for r in range(X_train.shape[0]):
                    crops = np.random.random_integers(0, high=64, size=(1, 2))
                    random_cropped[r, :, :, :] = padded[r, :, crops[0, 0]:(crops[0, 0] + 224),
                                                 crops[0, 1]:(crops[0, 1] + 224)]
                X_train = random_cropped
            X_train[:, 0, :, :] = X_train[:, 0, :, :] - MEAN_VALUE[0]
            X_train[:, 1, :, :] = X_train[:, 1, :, :] - MEAN_VALUE[1]
            X_train[:, 2, :, :] = X_train[:, 2, :, :] - MEAN_VALUE[2]
            model.fit(X_train, Y_train, batch_size=16, nb_epoch=1, verbose=1, shuffle=True)

    save_model(model, modelnumber, modelname)


def VGG_semi_supervised(num_epochs, in_number, modelname_in, out_number, modelname_out):
    path = 'C:\\Users\\nikhil\\PycharmProjects\\imgs\\test'
    model = vgg_std16_model()
    model = read_model(in_number, modelname_in)
    sgd = SGD(lr=7e-5, decay=4e-4, momentum=0.8, nesterov=True)
    model.compile(optimizer=sgd, loss='categorical_crossentropy')
    MEAN_VALUE = np.array([103.939, 116.779, 123.68])

    with open('test_labels.p', 'rb') as fp:
        test_labels = pickle.load(fp)
    for f in os.listdir(path):
        if f in test_labels:
            Xtest = []
            curr_path = path + '\\' + f
            data = misc.imread(curr_path)
            data = misc.imresize(data, (224, 224))
            data = np.transpose(data, (2, 0, 1))
            data = data.astype(np.float32)
            data = data[::-1, :, :]
            data[0, :, :] = data[0, :, :] - MEAN_VALUE[0]
            data[1, :, :] = data[1, :, :] - MEAN_VALUE[1]
            data[2, :, :] = data[2, :, :] - MEAN_VALUE[2]
            Xtest.append(data)
            Xtest = np.array(Xtest)
            Ytest = np_utils.to_categorical([test_labels[f]], 10)
            model.fit(Xtest, Ytest, batch_size=1, nb_epoch=1, verbose=1)

    # train on main labelled data
    with h5py.File('data224.h5', 'r') as hf:
        trainY = np.array(hf.get('trainY'))
        unique_drivers = list(hf.get('unique_drivers'))
    # read from memmap file
    train = np.memmap('data224.dat', dtype='uint8', mode='r', shape=(22424, 3, 224, 224))
    train_indices = [i for i in range(train.shape[0])]
    print('Train drivers: ', unique_drivers)
    prev_image = train[0]
    for epoch in range(num_epochs):
        print('Epoch: ' + str(epoch + 1))
        np.random.seed(5 + epoch)
        for i in range(0, train.shape[0], 1024):
            if i + 1024 > train.shape[0] and i < train.shape[0] - 1:
                idxs = train_indices[i:]
            else:
                idxs = train_indices[i:i + 1024]
            X_train = np.array(train[idxs]).copy()
            Y_train = trainY[idxs]
            Y_train = np_utils.to_categorical(Y_train)
            # train method 1
            # augment data
            if np.random.randint(0, 10) > 4:
                padded = np.pad(X_train, ((0, 0), (0, 0), (32, 32), (32, 32)), mode='constant')
                random_cropped = np.zeros(X_train.shape, dtype=np.float32)
                for r in range(X_train.shape[0]):
                    crops = np.random.random_integers(0, high=64, size=(1, 2))
                    random_cropped[r, :, :, :] = padded[r, :, crops[0, 0]:(crops[0, 0] + 224),
                                                 crops[0, 1]:(crops[0, 1] + 224)]
                X_train = random_cropped
            # convert to BGR
            X_train = X_train[:, ::-1, :, :]
            X_train = zero_out(X_train)
            X_train = X_train.astype(np.float32)
            X_train[:, 0, :, :] = X_train[:, 0, :, :] - MEAN_VALUE[0]
            X_train[:, 1, :, :] = X_train[:, 1, :, :] - MEAN_VALUE[1]
            X_train[:, 2, :, :] = X_train[:, 2, :, :] - MEAN_VALUE[2]
            model.fit(X_train, Y_train, batch_size=16, nb_epoch=1, verbose=1, shuffle=True)
        save_model(model, out_number, modelname_out)


def readcsv():
    test_labels = {}
    idx = 0
    with open('result_file.csv', 'rb') as incsv:
        reader = csv.reader(incsv)
        for row in reader:
            if idx > 0:
                f = row[0]
                preds = np.array(row[1:])
                preds = preds.astype(np.float32)
                label = np.argmax(preds)
                if np.random.randint(0, high=10) > 6:
                    test_labels[f] = label
            idx += 1
    print(str(len(test_labels.keys())))
    print(idx)
    with open('test_labels.p', 'wb') as fp:
        pickle.dump(test_labels, fp)


def VGG_ensemble(num_iterations):
    for it in range(num_iterations):
        VGG_semi_supervised(2, 11, '_vgg_16', 11, '_vgg_16')
        VGG_semi_supervised(2, 12, '_vgg_16', 12, '_vgg_16')


# VGG_single(2, 11, '_vgg_16')
#VGG_ensemble(1)
VGG_predict(12, '_vgg_16')
# single()
#readcsv()
