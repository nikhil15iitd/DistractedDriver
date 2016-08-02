import numpy as np
import os
from scipy import misc
import h5py
import matplotlib.pyplot as plt

def get_driver_data():
    dr = dict()
    path = 'C:\\Users\\nikhil\\PycharmProjects\\imgs\\driver_imgs_list.csv\\driver_imgs_list.csv'
    print('Read drivers data')
    f = open(path, 'r')
    line = f.readline()
    while (1):
        line = f.readline()
        if line == '':
            break
        arr = line.strip().split(',')
        dr[arr[2]] = arr[0]
    f.close()
    return dr

def calculate_mean():
    '''
    with h5py.File('data_color.h5', 'r') as hf:
        X = np.array(hf.get('trainX'))
    X = X.astype(np.float32)
    meanR = np.mean(X[:, 0, :, :])
    meanG = np.mean(X[:, 1, :, :])
    meanB = np.mean(X[:, 2, :, :])
    print('mean R: '+str(meanR))
    print('mean G: ' + str(meanG))
    print('mean B: ' + str(meanB))
    '''
    meanR = 80.158
    meanG = 97.0143
    meanB = 95.1707
    with h5py.File('data_color.h5', 'r') as hf:
        X = np.array(hf.get('trainX'))
    I = X[1].astype(np.float32)
    I[0, :, :] = I[0, :, :] - meanR
    I[1, :, :] = I[1, :, :] - meanG
    I[2, :, :] = I[2, :, :] - meanB
    I = np.transpose(I, (1, 2, 0))
    plt.imshow(I)
    plt.show()

def load_train(path):
    X = []
    Y = []
    driver_id = []
    driver_data = get_driver_data()

    for i in range(10):
        new_path = path + '\\c' + str(i)
        for f in os.listdir(new_path):
            curr_path = new_path + '\\' + f
            fbase = str(f)
            data = misc.imread(curr_path)
            data = misc.imresize(data, (224, 224))
            data = np.transpose(data, (2, 0, 1))
            X.append(data)
            Y.append(i)
            driver_id.append(driver_data[fbase])
        print(str(len(X)))

    # shuffle array in-place
    indices = np.arange(len(X))
    np.random.shuffle(indices)
    X = np.array(X)
    Y = np.array(Y)
    driver_id = np.array(driver_id)
    X = X[indices]
    Y = Y[indices]
    driver_id = driver_id[indices]

    unique_drivers = sorted(list(set(driver_id)))

    fp = np.memmap('data224.dat', dtype='uint8', mode='w+', shape=X.shape)
    fp[:] = X[:]
    fp.flush()

    with h5py.File('data224.h5', 'w') as hf:
        hf.create_dataset('trainY', data=Y)
        hf.create_dataset('trainZ', data=driver_id)
        hf.create_dataset('unique_drivers', data=unique_drivers)

path = 'C:\\Users\\nikhil\\PycharmProjects\\imgs\\train'
load_train(path)
#calculate_mean()

