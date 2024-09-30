import os
import datetime

import tensorflow as tf
import keras
from keras.utils import to_categorical

from utils.input_data import read_data_sets
import utils.datasets as ds
from utils.models import *


def dataset_adjustment(a_dataset, required_length):
    new_dataset = []
    for i in range(len(a_dataset)):
        new_dataset.append(data_adjustment(a_dataset[i], required_length))
    return np.array(new_dataset)

def data_adjustment(a_data, required_length):
    from skimage.transform import resize
    d = a_data
    l = required_length
    new_data = []
    return resize(d, (l, 1))

def load_data(source, categorical=False):
    nb_class = ds.nb_classes(source)
    train_data_file = os.path.join(
        'data', source, '%s_TRAIN.tsv' % source)
    test_data_file = os.path.join(
        'data', source, '%s_TEST.tsv' % source)
    x_train, y_train, x_test, y_test = read_data_sets(
        train_data_file, "", test_data_file, "", delimiter="\t")

    x_train = np.nan_to_num(x_train)
    x_test = np.nan_to_num(x_test)

    y_test = ds.class_offset(y_test, source)
    y_train = ds.class_offset(y_train, source)
    if categorical:
        y_train = to_categorical(y_train, nb_class)
        y_test = to_categorical(y_test, nb_class)

    return x_train, y_train, x_test, y_test



def pre_train_multi_source(source_list, target, dataset_balancing = True, model_architecture = "vgg",
                           save_model = False, metric = "Class-Based_Shapelet", nb_iterations = 10000, save_path=None):
    nb_class = 0
    max_num = 0
    for source in source_list:
        x_train_temp, y_train_temp, _, __ = load_data(source)
        if x_train_temp.shape[0] > max_num:
            max_num = x_train_temp.shape[0]

    # get target_length as the timesteps of target dataset
    target_length = load_data(target)[0].shape[1]
    
    if dataset_balancing ==True:
        for source in source_list:
            x_train_temp, y_train_temp, _, __ = load_data(
                source)
            x_train_temp = dataset_adjustment(x_train_temp, target_length)
            nb_class += ds.nb_classes(source)
            if x_train_temp.shape[0] != max_num:
                x_train_temp, y_train_temp = oversample_keep_ratio(x_train_temp, y_train_temp, max_num)
            if source == source_list[0]:
                x_train = x_train_temp
                y_train = y_train_temp
            else:
                x_train = np.concatenate((x_train, x_train_temp), axis=0)
                # add maximum label of y_test to y_test_temp
                for i in range(len(y_train_temp)):
                    y_train_temp[i] = y_train_temp[i] + np.max(y_train) + 1
                y_train = np.concatenate((y_train, y_train_temp), axis=0)

    else:
        for source in source_list:
            x_train_temp, y_train_temp, _, __ = load_data(
                source)
            x_train_temp = dataset_adjustment(x_train_temp, target_length)

            nb_class += ds.nb_classes(source)
            if source == source_list[0]:
                x_train = x_train_temp
                y_train = y_train_temp
            else:
                x_train = np.concatenate((x_train, x_train_temp), axis=0)
                # add maximum label of y_test to y_test_temp
                for i in range(len(y_train_temp)):
                    y_train_temp[i] = y_train_temp[i] + np.max(y_train) + 1
                y_train = np.concatenate((y_train, y_train_temp), axis=0)

    x_train = x_train.reshape(x_train.shape[0], -1)
    y_train = to_categorical(y_train, nb_class)

    batch_size = 32
    nb_dims = 1

    nb_timesteps = int(x_train.shape[1] / nb_dims)
    nb_epochs = np.ceil(nb_iterations * (batch_size /
                        x_train.shape[0])).astype(int)
    input_shape = (nb_timesteps, nb_dims)

    # load pre-training model
    pre_training_model = get_model(
        model_architecture, input_shape, nb_class)
    opt = keras.optimizers.Adam(learning_rate=0.0001)
    pre_training_model.compile(loss='categorical_crossentropy',
                               optimizer=opt, metrics=['accuracy'])
    # training
    pre_training_model.fit(
        x_train, y_train, epochs=nb_epochs, batch_size=batch_size)


    if save_model:
            pre_training_model.save(save_path)
            
    return pre_train




def oversample_keep_ratio(x_train, y_train, max_num):
    ratio = max_num // len(x_train)
    left = max_num % x_train.shape[0]
    # make ratio times of x_train
    x_train = np.repeat(x_train, ratio, axis=0)
    y_train = np.repeat(y_train, ratio, axis=0)
    # make random index of x_train
    if left > 0:
        index = np.random.randint(low=x_train.shape[0], size=left)
        x_train = np.concatenate((x_train, x_train[index]), axis=0)
        y_train = np.concatenate((y_train, y_train[index]), axis=0)

    return x_train, y_train



def fine_tuning(target, pre_trained_model, nb_iterations=5000, batch_size=32):

    nb_dims = 1
    # load data
    x_train, y_train, x_test, y_test = load_data(target, categorical=True)
    # load pre-trained model

    model = get_model('vgg', (x_train.shape[1], 1), ds.nb_classes(target))
    # copy weights except the last layer
    for i in range(len(model.layers) - 1):
        model.layers[i].set_weights(pre_trained_model.layers[i].get_weights())

    nb_epochs = np.ceil(nb_iterations * (batch_size /
                            x_train.shape[0])).astype(int)
    # compile with adam optimizer of learning rate 0.0001
    opt = keras.optimizers.Adam(learning_rate=0.0001)
    model.compile(loss='categorical_crossentropy',
                    optimizer=opt, metrics=['accuracy'])
    # fine tuning
    model.fit(x_train, y_train, epochs=nb_epochs, batch_size=batch_size)
    # evaluation
    result = model.evaluate(x_test, y_test, verbose=0)
    print('Test loss:', result[0])
    print('Test accuracy:', result[1])

    return result[0], result[1]