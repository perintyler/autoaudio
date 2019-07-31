#!/usr/bin/env python
# coding= UTF-8
#
# Author: Fing
# Date  : 2017-12-03
#

import feature_extraction
# from feat_extract import *
import time
import argparse
import numpy as np
import keras
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Activation
from keras.layers import Dense, Dropout
from keras.layers import Embedding
from keras.layers import Conv1D, GlobalAveragePooling1D, MaxPooling1D
from keras.optimizers import SGD
import os
import os.path as op
from sklearn.model_selection import train_test_split
import file_utils

import sys

# Extracts features from the training data for the given labe
# The features are stored as 2 .npy files
def extract_features(label, feature_file, label_file, file_ext='.wav'):
    data_dir, data_exists = file_utils.get_training_data_dir(label)
    if(data_exists == False):
        # gathering training data is custom so if the data does not exist,
        # there is nothing that can be done.
        raise Exception(f"Training data for {label} does not exists. Cannot extract features")

    # features, labels = feature_extraction.parse_audio_files('training_data', file_ext=file_ext)
    features, labels = feature_extraction.parse_audio(label, data_dir, file_ext=file_ext)

    # features, labels = feature_extraction.parse_audio_files(data_dir, file_ext=file_ext)
    np.save(feature_file, features)
    np.save(label_file, labels)

    # Predict new
    features, filenames = feature_extraction.parse_predict_files('predict')
    feature_pfile, filename_pfile = file_utils.get_prediction_files(label)
    np.save(feature_pfile, features)
    np.save(filename_pfile, filenames)


def create_model(label):
    feature_file, label_file, features_exist = file_utils.get_feature_files(label)
    if(features_exist == False):
        extract_features(label, feature_file, label_file)

    X = np.load(feature_file)
    y = np.load(label_file).ravel()

    print('x', X.shape)
    print('y', y.shape)
    sys.exit(0)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=233)

    data_dir, data_exists = file_utils.get_training_data_dir(label)
    # class_count = file_utils.get_file_count(data_dir)
    class_count = 1 # number of labels
    batch_size = file_utils.get_file_count(data_dir)
    epochs = 500

    # Build the Neural Network
    model = Sequential()
    model.add(Conv1D(64, 3, activation='relu', input_shape=(193, 1)))
    model.add(Conv1D(64, 3, activation='relu'))
    model.add(MaxPooling1D(3))
    model.add(Conv1D(128, 3, activation='relu'))
    model.add(Conv1D(128, 3, activation='relu'))
    model.add(GlobalAveragePooling1D())
    model.add(Dropout(0.5))
    model.add(Dense(class_count, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

    # Convert label to onehot
    y_train = keras.utils.to_categorical(y_train - 1, num_classes=class_count)
    y_test = keras.utils.to_categorical(y_test - 1, num_classes=class_count)

    X_train = np.expand_dims(X_train, axis=2)
    X_test = np.expand_dims(X_test, axis=2)

    start = time.time()

    print('batch_size', batch_size)
    print('x shape', X_train.shape)
    print('y shape', y_train.shape)
    sys.exit(0)
    model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs)
    score, acc = model.evaluate(X_test, y_test, batch_size=16)

    print('Test score:', score)
    print('Test accuracy:', acc)
    print('Training took: %d seconds' % int(time.time() - start))

    model_file = file_utils.get_classification_model(label)
    model.save(model_file)
    return model

def predict(model):

    predict_feat_path = 'predict_feat.npy'
    predict_filenames = 'predict_filenames.npy'
    filenames = np.load(predict_filenames)
    X_predict = np.load(predict_feat_path)
    X_predict = np.expand_dims(X_predict, axis=2)
    pred = model.predict_classes(X_predict)
    for pair in list(zip(filenames, pred)): print(pair)

def classify(label):
    model_file, model_exists = file_utils.get_classification_model(label)
    model = load_model(model_file) if model_exists else create_model(label)
    predict(model)

if __name__ == '__main__':
    classify('breathing')
    # parser = argparse.ArgumentParser(description=__doc__)
    # parser.add_argument('-t', '--train',             action='store_true',                           help='train neural network with extracted features')
    # parser.add_argument('-m', '--model',             metavar='path',     default='trained_model.h5',help='use this model path on train and predict operations')
    # parser.add_argument('-e', '--epochs',            metavar='N',        default=500,              help='epochs to train', type=int)
    # parser.add_argument('-p', '--predict',           action='store_true',                           help='predict files in ./predict folder')
    # parser.add_argument('-P', '--real-time-predict', action='store_true',                           help='predict sound in real time')
    # parser.add_argument('-v', '--verbose',           action='store_true',                           help='verbose print')
    # parser.add_argument('-s', '--log-speed',         action='store_true',                           help='performance profiling')
    # parser.add_argument('-b', '--batch-size',        metavar='size',     default=64,                help='batch size', type=int)
    # args = parser.parse_args()
    # main(args)


# This would be for streaming I assume
# def real_time_predict(args):
#     import sounddevice as sd
#     import soundfile as sf
#     import queue
#     import librosa
#     import sys
#     if op.exists(args.model):
#         model = keras.models.load_model(args.model)
#         while True:
#             try:
#                 features = np.empty((0,193))
#                 start = time.time()
#                 mfccs, chroma, mel, contrast,tonnetz = extract_feature()
#                 ext_features = np.hstack([mfccs,chroma,mel,contrast,tonnetz])
#                 features = np.vstack([features,ext_features])
#                 features = np.expand_dims(features, axis=2)
#                 pred = model.predict_classes(features)
#                 for p in pred:
#                     print(p)
#                     if args.verbose: print('Time elapsed in real time feature extraction: ', time.time() - start)
#                     sys.stdout.flush()
#             except KeyboardInterrupt: parser.exit(0)
#             except Exception as e: parser.exit(type(e).__name__ + ': ' + str(e))
#     elif input('Model not found. Train network first? (y/N)') in ['y', 'yes']:
#         train()
#         real_time_predict(args)
