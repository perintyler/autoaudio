
# import feature_extraction
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
import settings

import feature_extraction

def model_exists():
    return not file_utils.is_dir_empty(settings.model_dir)

def train_classifier(features_file, labels_file):
    X = np.load(features_file)
    y = np.load(labels_file).ravel()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=233)

    # data_dir, data_exists = file_utils.get_training_data_dir(label)
    # class_count = file_utils.get_file_count(data_dir)
    class_count = 50 # number of labels
    batch_size = 40
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

    model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs)
    score, acc = model.evaluate(X_test, y_test, batch_size=16)

    print('Test score:', score)
    print('Test accuracy:', acc)
    print('Training took: %d seconds' % int(time.time() - start))

    model.save(f'{settings.model_dir}/model.h5')
    return model

def predict(model):
    predict_feat_path = 'predict_feat.npy'
    predict_filenames = 'predict_filenames.npy'
    filenames = np.load(predict_filenames)
    X_predict = np.load(predict_feat_path)
    X_predict = np.expand_dims(X_predict, axis=2)
    pred = model.predict_classes(X_predict)
    for pair in list(zip(filenames, pred)): print(pair)

if __name__ == '__main__':
    # Get features file. If features haven't been extracted yet, prompt user
    # if it should be down now. If so, extract features and then train.
    feature_file, label_file, features_extracted = feature_extraction.get_feature_files()
    if features_extracted:
        train_classifier(feature_file, label_file)
    else:
        print('Features need to be extracted. Extract Features now? (y/n)')
        user_input = input()
        if user_input == 'y':
            feature_extraction.extract_features()
            train_classifier(feature_file, label_file)
        else:
            print('Cannot train model without extracted features. Exiting.')


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
