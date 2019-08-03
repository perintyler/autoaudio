import code
import glob
import os
import librosa
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import specgram
import soundfile as sf
import sounddevice as sd
import queue
from file_utils import is_dir_empty, get_label_index

import settings
import esc50


def get_feature_files():
    dir = settings.features_dir
    features_extracted = os.path.isdir(dir) and not is_dir_empty(dir)
    features_file = f'{settings.features_dir}/feat.py'
    labels_file = f'{settings.features_dir}/labels.py'
    return features_file, labels_file, features_extracted


# Extracts features from the training data for the given labe
# The features are stored as 2 .npy files
def extract_features(sounds=esc50.all_sounds):
    if not os.path.isdir(settings.features_dir):
        os.mkdir(settings.features_dir)
    elif is_dir_empty(settings.features_dir):
        user_input = input('No training data present. Retrieve training data now? (y/n)')
        if user_input == 'y':
            esc50.retrieve_all_sounds()
        else:
            print('Cannot extract features without training data')
            return

    features_file, labels_file, features_extracted = get_feature_files()
    if features_extracted:
        print('Features have already been extracted. Exiting')
        return


    # Iterate through each sound and extract the features of every training
    # file for each sound
    num_features = 193
    features, labels = np.empty((0, num_features)), np.empty(0)
    for sound in sounds:
        dir, data_exists = esc50.get_training_dir(sound)
        if not data_exists:
            print(f'data does not exist for {sound}. Cannot extract features')
            continue

        for fn in os.listdir(dir):
            if fn.endswith(file_ext):
                file_path = dir + fn
                mfccs, chroma, mel, contrast,tonnetz = calculate_features_from_file(file_path)
                ext_features = np.hstack([mfccs,chroma,mel,contrast,tonnetz])
                features = np.vstack([features,ext_features])
                sound_index = get_label_index(sound)
                labels = np.append(labels, sound_index)

    features, labels = np.array(features), np.array(labels, dtype = np.int)
    np.save(features_file, features)
    np.save(labels_file, labels)


def calculate_features_from_file(file):

    X, sample_rate = sf.read(file, dtype='float32')

    if X.ndim > 1: X = X[:,0]
    X = X.T

    # short term fourier transform
    stft = np.abs(librosa.stft(X))

    # mfcc (mel-frequency cepstrum)
    mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T,axis=0)

    # chroma
    chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T,axis=0)

    # melspectrogram
    mel = np.mean(librosa.feature.melspectrogram(X, sr=sample_rate).T,axis=0)

    # spectral contrast
    contrast = np.mean(librosa.feature.spectral_contrast(S=stft, sr=sample_rate).T,axis=0)

    tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(X), sr=sample_rate).T,axis=0)
    return mfccs,chroma,mel,contrast,tonnetz


if __name__ == '__main__':
    extract_features()


    # from esc50 import all_labels
    #
    #
    #
    # # features, labels = feature_extraction.parse_audio_files('training_data', file_ext=file_ext)
    # features, labels = parse_audio(all_labels)
    # print(features.shape, labels.shape)
    #
    # # features, labels = feature_extraction.parse_audio_files(data_dir, file_ext=file_ext)
    # np.save('features/feat.npy', features)
    # np.save('features/labels.npy', labels)

    # Predict new
    # features, filenames = parse_predict_files('predictions')
    # feature_pfile, filename_pfile = file_utils.get_prediction_files(label)
    # np.save('predictions/feat.npy', features)
    # np.save('predictions/file_name', filenames)


'''I think below commented out code is for generating random test audio data'''
#     device_info = sd.query_devices(None, 'input')
#     sample_rate = int(device_info['default_samplerate'])
#     q = queue.Queue()
#     def callback(i,f,t,s): q.put(i.copy())
#     data = []
#     with sd.InputStream(samplerate=sample_rate, callback=callback):
#         while True:
#             if len(data) < 100000: data.extend(q.get())
#             else: break
#     X = np.array(data)
