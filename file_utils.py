from pydub import AudioSegment
import datetime
import os
from shutil import copyfile
import settings



def stereo_to_mono(infile, outfile=None, file_type='wav'):
    sound = AudioSegment.from_file(infile, format=file_type)
    sound = sound.set_channels(1)
    # remove filetype, ammend name, add on file type
    if outfile is None:
        outfile = infile[:(len(file_type) + 1)] + '-mono.' + file_type
    sound.export(outfile, format=file_type)
    return outfile

def get_test_outfile_path():
    timestamp = str(datetime.datetime.now()).replace(' ','_')
    return f'processed/{timestamp}.wav'


def delete_file(path):
    os.remove(path)

def copy_to(src, dst):
    copyfile(src, dst)

# Returns the file location of 2 numpy file (feat.npy and label.npy) if
# features have been extracted for the given labels. If feature extraction
# hasn't occured, None, None tuple is returned.
def get_feature_files(label):
    label_feat_dir = f'{settings.feature_dir}/{label}'
    if(not os.path.isdir(label_feat_dir)):
        os.mkdir(label_feat_dir)
    feature_file = f'{label_feat_dir}/feat.npy'
    label_file = f'{label_feat_dir}/label.npy'
    features_exist = os.path.isfile(feature_file) and os.path.isfile(label_file)
    return feature_file, label_file, features_exist

model_dir = 'models'
training_data_dir = 'training_data'
# Returns the location of the tensorflow audio classification model file (.h5)
# is a model for the given label has been trained. Otherwise, None is returned
def get_classification_model(label):
    model_file = f'{settings.classification_dir}/{label}.h5'
    file_exists = os.path.isfile(model_file)
    return model_file, file_exists

def get_prediction_files(label):
    label_dir = f'{settings.prediction_dir}/{label}'
    if(not os.path.isdir(label_dir)):
        os.mkdir(label_dir)
    feat_pfile = f'{label_dir}/feat.npy'
    filename_pfile = f'{label_dir}/filenames.npy'
    return feat_pfile, filename_pfile

def get_file_count(dir):
    return len(os.listdir(dir))

def is_dir_empty(dir):
    return get_file_count(dir) == 0


def get_label_index(label):
    # index labels by sorted location in directory
    subdirs = os.listdir(settings.training_data_dir)
    subdirs.sort()
    return subdirs.index(label)

def get_training_data_dir(label):
    dir = f'{settings.training_data_dir}/{label}/'
    # Check to see if the directory exists and is not empty
    training_data_exists = os.path.isdir(dir) and not is_dir_empty(dir)
    return dir, training_data_exists


def num_training_files(label):
    dir, data_exists = get_training_data_dir(label)
    return len([fn for fn in os.listdir(dir) if not fn.endswith('.asd')])



def cut_file(file, seconds): # seconds can be a double
    cut_point = seconds * 1000
    sound = AudioSegment.from_file(file)

    first = sound[:cut_point]

    # create a new file "first_half.mp3":
    first.export("testaudio/split.wav", format="wav")
