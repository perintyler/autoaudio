from pydub import AudioSegment
import datetime
import os
from shutil import copyfile


def stereo_to_mono(infile, file_type='wav'):
    sound = AudioSegment.from_file(infile, format=file_type)
    sound = sound.set_channels(1)
    # remove filetype, ammend name, add on file type
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
