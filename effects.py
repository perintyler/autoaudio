from pysndfx import AudioEffectsChain
import datetime
# from chain import EffectsChain
import master_settings as m_settings
from isolate import isolate_speech
import eq_settings
import file_utils
from wpe import dereverberate
from noise_reduction import remove_background_noise

infile = 'testaudio/test.wav'

# STEP 1: compress and normalize for more effective speech isolation in step 2

normalize_fx = (
    AudioEffectsChain()
    .compand(**m_settings.compand)
    .normalize()
)


# perform audio effects on infile and save results to effected files
normalized_file = file_utils.get_test_outfile_path()
normalize_fx(infile, normalized_file, sample_out=48000, channels_out=1)


# STEP 3: Isolate speech and make non-speech frames silent
isolated_speech_file = isolate_speech(normalized_file)
file_utils.delete_file(normalized_file)


# STEP 4: eq the signal
eq_fx = (
    AudioEffectsChain()
    .highpass(**eq_settings.highpass)
    .lowshelf(**eq_settings.lowShelf)
    .equalizer(**eq_settings.lowFrequency)
    .equalizer(**eq_settings.midFrequency)
    .highshelf(**eq_settings.highShelf)
    # .reverb(**m_settings.reverb)
)
eq_file = file_utils.get_test_outfile_path()
eq_fx(isolated_speech_file, eq_file, sample_out=48000, channels_out=1)


# STEP 2: Remove background noise from signal
# noise_reduced_file = file_utils.get_test_outfile_path()
# remove_background_noise(isolated_speech_file, noise_reduced_file)
# file_utils.delete_file(isolated_speech_file)
