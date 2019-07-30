from pyAudioAnalysis import audioBasicIO
from pyAudioAnalysis import audioFeatureExtraction
import matplotlib.pyplot as plt
from convert import stereo_to_mono

# Step 1: Audio Feuture Extraction

# Fs is the audio file and x is the AIF audio info file
[Fs, x] = audioBasicIO.readAudioFile("testaudio/mono_test.wav");

# Set the Frame Size and Set to extract feutures from
frame_size = 0.050*Fs # 50 msecs
frame_step = 0.025*Fs # 25 msecs (25 msec frame overlap)

# extract feutures
F, f_names = audioFeatureExtraction.stFeatureExtraction(x, Fs, frame_size, frame_step)


# Step 2: Audio Feuture Classification
is_classifier_trained = False

if(is_classifier_trained == False):
    # train the model
    pass


# plt.subplot(2,1,1)
# plt.plot(F[0,:])
# plt.xlabel('Frame no')
# plt.ylabel(f_names[0])
# plt.subplot(2,1,2)
# plt.plot(F[1,:])
# plt.xlabel('Frame no')
# plt.ylabel(f_names[1])
# plt.show()
