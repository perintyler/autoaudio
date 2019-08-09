# from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display

from mpl_toolkits import mplot3d


# By default, librosa will resample the signal to 22050Hz.  You can
# change this behavior by saying: librosa.load(audio_path, sr=44100)
# audio_path = 'processed.wav'
#
def frequency_amplitude_graph(arr, duration):
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    # X data should be time
    # Y data is the different frequencies
    # Z data is the amplitude
    ax.scatter3D(xdata, ydata, zdata, c=zdata, cmap='Greens');
    return


def spectogram(audio_path, plot=True):
    y, sr = librosa.load(audio_path)

    # Convert to log scale (dB). We'll use the peak power (max) as reference.
    S = librosa.feature.melspectrogram(y, sr=sr, n_mels=128)
    log_S = librosa.power_to_db(S, ref=np.max)
    if plot:
        plt.figure(figsize=(12,4))
        librosa.display.specshow(log_S, sr=sr, x_axis='time', y_axis='mel')
        plt.title('mel power spectrogram')
        plt.colorbar(format='%+02.0f dB')

        plt.tight_layout()
        plt.show()
    return log_S


def show_chromagram(audio_path):
    y, sr = librosa.load(audio_path)
    y_harmonic, y_percussive = librosa.effects.hpss(y)
    # We'll use a CQT-based chromagram with 36 bins-per-octave in the CQT analysis.  An STFT-based implementation also exists in chroma_stft()
    # We'll use the harmonic component to avoid pollution from transients
    C = librosa.feature.chroma_cqt(y=y_harmonic, sr=sr, bins_per_octave=36)
    plt.figure(figsize=(12,4))
    librosa.display.specshow(C, sr=sr, x_axis='time', y_axis='chroma', vmin=0, vmax=1)

    plt.title('Chromagram')
    plt.colorbar()

    plt.tight_layout()
    plt.show()
