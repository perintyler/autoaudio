
import librosa
import numpy as np
import matplotlib.pyplot as plt


file_path = 'testaudio/trimmed_b.wav'
amplitudes, sr = librosa.load(file_path)

# Proccessing low frequencies should be different than for high frequencies


spectogram = librosa.feature.melspectrogram(amplitudes, sr=sr, n_mels=512, hop_length=1)
# convert power to db, which is a log scale
#spectogram = librosa.power_to_db(power_spectogram, ref=np.max)

plt.ion()
starting_hz = 20
num_hz = 20
step = 50
for hz in range(starting_hz, starting_hz + num_hz*step, step):
    freq = spectogram[:,hz]
    x_step = len(amplitudes) * sr / len(freq)
    # frequencies get redder the higher they are
    red_value = (hz - starting_hz)/(step*num_hz)
    color = (red_value, red_value, 0.5)
    x = [i*x_step for i in range(0,len(freq))]
    #plt.figure(hz/25 - 50)
    #plt.title(f'{hz} hz')
    plt.plot(x, freq, c=color) #, c=np.random.rand(3,))
    input(hz)
input()
