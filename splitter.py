# Split an audio file into multiple audio file each consiting of an
# individual attack

from pydub import AudioSegment
import wave
import numpy as np
import matplotlib.pyplot as plt
# sound = AudioSegment.from_file(file_path)
import visualize



def find_inflections(signal, sr):
    last_amplitude = signal.pop(0)
    inflections = { 0: last_amplitude}  # t -> amplitude
    increasing = signal[0] > last_amplitude
    t = sr # because first value already popped

    for amplitude in signal:
        if increasing and last_amplitude > amplitude:
            inflections[t] = amplitude
            increasing = False
        elif not increasing and last_amplitude < amplitude:
            inflections[t] = amplitude
            increasing = True
        t += sr

    return {k: v for k, v in inflections.items() if v > 0}



file_path = 'testaudio/split.wav' #'training_data/breathing/breathing_0.wav'
spf = wave.open(file_path, 'r')
sr = spf.getframerate()
signal = spf.readframes(-1)
signal = np.fromstring(signal, 'Int16')

# inflections = find_inflections(signal.tolist(), sr)
#
# plt.plot(list(inflections.keys()), list(inflections.values()))
# plt.show()



# signal = np.absolute(signal)
#
# plt.figure(1)
# plt.title('Signal Wave...')
# plt.plot(signal, 'k')
# plt.show()

# spectogram = visualize.spectogram(file_path, show=True)
# freq_arr = np.array(spectogram)
# print(freq_arr.shape)

def cut_file(file, seconds): # seconds can be a double
    cut_point = seconds * 1000
    sound = AudioSegment.from_file(file)

    first = sound[:cut_point]

    # create a new file "first_half.mp3":
    first.export("testaudio/split.wav", format="wav")

def get_min_amplitude():
    return 0

def get_max_amplitude():
    return 0

# The attack starts when amplitude starts increasing. Once a decrease of
# amplitude starts, the file should end whenever the next increase happens
# there should be a slope threshold that has to be reached to institue
# amplitude increase. There should be a min file time.
def find_attacks():
    return []




# Finds frequency segment that repeats (can lower in amplitude)
def find_repeated_segment():
    return
    # split sound1 in 5-second slices: slices = sound1[::5000]
# If amplitude does not reach 0 (or baseline amplitude before attack) after a
# attack starts, then it is not a complete 'note'


# It would be cool to be able to tell if an audio clip contains isolate 'notes'



# Using attack and the initial amplitude slope decrease, an amplitude regression
# function should be able to be made which I'm pretty sure is logarthmic
# If the full sound of every attack of every audio source can be predicted,
# it should be possible to seperate all sound sources into seperate audio files
class Frame:
    def __init__(self):
        return


class Note:
    def __init__(self):
        self.startTime = 'tbt'

    def find_end(self, file):
        return

# A 'note' has an attack, a hold time, and then a slow death time
# The hold time will probably be the hardest caviat in this idea because
# the attack should be a smooth (exponential? / inverse of decay log func?) and
# the decay should be a smoth log function. Holding notes can change un frequency/amplitude
