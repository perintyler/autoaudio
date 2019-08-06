# Split an audio file into multiple audio file each consiting of an
# individual attack

from pydub import AudioSegment
import wave
import numpy as np
import matplotlib.pyplot as plt
# sound = AudioSegment.from_file(file_path)
import visualize
import scipy

# https://en.wikipedia.org/wiki/Envelope_(music)
# 
# class Phase(Enum):
#     ATTACK = 0
#     DECAY = 1
#     SUSTAIN = 2
#     RELEASE = 3
#
# # Does every sound go through every phase
#
# class ConvexHull:
#
#     def __init__(self, p0):
#         self.phase = Phase.ATTACK
#         self.y_min, self.y_max = p0, p0
#         self.upper = [p0]
#         self.lower = [p0]
#         self.phase_markers = []
#
#     def constrains(self, y_val):
#         return self.y_min < y_val and self.y_max > y_val
#
#     # since after an attack the wave is always decaying, removing the y_max
#     # means a new phase has begun
#     def add_points(self, points):
#         y_min, y_max = points.max(), points.min()
#         if self.phase == Phase.ATTACK:
#             upper_hull_change = self.upper[-1][1] - self.upper[-2][1]
#             lower_hull_change = self.lower[-1][1] - self.lower[-2][1]
#             if upper_hull_change < 0 and lower_hull_change > 0:
#                 attack_end = min(self.upper[-1][0], self.lower[-1][0])
#                 self.phase_markers.append(attack_end)
#             self.phase = Phase.DECAY
#         elif self.phase == Phase.DECAY:
#             # while its decaying, if you have to remove a point from the hull
#             # then a new phase has started
#             return
#         else: # decay
#             return
#         # elif self.phase = SoundPhase.DECAY:
#         #     return
#         # elif self.phase = SoundPhase.SUSTAIN:
#         #     return
#         # else: # DECAY
#         #     return
#
#
#     class Result(Enum):
#         INSIDE = 0
#         ON = 1
#         Y_
#     def add_point():
#
#
#
#     def is_point_inside(self, pnt):
#         return
#
#     def rate_of_change(self):
#         upper_hull_change = self.upper[-1] - self.upper[-2]
#         lower_hull_change = self.lower[-1] - self.lower[-2]
#
#
# class SignalStream:
#
#     def __init__(self, sampe_rate):
#         self.decaying = False
#         self.attacks = []
#         self.previous_hulls = []
#         self.hull = ConvexHull()
#         self.upper_hull = []
#         self.lower_hull = []
#         return
#
#     def add_frame(self, f):
#         attack_found = False
#         for x,y in f.points():
#             if not self.hull.constrains(y):
#                 self.previous_hulls.append(self.hull)
#                 self.hull = ConvexHull()
#                 attack_found = True
#                 attack_start = x-1
#
#         if not attack_found:
#             self.hull.add_points(f.points())
#             upper_decay, lower_decay = self.hull.get_y_change()
#
#
#         else:
#
#
#         attack_found = self.dosomething()
#         if attack_found:
#             self.decaying = False
#
#         return
#
#     def add_point_to_hull(self, amplitude, upper):
#         pnt = self.add_sample(amplitude)
#         # if pnt is in hull do nothing
#         if not self.is_point_in_hull(pnt):
#             # find upper tangent
#             hull = self.upper_hull if upper else self.lower_hull
#             while(pnt.tangent(neighbor).crosses_inside(hull)):
#                 self.upper_hull.remove(neighbor)
#                 neighbor = self.upper_hull.last
#             neighbor = self.lower_hull.last
#             while(pnt.tangent(neighbor).crosses_inside(hull)):
#                 self.lower_hull.remove(neighbor)
#                 neighbor = self.lower_hull.last
#
#     def is_point_in_hull(self, pnt):
#         #
#         return
#
#     def get_baseline_amplitude(self):
#         return
#
#     def get_attack_points(self):
#         return self.attacks
#
#
#
# def isolate_notes():
#
#     return
#
# def find_next_attack(amplitudes, start_index, end_index, hull):
#     frame_size = 10
#     for i in range(start_index, end_index, frame_size):
#         frame = amplitudes[i:frame_size]
#
#     return




def get_2d_signal(amplitudes, sr, plot=False):
    duration = sr * len(amplitudes)
    time_axis = np.arange(0., duration, sr)
    if plot:
        plt.plot(time_axis, amplitudes)
    return np.column_stack( (time_axis, amplitudes) )


def get_outline(signal, plot=False):
    hull = scipy.spatial.ConvexHull(signal)
    if plot:
        # add the first point to the end of vertices to close the outline
        outline = np.append(hull.vertices, hull.vertices[0])
        plt.plot(signal[outline,0], signal[outline,1], 'r--', lw=1)
    # x = list(map(lambda simplex: signal[simplex, 0], hull.simplices))
    # y = list(map(lambda simplex: signal[simplex, 1], hull.simplices))
    return signal[hull.vertices,0], signal[hull.vertices,1]


# Find repetitive cycle first start and end
def find_cycles(signal, sr):
    return

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


if __name__ == '__main__':


    file_path = 'testaudio/trimmed2.wav' #'training_data/breathing/breathing_0.wav
    spf = wave.open(file_path)
    sr = spf.getframerate()
    amplitudes = np.fromstring(spf.readframes(-1), 'Int16')

    # Make all negative amps positive with absolute value
    amplitude = np.absolute(amplitude)
    signal = get_2d_signal(amplitudes, sr, plot=True)

    x, y = get_outline(signal, plot=True)
    plt.show()
    # print(outline)
    #
    # plt.plot(x, y)

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
