# Split an audio file into multiple audio file each consiting of an
# individual attack

from pydub import AudioSegment
import wave
import numpy as np
import matplotlib.pyplot as plt
# sound = AudioSegment.from_file(file_path)
import visualize
import scipy
from enum import Enum
from pyAudioAnalysis import audioFeatureExtraction
import librosa



TRANSIENT_MAX = 5000 # Guessing here. half a second


# # Reversed sound fucks this all up
# class Envelope:
#     def __init__(self):
#         self.phases = 0
#
#
# # https://en.wikipedia.org/wiki/Envelope_(music)
# class Phase(Enum):
#     ATTACK = 0
#     DECAY = 1
#     SUSTAIN = 2
#     RELEASE = 3
#


# # end of attack happens at lowest added new hull point in decay phase
# class SignalStream:
#
#     def __init__(self, sampe_rate):
#         self.decaying = False
#         self.attacks = []
#         self.previous_hulls = []
#         self.hull = ConvexHull()
#         self.upper_hull = []
#         self.lower_hull = []
#         self.baseline = 0
#
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
#             return
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
# # Match and peaks are for finding when a wave pattern repeats. Even though
# # overall amplitudes decay, the slope at every point should not change
# # Repeated patterns can be used to find decay time
# # This shit won't work
# def match(changes, amplitudes):
#     matching = False
#     match_found = False
#     match_start = 0
#     index = 0
#     num_points = len(amplitudes)
#     num_changes = len(changes)
#     for index in range(num_points - num_changes):
#         difference = point[index+1] - point[index]
#         if difference == changes[0]:
#             match_start = index
#             matching = True
#             for match_index in range(match_start, num_changes):
#                 difference = point[match_index+1] - point[match_index]
#                 if difference == changes[match_index - start_index]:
#                     matching = False
#                     break
#             if matching: return match_start
#             else: continue
#     return -1
#
# def find_peaks():
#     return
#
# def match_peaks():
#     return
#
# def get_peak_derivative(peaks):
#     changes = []
#     for index in range(0, len(peaks)):
#         if index == len(peaks): difference = peaks[0] - peaks[i]
#         else: difference = peaks[i+1] - peaks[i]
#         changes.append(difference)
#     return changes
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


# Does every sound go through every phase
class Phase: # this should be called phase

    def __init__(self, sr, decaying):
        self.sr = sr
        self.decaying = decaying
        self.hull = []
        self.max_index = 0

    def test(self):
        if len(self.hull) > 2:
            m = 0
            max_index = 0
            for i in range(len(self.hull)):
                p = self.hull[i]
                if(p[1] > m):
                    m = p[1]
                    max_index = i
            print('d0', self.hull[max_index][0] - self.hull[max_index-1][0])
            print('el', self.hull[len(self.hull)-1][0] - self.hull[max_index][0])
            # if len(self.hull) >= 2:
            #     m = 0
            #     max_index = 0
            #     for i in range(len(self.hull)):
            #         p = self.hull[i]
            #         if(p[1] > m):
            #             m = p[1]
            #             max_index = i
            #     pre_max = self.hull[max_index][0] - self.hull[max_index-1][0]
            #     post_max = self.hull[len(self.hull)-1][0] - self.hull[max_index][0]
            #     threshold = 4200000
            #     if post_max > pre_max and pre_max > threshold:
            #         print(pre_max, post_max)
            #         self.decaying = True
            # self.test()

            # if len(self.hull) >= 2:
            #     increasing = self.hull[-1][1] > self.hull[-2][1]
            #     if len(self.hull) == 2:
            #         a = 'increasing' if increasing else 'decreasing'
            #         print(a + ' for 1 iteration. delta t = ' + str((self.hull[-1][0] > self.hull[-2][0])/self.sr) )
            #     else:
            #         i = 3
            #         total_change = 0
            #         while i <= len(self.hull):
            #             dif = self.hull[-i + 1][1] - self.hull[-i][1]
            #             if increasing and dif < 0 or not increasing and dif > 0:
            #                 break
            #             i+=1
            #         dt = self.hull[-1][0] - self.hull[-i+1][0]
            #         if increasing:
            #             print('increasing for ' + str(i-2) + ' iterations. delta t = ' + str(dt/self.sr))
            #         #ya = "increasing"  if increasing else "decreasing"
            #         else:
            #
            #             print('decreasing for ' + str(i-2) + ' iterations. delta t = ' + str(dt/self.sr))



    # https://www.geeksforgeeks.org/dynamic-convex-hull-adding-points-existing-convex-hull/
    def update_hull(self, new_point):
        # Since points are fed in chronological order, point.x is always
        # going to be larger than the lastHullPoint.x. Therefore, each new
        # point is garunteed to be outside the hull, and must be added. The
        # role of this method is to remove hull points if adding the new
        # point requires so
        if len(self.hull) < 2:
            self.hull.append(new_point)
            self.y_max = new_point
            # if len(self.hull) == 2:
            #     increasing = 'increasing' if self.hull[-1] > self.hull[-2] else 'decreasing'
            #     print(increasing + ' for 1 iteration. delta t = ' + str((self.hull[-1][1] - self.hull[-2][1])/self.sr))
            return []



        def tangent(p0, p1):
            slope = (p1[1] - p0[1])/(p1[0] - p0[0])
            return lambda time: slope*time + p0[1]

        removed = []
        for hull_index in range(len(self.hull) - 1, 1, -1):
            hull_pnt = self.hull[hull_index]

            # Get the tangent line of the new point and the last hull point
            point_tangent = tangent(hull_pnt, new_point)
            # Get the tangent line of the last hull point and second to last hull point
            hull_tangent = tangent(hull_pnt, self.hull[hull_index-1])

            # Get the closest possible t value that is less than the hull
            # points t value. Since you can't subtract 1/inf to get infinitely
            # close value to t, I use the sample rate
            t_close = hull_pnt[0] - self.sr

            # if the points tangent line encroaches inside the hull, the hull
            # point must be removed. Since only an upper hull is computed,
            # there will be encroachment whenever point_tangent(hpT - tS)
            # is greater than hull_tangent(hpT - tS) where hpT is the t value
            # of the hull point and tS is any very close t value that is less
            # than hpT. in this case hpT - sample rate
            if point_tangent(t_close) > hull_tangent(t_close):
                # hull_pnt must be deleted
                del self.hull[hull_index]
                removed.append(hull_pnt)
            else:
                # hull point is valid. Therefore, all previous hull points
                # must be valid too. Adding new point to hull will result
                # in valid hull. Can break out of loop now
                break

        # Finally, add the new point to the hull
        self.hull.append(new_point)
        # update y_max if new_point is new max
        if new_point[1] > self.y_max[1]:
            self.y_max = new_point
            self.max_index = len(self.hull) - 1

    def shift_detected(self):
        if len(self.hull) > 3:
            max_t = self.y_max[0]
            first_peak_t = self.hull[1][0]
            if self.hull[-1][0] != max_t:
                d0 = max_t - first_peak_t
                d1 = self.hull[-1][0] - max_t
                if d1 > d0:
                    return True
        return False

    def get_end(self, points):
        for pnt in points:
            self.update_hull(pnt)
            if self.shift_detected():
                self.hull = self.hull(0::self.max_index)
                while self.hull[-1][0] != self.y_max[0]:
                    print('deleting', self.hull[-1])
                    print('y_max', self.y_max)
                    del self.hull[-1]
                    return self.y_max
        # No phase end found. Return last point
        return self.hull[-1]


    def hull_distances(self):
        if len(self.hull) == 0: return
        distances = []
        max_index = 0
        for i in range(len(self.hull)-1):
            if(self.hull[i][0] == self.y_max[0]):
                max_index = i
            distances.append(int((self.hull[i+1][0] - self.hull[i][0])/self.sr))
        print(distances, max_index)

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
        plt.plot(signal[outline,0], signal[outline,1], 'g.')
    # x = list(map(lambda simplex: signal[simplex, 0], hull.simplices))
    # y = list(map(lambda simplex: signal[simplex, 1], hull.simplices))
    print(signal[hull.vertices,1])
    return signal[hull.vertices,0], signal[hull.vertices,1]


def find_cross_peaks(amplitudes, sr):
    peaks = []
    cur_peak = amplitudes[0]
    cur_peak_t = 0
    positive = amplitudes[0] > 0
    for i in range(1, len(amplitudes)):
        amp = amplitudes[i]
        pnt_is_positive =  amp > 0
        if positive != pnt_is_positive:
            peak = (cur_peak_t, cur_peak)
            peaks.append(peak)
            positive = pnt_is_positive
            cur_peak = amp
            cur_peak_t = sr*i
        elif abs(amp) > abs(cur_peak):
            cur_peak_t = sr * i
            cur_peak = amp
    return peaks


def find_peaks(amplitudes, sr):
    peaks = []
    for i in range(1, len(amplitudes) - 1):
        was_increasing = amplitudes[i] - amplitudes[i-1] > 0
        is_increasing = amplitudes[i+1] - amplitudes[i] > 0
        if was_increasing != is_increasing:
            t = sr * i
            inflection_point = (t, amplitudes[i])
            peaks.append(inflection_point)
    return peaks

def cut_file(file, seconds): # seconds can be a double
    cut_point = seconds * 1000
    sound = AudioSegment.from_file(file)

    first = sound[:cut_point]

    # create a new file "first_half.mp3":
    first.export("testaudio/split.wav", format="wav")


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

def energy_entropy(amplitudes, sr):
    features, f_names = audioFeatureExtraction.stFeatureExtraction(amplitudes, sr, 0.002*sr, 0.001*sr);
    return features[f_names.index('energy_entropy')]


if __name__ == '__main__':


    # file_path = 'training_data/breathing/breathing_0.wav'
    file_path = 'testaudio/trimmed_b.wav'
    # specto = visualize.spectogram(file_path, plot=False)
    # plt.ion()
    spf = wave.open(file_path)
    sr = spf.getframerate()
    amplitudes = np.fromstring(spf.readframes(-1), 'Int16')

    # tempo = np.array([float(a) for a in amplitudes])
    # tempo = librosa.feature.tempogram(y=tempo,sr=sr)
    # print(tempo)
    # eng = energy_entropy(amplitudes, sr)
    amplitudes = amplitudes[4150::]

    # HIGH AMOUNTS OF LOW AMPLITUDES CROSS PEAKS IN A FRAME INDICATES
    # NEW ATTACK
    peaks = find_cross_peaks(amplitudes, sr)
    peaks = [(pnt[0], abs(pnt[1])) for pnt in peaks]

    # # Make all negative amps positive with absolute value
    amplitudes = np.absolute(amplitudes)
    # peaks = find_peaks(amplitudes, sr)

    signal = get_2d_signal(amplitudes, sr, plot=True)
    # energy2d = [signal[0], [max(fr) for fr in eng] ]
    t_axis = []
    amp_axis = []
    # for pnt in peaks:
    #     t_axis.append(pnt[0])
    #     amp_axis.append(abs(pnt[1]))
    # plt.plot(t_axis, amp_axis, 'r,')

    transient = Phase(sr, False)
    shift = transient.get_end(peaks)
    for pnt in transient.hull:
        plt.plot(pnt[0], pnt[1], 'g.')
    plt.axvline(x=shift[0], color='r')
    plt.show()
    #decay = Phase(sr, True)



    # x = []
    # y = []
    # for pnt in phase.hull:
    #     x.append(pnt[0])
    #     y.append(pnt[1])
        # plt.plot(pnt[0], pnt[1], 'r.')
        # input('do something man')

    # plt.plot(x, y, 'r--', lw = 1)
    plt.show()
    #
    # x, y = get_outline(signal, plot=True)
    # plt.show()
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
