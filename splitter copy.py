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

# from visualize import Point, LinearLine, VerticalLine, LinePlot, GraphItem


TRANSIENT_MAX = 5000 # Guessing here. half a second

class Sample:

    def __init__(self, t, amp):
        self.t = t
        self.amplitude = amp

    def tangent(self, other_sample):
        slope = (other_sample.amplitude - self.amplitude)/(other_sample.t - self.t)
        y_intercept = other_sample.amplitude - slope*other_sample.t
        return lambda time: slope*time + y_intercept

    @staticmethod
    def get_list(signal, sr):
        num_samples = len(signal)
        return [Sample(signal[i], sr*i) for i in range(num_samples)]

    @staticmethod
    def split(samples):
        t_vals = map(lambda s: s.t, samples)
        amp_vals = map(lambda s: s.amplitude, samples)
        return list(t_vals), list(amp_vals)

# Does every sound go through every phase
class Phase: # this should be called phase


    def __init__(self, sr, decaying, animate=False):
        self.sr = sr
        self.decaying = decaying
        self.hull = []
        self.max_index = -1
        self.animate = animate

    def growth_rate(self):
        return

    # https://www.geeksforgeeks.org/dynamic-convex-hull-adding-points-existing-convex-hull/
    def update_hull(self, new_point):
        def tangent(p0, p1):
            slope = (p1[1] - p0[1])/(p1[0] - p0[0])
            y_intercept = p1[1] - slope*p1[0]
            return lambda time: slope*time + y_intercept

        # Since points are fed in chronological order, point.x is always
        # going to be larger than the lastHullPoint.x. Therefore, each new
        # point is garunteed to be outside the hull, and must be added. The
        # role of this method is to remove hull points if adding the new
        # point requires so
        if self.hull_size() > 1:

            for hull_index in range(self.hull_size() - 1, 0, -1):
                hull_pnt = self.hull[hull_index]

                # Get the tangent line of the new point and the last hull point
                point_tangent = tangent(hull_pnt, new_point)
                # Get the tangent line of the last hull point and second to last hull point
                hull_tangent = tangent(self.hull[hull_index-1], hull_pnt)

                # Get the closest possible t value that is less than the hull
                # points t value. Since you can't subtract 1/inf to get infinitely
                # close value to t, I use the sample rate
                t_close = hull_pnt[0] - self.sr
                if point_tangent(t_close) < hull_tangent(t_close):
                    # hull_pnt must be deleted
                    del self.hull[hull_index]
                    if self.max_index == hull_index:
                        # This means the new point is the new y_max point
                        # and the old y_max is beind removed
                        self.max_index = -1

                    # if self.animate:
                        # Point.delete_at(hull_pnt[0], hull_pnt[1])
                    plt.plot(hull_pnt[0], hull_pnt[1], 'g.')
                        # for line in GraphItem.get_all(hull_pnt):
                        #     line.delete()
                else:
                    # hull point is valid. Therefore, all previous hull points
                    # must be valid too. Adding new point to hull will result
                    # in valid hull. Can break out of loop now
                    break


        hull_size = self.hull_size() # size may have changed from hp removal above
        if new_point[1] > self.max_amplitude():
            # new point is the new highest point. Set max index to last index
            # which will be the index of new point after being added
            self.max_index = hull_size

        # if self.animate:
        plt.plot(new_point[0], new_point[1], 'b.')#, markersize=0.5)
        # plot the new point if plot mode is on
        # if self.animate and hull_size != 0:
            # plot new hull point
            # p = Point(new_point[0], new_point[1], color='r', point_size=5).draw()
            # plt.plot(new_point[0], new_point[1], 'b.', markersize=0.5)
            # Next, plot a line from the last hull point to the new one
            # last_hull_point = self.hull[-1]
            # t0, t1 = last_hull_point[0], new_point[0]
            # hull_line = tangent(last_hull_point, new_point)
            # l = LinePlot.draw_function(hull_line, t0, t1, step_size=self.sr )
            # l.set_tag(new_point) # set tag to be deleted later if point is removed from hull

        # Finally, add the new point to the hull
        self.hull.append(new_point)



    def hull_size(self):
        return len(self.hull)

    def shift_detected(self):
        if self.decaying:
            return False
        else:
            hull_size = len(self.hull)
            if hull_size > 3 and self.max_index != hull_size-1: # TODO Should this really be 3?
                highest_point = self.hull[self.max_index]
                first_hull_point = self.hull[0]
                last_hull_point = self.hull[-1]

                gain_time = highest_point[0] - first_hull_point[0]
                decay_time = last_hull_point[0] - highest_point[0]
                if decay_time > gain_time:
                    return True
            return False

    def find_phase_shift(self, points):
        for pnt in points:
            self.update_hull(pnt)
            if self.shift_detected():
                self.hull = self.hull[:self.max_index+1]
                return self.hull[self.max_index]
        raise Exception('Could not find end of phase')

    def get_last_point(self):
        return self.hull[-1]

    def max_amplitude(self):
        if self.max_index == -1:
            return 0
        else:
            highest_point = self.hull[self.max_index]
            return highest_point[1]

    def plot(self):
        last_point = self.get_last_point()
        t = last_point[0]
        amp = last_point[1]
        plt.plot([t, t],[0,amp],'r--', lw=1)
        t_vals = [p[0] for p in self.hull]
        a_vals = [p[1] for p in self.hull]
        plt.plot(t_vals, a_vals, 'g--', lw=1)


def get_2d_signal(amplitudes, sr, plot=False):
    duration = sr * len(amplitudes)
    time_axis = np.arange(0., duration, sr)
    if plot:
        plt.plot(time_axis, amplitudes, color='black')
    return np.column_stack( (time_axis, amplitudes) )

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


# cross rates are really high after decay is finished.
# Cross rates and amplitude sums are high when new phase starts
# Find cycle start/end with num cross peak?
def energy(peaks, sr, duration):
    energy_snapshots = []
    frame_size = sr*50
    t = 0
    num_peaks = len(peaks)
    while t < duration:
        frame_start = t - frame_size/2
        frame_end = t + frame_size/2
        index = 0
        peaks_found = 0
        amplitude_sum = 0
        while index < num_peaks and peaks[index][0] < frame_end:
            if peaks[index][0] > frame_start:
                peaks_found += 1
                amplitude_sum += abs(peaks[index][1])
            index += 1
        snapshot = (peaks_found, amplitude_sum)
        energy_snapshots.append(snapshot)
        t += sr
    return energy_snapshots


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
    plt.ion()
    spf = wave.open(file_path)
    sr = spf.getframerate()
    amplitudes = np.fromstring(spf.readframes(-1), 'Int16')

    # tempo = np.array([float(a) for a in amplitudes])
    # tempo = librosa.feature.tempogram(y=tempo,sr=sr)
    # print(tempo)
    # eng = energy_entropy(amplitudes, sr)
    amplitudes = amplitudes[4150::]

    samples = Sample.get_list(amplitudes, sr)

    duration = len(amplitudes)*sr
    idk = find_peaks(amplitudes, sr)

    # HIGH AMOUNTS OF LOW AMPLITUDES CROSS PEAKS IN A FRAME INDICATES
    peaks = find_cross_peaks(amplitudes, sr)
    peaks = [(pnt[0], abs(pnt[1])) for pnt in peaks]
    #eng = energy(peaks, sr, duration)
    # Make all negative amps positive with absolute value
    amplitudes = np.absolute(amplitudes)
    signal = get_2d_signal(amplitudes, sr, plot=True)

    transient = Phase(sr, False)
    last_point = transient.find_phase_shift(peaks)
    transient.plot()

    decay_start_index = peaks.index(last_point)
    peaks = peaks[decay_start_index::]
    decay = Phase(sr, True, animate=True)

    for point in peaks:
        decay.update_hull(point)
        # input()
    t_axis, amp_axis = [], []
    for pnt in idk:
        t_axis.append(pnt[0])
        amp_axis.append(abs(pnt[1]))
    plt.plot(t_axis, amp_axis, 'r.', markersize=0.8)
    input()
    # cross_rates = []
    # amp_sums = []
    # for i in range(len(eng)):
    #     e = eng[i]
    #     cross_rates.append((i*sr,1000*e[0]))
    #     amp_sums.append((i*sr,e[1]))
    #
    # l0 = LinePlot(cross_rates, color='m').draw()
    # l1 = LinePlot(amp_sums, color='g').draw()

    # energy2d = [signal[0], [max(fr) for fr in eng] ]
    # for pnt in peaks:
    #     t_axis.append(pnt[0])
    #     amp_axis.append(abs(pnt[1]))
    # plt.plot(t_axis, amp_axis, 'r,')
