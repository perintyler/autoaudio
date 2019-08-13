# Split an audio file into multiple audio file each consiting of an
# individual attack

from pydub import AudioSegment
import wave
import math
import numpy as np
# sound = AudioSegment.from_file(file_path)
import visualize
import scipy
from enum import Enum
from pyAudioAnalysis import audioFeatureExtraction
import librosa
import time

from features import *


# from visualize import Point, LinearLine, VerticalLine, LinePlot, GraphItem

class Signal:

    def __init__(self, amplitudes, sr):
        self.samples = []

        # compute density and cross peaks
        return

    def compute_features(self):
        y, sr = librosa.load(audio_path)
        S = librosa.feature.melspectrogram(y, sr=sr, n_mels=128)
        self.spectogram = librosa.power_to_db(S, ref=np.max)

    def splice(time):
        return

    def cross_peaks(self):
        return self.cross_peaks

    def density(self):
        return

    def spectogram(self):
        return


    def plot(self):
        return


TRANSIENT_MAX = 5000 # Guessing here. half a second

class Sample:

    def __init__(self, t, amp):
        self.t = t
        self.amplitude = amp

    def tangent(self, other_sample):
        slope = (other_sample.amplitude - self.amplitude)/(other_sample.t - self.t)
        y_intercept = other_sample.amplitude - slope*other_sample.t
        return lambda time: slope*time + y_intercept

    def __str__(self):
        return f'({self.t},{self.amplitude})'

    def __eq__(self, other):
        if isinstance(other, Sample):
            return self.t == other.t and self.amplitude == other.amplitude
        return False

    @staticmethod
    def get_list(amplitudes, sr):
        num_samples = len(amplitudes)
        return [Sample(amplitudes[i], sr*i) for i in range(num_samples)]

    @staticmethod
    def get_list_from_points(signal):
        samples = map(lambda p: Sample(p[0],p[1]), signal)
        return list(samples)

    @staticmethod
    def split(samples):
        t_vals = map(lambda s: s.t, samples)
        amp_vals = map(lambda s: s.amplitude, samples)
        return list(t_vals), list(amp_vals)

# may not need a class for this
class Profile:

    def __init__(self):
        self.density = []
        self.energy = []
        self.frequencies = []
        return

# Does every sound go through every phase
class Phase:

    def __init__(self, sr, decaying, animate=False, floor=20, energy=-1):
        self.sr = sr
        self.decaying = decaying
        self.hull = []
        self.max_index = -1
        self.animate = animate
        self.floor = floor
        self.energy_index = -1
        self.max_ed = -1

        if energy!=-1:
            self.ed = energy
            self.density_min = math.inf
            self.density_max = 0

    def growth_rate(self):
        return

    # Since points are fed in chronological order, point.x is always
    # going to be larger than the lastHullPoint.x. Therefore, each new
    # point is garunteed to be outside the hull, and must be added. The
    # role of this method is to remove hull points if adding the new
    # point requires so
    # https://www.geeksforgeeks.org/dynamic-convex-hull-adding-points-existing-convex-hull/
    def update_hull(self, sample, ya=True):
        if self.decaying and sample.t in self.ed:

            threshold = 0
            ed =  self.ed[sample.t]
            if self.max_ed == -1:
                self.max_ed == ed

            # lookahead_index = sample.t + 100*self.sr
            if sample.t >= self.sr:
                ded = abs(self.ed[sample.t+self.sr] - ed)

                #plt.plot(sample.t,10000*ded, 'r.')#,markersize=0.3)
                plt.plot(sample.t,10000*ed, 'm.', markersize=0.7)
            return
                #density_range = self.density_max - self.density_min
                #print(sample.t, density_range)
                # if ded > density_range and ded > threshold:
                    # plt.plot([sample.t, sample.t],[0,8000],'r--', lw=1)
                    # self.density_max, self.density_min = 0,math.inf
            # if ed > self.density_max:
            #     self.density_max = ed
            # elif ed < self.density_min:
            #     self.density_min = ed
            # ed0 = self.energy[sample.t - self.sr]
            # ed1 = self.energy[sample.t + self.sr]
            # print(sample.t, sample.t / self.sr, ed)
            # input()
            # plt.plot(sample.t,10*ed, 'r.')#,markersize=0.3)
            # print(self.spectogram[sample.t])
            #print(self.energy[i][1])

        hull_size = self.hull_size()
        if hull_size == 0 and sample.amplitude < self.floor:
            return

        if hull_size > 1:

            for hull_index in range(self.hull_size() - 1, 0, -1):
                hull_pnt = self.hull[hull_index]

                point_tangent = hull_pnt.tangent(sample)
                hull_tangent = self.hull[hull_index-1].tangent(hull_pnt)
                # # Get the tangent line of the new point and the last hull point
                # point_tangent = tangent(hull_pnt, new_point)
                # # Get the tangent line of the last hull point and second to last hull point
                # hull_tangent = tangent(self.hull[hull_index-1], hull_pnt)

                # Get the closest possible t value that is less than the hull
                # points t value. Since you can't subtract 1/inf to get infinitely
                # close value to t, I use the sample rate
                # t_close = hull_pnt[0] - self.sr
                t_close = hull_pnt.t - self.sr

                if point_tangent(t_close) < hull_tangent(t_close):
                    # hull_pnt must be deleted
                    del self.hull[hull_index]
                    if self.max_index == hull_index:
                        # This means the new point is the new y_max point
                        # and the old y_max is beind removed
                        self.max_index = -1

                    # if self.animate:
                        # Point.delete_at(hull_pnt[0], hull_pnt[1])

                    if self.animate: plt.plot(hull_pnt.t, hull_pnt.amplitude, 'g.')
                        # for line in GraphItem.get_all(hull_pnt):
                        #     line.delete()
                else:
                    # hull point is valid. Therefore, all previous hull points
                    # must be valid too. Adding new point to hull will result
                    # in valid hull. Can break out of loop now
                    break


        hull_size = self.hull_size() # size may have changed from hp removal above
        if sample.amplitude > self.max_amplitude():
            # new point is the new highest point. Set max index to last index
            # which will be the index of new point after being added
            self.max_index = hull_size

        if self.animate: plt.plot(sample.t, sample.amplitude, 'b.')#, markersize=0.5)

        # Finally, add the new point to the hull
        self.hull.append(sample)



    def hull_size(self):
        return len(self.hull)

    def duration(self):
        return self.hull[-1].t - self.hull[0].t

    def shift_detected(self):
        if self.decaying:
            # If abrupt change in energy, phase is probably over
            return False
        else:
            min_attack_time = 50000000
            hull_size = len(self.hull)
            if hull_size > 3 and self.max_index != hull_size-1: # TODO Should this really be 3?
                highest_point = self.hull[self.max_index]
                first_hull_point = self.hull[0]
                last_hull_point = self.hull[-1]

                gain_time = highest_point.t - first_hull_point.t
                decay_time = last_hull_point.t - highest_point.t
                if gain_time > min_attack_time and decay_time > gain_time:
                    print('*', gain_time, decay_time)
                    return True
            return False

    def find(self, samples):
        for s in samples:
            self.update_hull(s)
            if self.shift_detected():
                if not self.decaying:
                    self.hull = self.hull[:self.max_index+1]
                return self.hull[0], self.hull[-1]

        raise Exception('Could not find end of phase')

    def get_last_sample(self):
        return self.hull[-1]

    def max_amplitude(self):
        if self.max_index == -1:
            return 0
        else:
            highest_point = self.hull[self.max_index]
            return highest_point.amplitude

    def plot(self):
        last_sample = self.get_last_sample()
        t = last_sample.t
        amp = last_sample.amplitude
        plt.plot([t, t],[0,amp],'r--', lw=1)
        t_vals, amp_vals = Sample.split(self.hull)
        # t_vals = [p[0] for p in self.hull]
        # a_vals = [p[1] for p in self.hull]
        plt.plot(t_vals, amp_vals, 'm--', lw=1)
        plt.plot(t_vals, amp_vals, 'b.')





if __name__ == '__main__':


    # file_path = 'testaudio/mono_test.wav'
    # file_path = 'training_data/breathing/breathing_0.wav'
    file_path = 'testaudio/trimmed_b.wav'

    timer0 = time.time()
    y, sr = librosa.load(file_path)
    S = librosa.feature.melspectrogram(y, sr=sr, n_mels=128)
    spectogram = librosa.power_to_db(S, ref=np.max)
    print(f'took {time.time() - timer0} to get spectogram')

    # specto = visualize.spectogram(file_path, plot=False)
    plt.ion()

    timer0 = time.time()
    spf = wave.open(file_path)
    sr = spf.getframerate()
    amplitudes = np.fromstring(spf.readframes(-1), 'Int16')
    print(f'took {time.time() - timer0} to read wave into array')
    # tempo = np.array([float(a) for a in amplitudes])
    # tempo = librosa.feature.tempogram(y=tempo,sr=sr)
    # print(tempo)
    # eng = energy_entropy(amplitudes, sr)
    # amplitudes = amplitudes[4150::]

    #samples = Sample.get_list(amplitudes, sr)
    duration = len(amplitudes)*sr




    timer0 = time.time()
    inflections, density = find_change_density(amplitudes, sr, pos_only=True)

    # inflections, density = find_change_density(amplitudes, sr, pos_only=True)
    # for i in range(10000):
    #     a = density[i*sr]
    #     if a != 0:
    #         print(i*sr, a)
    print(f'took {time.time() - timer0} to get density')
    timer0 = time.time()
    peaks = find_cross_peaks(amplitudes, sr, pos_only=True)
    print(f'took {time.time() - timer0} to get cross peaks')


    samples = Sample.get_list_from_points(peaks)
    # # timer0 = time.time()
    # energy = energy_density(samples, sr, duration)
    # print(f'took {time.time() - timer0} to get energy')
    # energy_lookup = {e[0]:e[1] for e in energy }

    amplitudes = np.absolute(amplitudes)
    signal = get_2d_signal(amplitudes, sr, plot=True)

    timer0 = time.time()
    transient = Phase(sr, False, energy=density)
    first_sample, last_sample = transient.find(samples)
    print(f'took {time.time() - timer0} to find attack')
    transient.plot()

    # t_axis, amp_axis = Sample.split(Sample.get_list_from_points(density))
    # plt.plot(t_axis, amp_axis, 'r.', markersize=0.8)
    timer0 = time.time()
    decay_start_index = samples.index(last_sample)
    # samples = Sample.get_list_from_points(inflections)
    # decay_start_index = 0
    # while samples[decay_start_index].t != last_sample.t: decay_start_index+=1

    #decay_start_index = samples.index(last_sample)
    samples = samples[decay_start_index::]

    print(f'took {time.time() - timer0} to splice samples')
    timer0 = time.time()

    decay = Phase(sr, True, animate=False, energy=density)

    for s in samples:
        #print('adding', s)
        decay.update_hull(s, ya=True)
        #if i % 5000 == 0:


    print(f'took {time.time() - timer0} to find decay')
    input()


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
