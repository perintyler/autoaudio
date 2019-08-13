
import numpy as np
import matplotlib.pyplot as plt
import librosa
# cross rates are really high after decay is finished.
# Cross rates and amplitude sums are high when new phase starts
# Find cycle start/end with num cross peak?
def energy_density(peaks, sr, duration):
    energy_snapshots = []
    num_points_per_frame = 2000
    frame_size = sr*num_points_per_frame
    t = 0
    num_peaks = len(peaks)
    while t < duration:
        frame_start = t #- frame_size/2
        frame_end = t + frame_size#/2
        index = 0
        peaks_found = 0
        amplitude_sum = 0
        while index < num_peaks and peaks[index].t < frame_end:
            if peaks[index].t > frame_start:
                peaks_found += 1
                amplitude_sum += abs(peaks[index].amplitude)
            index += 1
        snapshot = (peaks_found, amplitude_sum)
        e = (peaks_found) / num_points_per_frame
        # e = peaks_found / num_points_per_frame
        energy_snapshots.append((t, e))
        #energy_snapshots.append(snapshot)
        t += sr
    return energy_snapshots

def get_2d_signal(amplitudes, sr, plot=False):
    duration = sr * len(amplitudes)
    time_axis = np.arange(0., duration, sr)
    if plot:
        plt.plot(time_axis, amplitudes, color='black')
    return np.column_stack( (time_axis, amplitudes) )

def find_cross_peaks(amplitudes, sr, pos_only=False):
    num_points_per_frame = 2000
    frame_size = sr*num_points_per_frame
    density = {}
    buffer = []
    buffer_size = 0

    peaks = []
    cur_peak = amplitudes[0]
    cur_peak_t = 0
    positive = amplitudes[0] > 0
    for i in range(1, len(amplitudes)):
        t = sr*i
        if buffer_size != 0:
            while t - buffer[0] > frame_size:
                amp = buffer.pop(0)[1]
                buffer_size -= 1

        amp = amplitudes[i]
        pnt_is_positive =  amp > 0
        if positive != pnt_is_positive:
            peak = (t, cur_peak)
            peaks.append(peak)
            positive = pnt_is_positive
            cur_peak = amp if not pos_only else abs(amp)
            buffer.append(t)
            buffer_size+=1
        elif abs(amp) > abs(cur_peak):
            cur_peak = amp if not pos_only else abs(amp)
            buffer.append(t)
            buffer_size+=1
        density[t] = buffer_size
    return peaks, density

def find_change_density(amplitudes, sr, pos_only=False):
    density = {}
    buffer = []
    buffer_size = 0
    num_points_per_frame = 2000
    frame_size = sr*num_points_per_frame

    peaks = []
    for i in range(1, len(amplitudes) - 1):
        was_increasing = amplitudes[i] - amplitudes[i-1] > 0
        is_increasing = amplitudes[i+1] - amplitudes[i] > 0
        t = sr * i

        if buffer_size != 0:
            while t - buffer[0][0] > frame_size:
                amp = buffer.pop(0)[1]
                buffer_size -= 1

        if was_increasing != is_increasing:
            amp = amplitudes[i] if not pos_only else abs(amplitudes[i])
            inflection_point = (t, amp)
            peaks.append(inflection_point)

            buffer.append(inflection_point)
            buffer_size+=1

        if t > frame_size:
            density[t-frame_size] = buffer_size / num_points_per_frame
        #density[t] = buffer_size / num_points_per_frame
    return peaks, density

def find_phase_change_canidates():
    return

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
