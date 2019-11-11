# from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display

from mpl_toolkits import mplot3d

items = []
lines = []
points = []

class GraphItem:

    def __init__(self, points, color, point_type, plt_params={}):
        self.plt_arg = f'{color}{point_type}'
        self.plt_params = plt_params
        self.points = points
        self.drawn = []
        items.append(self)

    def is_drawn(self):
        return len(self.drawn) != 0

    def draw(self):
    #    args = self.plt_arg if not self.plt_params else **self.plt_params
        #drawn_objects = plt.plot(self.points[0], self.points[1], self.plt_arg, **self.plt_params)
        # if not self.plt_params:
        #     drawn_objects = plt.plot(self.points[0], self.points[1], **self.plt_params)
        # else:
        #     drawn_objects = plt.plot(self.points[0], self.points[1], self.plt_arg)
        drawn_objects = plt.plot(self.points[0], self.points[1], self.plt_arg, **self.plt_params)

        self.drawn.extend(drawn_objects)
        # for point in self.points:
        #     x, y = point[0], point[1]
        #     plt_obj, = plt.plot(x, y, self.plt_arg, **self.plt_params)
        #     self.drawn.append(plt_obj)
        return self

    def delete(self):
        for plt_obj in self.drawn:
            plt_obj.remove()
        self.drawn = []

    def set_tag(self, tag):
        self.tag = tag if isinstance(tag, str) else str(tag)

    def get_tag(self):
        if not hasattr(self, 'tag'): return '*'
        else: return self.tag

    @staticmethod
    def get_all(tag):
        tag_str = tag if isinstance(tag, str) else str(tag)
        tagged = []
        for item in GraphItem.all:
            if item.get_tag() == tag:
                tagged.append(tag)
        return tagged


class Point(GraphItem):

    def __init__(self, x, y, color, point_size=1):
        self.item_type = 'point'
        points = [(x,y)]
        point_type = '.'
        params = {'markersize': point_size}
        super().__init__(points, color, point_type, plt_params=params)
        points.append(self)

    @staticmethod
    def delete_at(x, y):
        Point.all.pop((x,y)).delete()


# TODO line plot and linear line should be the same class
class LinePlot(GraphItem):

    def __init__(self, points, color='r', dashed=False, linewidth=1):
        self.item_type = 'line'
        point_type = '-' if dashed else '--'
        params = { 'linewidth': linewidth }
        # if dashed: params['linestyle': 'dashed']
        # x_vals = list(map(lambda p: p[0], points))
        # y_vals = list(map(lambda p: p[1], points))
        # formatted_points = [x_vals, y_vals]
        super().__init__(points, color, point_type, plt_params=params)

        lines.append(self)

    @staticmethod
    def draw_function(f, x0, x1, step_size=1, color='r'):
        x_vals = list(range(x0, x1, step_size))
        y_vals = list(map(lambda x: f(x), x_vals))
        return LinePlot([x_vals, y_vals], color).draw()

    # def draw(self):
    #     x_vals = list(map(lambda p: p[0], self.points))
    #     y_vals = list(map(lambda p: p[1], self.points))
    #     line, = plt.plot(x_vals, y_vals, **self.plt_params)
    #     self.drawn.append(line)
    #     return self

class LinearLine(GraphItem):

    def __init__(self, f, x0, x1, dx, color, line_width=0.01, dashed=True):
        self.item_type = 'line'
        x_values = range(x0, x1, dx)
        get_point = lambda x: (x, f(x))
        points = list(map(get_point, x_values))
        point_type = '.' # pixel point value
        params = {'markersize': 0.3} #{'linewidth': line_width}
        # if dashed: params['linestyle'] = 'dashed'
        super().__init__(points, color, point_type, plt_params=params)



class VerticalLine(GraphItem):
    def __init__(self, x_intercept, color='r'):
        self.item_type = 'line'
        self.x_intercept = x_intercept
        self.color = color
        self.drawn = []
        self.draw = lambda: self.drawn.append(plt.axvline(x=self.x_intercept, color=color))
        #self.draw = self.draw_vertical_line




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
    S = librosa.feature.melspectrogram(y, sr=sr, n_mels=512, hop_length=1)
    log_S = librosa.power_to_db(S, ref=np.max)
    if plot:
        # plt.figure(figsize=(12,4))
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
