    # def test(self):
    #     if len(self.hull) > 2:
    #         m = 0
    #         max_index = 0
    #         for i in range(len(self.hull)):
    #             p = self.hull[i]
    #             if(p[1] > m):
    #                 m = p[1]
    #                 max_index = i
    #         print('d0', self.hull[max_index][0] - self.hull[max_index-1][0])
    #         print('el', self.hull[len(self.hull)-1][0] - self.hull[max_index][0])
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
            # if the points tangent line encroaches inside the hull, the hull
            # point must be removed. Since only an upper hull is computed,
            # there will be encroachment whenever point_tangent(hpT - tS)
            # is greater than hull_tangent(hpT - tS) where hpT is the t value
            # of the hull point and tS is any very close t value that is less
            # than hpT. in this case hpT - sample rate
            # if plot:
            #     if not onetime and new_point[0] == 207744000:
            #         slope = lambda p0,p1: (p1[1] - p0[1])/(p1[0] - p0[0])
            #         print(new_point, hull_pnt)
            #         onetime = True
            #         i = 207744000
            #         print(hull_pnt)
            #         print(hull_tangent(hull_pnt[0]))
            #         print(new_point)
            #         print(point_tangent(new_point[0]))


                    # while i > hull_pnt[0]:
                    #     val = point_tangent(i)
                    #     print('0', i, val)
                    #     plt.plot(i, val, 'm,')
                    #     i-=self.sr
                    # i = hull_pnt[0]
                    # while i > self.hull[hull_index-1][0]:
                    #     val = hull_tangent(i)
                    #     print('1', i, val)
                    #     plt.plot(i, val, 'y,')
                    #     i-=self.sr

                # pt = float(point_tangent(t_close))
                # ht = float(hull_tangent(t_close))
                # print('****')
                # print(pt)
                # print(ht)
                # print(new_point)
                # print(hull_pnt)
                # print('removed' if pt > ht else 'kept')
                # print('________')
    # def hull_distances(self):
    #     if len(self.hull) == 0: return
    #     distances = []
    #     max_index = 0
    #     for i in range(len(self.hull)-1):
    #         if(self.hull[i][0] == self.y_max[0]):
    #             max_index = i
    #         distances.append(int((self.hull[i+1][0] - self.hull[i][0])/self.sr))
    #     print(distances, max_index)


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
# def get_outline(signal, plot=False):
#     hull = scipy.spatial.ConvexHull(signal)
#     if plot:
#         # add the first point to the end of vertices to close the outline
#         outline = np.append(hull.vertices, hull.vertices[0])
#         plt.plot(signal[outline,0], signal[outline,1], 'r--', lw=1)
#         plt.plot(signal[outline,0], signal[outline,1], 'g.')
#     # x = list(map(lambda simplex: signal[simplex, 0], hull.simplices))
#     # y = list(map(lambda simplex: signal[simplex, 1], hull.simplices))
#     return signal[hull.vertices,0], signal[hull.vertices,1]
#
