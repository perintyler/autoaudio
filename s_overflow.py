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
