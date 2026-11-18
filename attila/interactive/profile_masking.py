import shapely as sh

from shapely.plotting import plot_polygon

from scipy.ndimage import binary_dilation, label

import numpy as np

import matplotlib.pyplot as plt

from scipy.spatial import ConvexHull

class InteractiveProfileMask:
    def __init__(self, fig, ax, mask=None):
        self.fig = fig
        self.ax = ax

        self.mask = mask

        self.ranges = []

    def to_mask(self):

        if self.mask is not None:
            self.mask = self.mask.astype(bool)
        else:
            self.mask = np.zeros(self.mask.shape)

        for _range in self.ranges:
            self.mask[slice(*_range)] = True

class InteractiveProfileMasking(InteractiveProfileMask):
    def __init__(self, fig, ax, x, y, mask=None):

        super().__init__(fig, ax, mask)

        self.ranges = []
        self.click_num = 0

        self.x = x
        self.y = y
        
    def on_click_select(self, event):


        if self.click_num % 2 == 0:
            self.lower_ind = event.ind[0]

            self.ax.axvline(self.x[self.lower_ind], color='grey', linestyle='--')

            self.click_num += 1

            
        elif self.click_num % 2 == 1:
            self.upper_ind = event.ind[0]

            self.ranges.append((self.lower_ind, self.upper_ind))

            self.ax.axvline(np.asarray(self.x)[self.upper_ind], color='grey', linestyle='--')

            self.ax.fill_between(np.asarray(self.x)[self.lower_ind:self.upper_ind+1], self.ax.get_ylim()[0], self.ax.get_ylim()[1], ec='grey', fc='none', hatch='/', alpha=0.5)

            self.click_num = 0



