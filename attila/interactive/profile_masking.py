"""
Interactive masking of index ranges on a 1D profile.

This module provides the matplotlib pick-event handler used to flag
bad stretches of a radial profile (e.g. the isophote table shown by
`attila.fitting.isophote`) before the 1D fits. Ranges are selected by
picking two points of the plotted profile; `to_mask` turns them into a
boolean mask over the profile samples.
"""

import numpy as np


class InteractiveProfileMask:
    """
    Base class holding a 1D mask and the index ranges to add to it.

    Attributes
    ----------
    fig : matplotlib.figure.Figure
        The figure whose canvas receives the events.
    ax : matplotlib.axes.Axes
        The axis on which the profile is plotted.
    mask : numpy.ndarray
        The 1D mask over the profile samples; non-zero means masked.
    ranges : list of tuple of int
        The selected `(lower_index, upper_index)` pairs.
    """
    def __init__(self, fig, ax, mask=None):
        """
        Store the figure, the axis and the starting mask.

        Parameters
        ----------
        fig : matplotlib.figure.Figure
            The figure whose canvas receives the events.
        ax : matplotlib.axes.Axes
            The axis on which the profile is plotted.
        mask : numpy.ndarray
            The existing 1D mask, one element per profile sample. The
            default `None` is accepted here, but `to_mask` needs an
            array because it reads the output shape from it.
        """
        self.fig = fig
        self.ax = ax

        self.mask = mask

        self.ranges = []

    def to_mask(self):
        """
        Update `mask` with the selected index ranges.

        The mask is converted to boolean and, for every pair in
        `ranges`, the points from `lower_index` to `upper_index`, both
        included, are set to `True`, as hatched on the plot. The result
        replaces the `mask` attribute; nothing is returned.
        """

        if self.mask is None:
            raise ValueError("An input mask is needed to define the shape of the output.")

        self.mask = self.mask.astype(bool)

        for lower, upper in self.ranges:
            self.mask[lower:upper + 1] = True

class InteractiveProfileMasking(InteractiveProfileMask):
    """
    Select index ranges on a plotted profile through pick events.

    Connect `on_click_select` to the `pick_event` of the figure; the
    profile must be plotted with a `picker` so that its points can be
    picked. Picks are taken in pairs: the first one sets the lower index
    of a range and the second one the upper index.

    Attributes
    ----------
    x : array_like
        The abscissa of the profile samples.
    y : array_like
        The ordinate of the profile samples (stored, not used here).
    click_num : int
        0 when the next pick opens a range, 1 when it closes it.
    lower_ind, upper_ind : int
        The sample indices of the last two picks.
    """
    def __init__(self, fig, ax, x, y, mask=None):
        """
        Initialise the selector with no range selected.

        Parameters
        ----------
        fig : matplotlib.figure.Figure
            The figure whose canvas receives the events.
        ax : matplotlib.axes.Axes
            The axis on which the profile is plotted.
        x : array_like
            The abscissa of the profile samples, used to place the
            vertical lines.
        y : array_like
            The ordinate of the profile samples.
        mask : numpy.ndarray
            The existing 1D mask, one element per profile sample.
        """

        super().__init__(fig, ax, mask)

        self.ranges = []
        self.click_num = 0

        self.x = x
        self.y = y
        
    def on_click_select(self, event):
        """
        Handle a pick event on the profile.

        The index of the first picked sample (`event.ind[0]`) is used.
        On the first pick of a pair a dashed vertical line is drawn at
        the lower limit. On the second pick the pair
        `(lower_ind, upper_ind)` is appended to `ranges`, a second line
        is drawn and the samples from `lower_ind` to `upper_ind`
        (included) are hatched over the current y-limits of the axis.

        Parameters
        ----------
        event : matplotlib.backend_bases.PickEvent
            The `pick_event` to process.
        """


        # First pick of a pair: lower limit of the range.
        if self.click_num % 2 == 0:
            self.lower_ind = event.ind[0]

            self.ax.axvline(self.x[self.lower_ind], color='grey', linestyle='--')

            self.click_num += 1

            
        # Second pick: upper limit, close and shade the range.
        elif self.click_num % 2 == 1:
            self.upper_ind = event.ind[0]

            self.ranges.append((self.lower_ind, self.upper_ind))

            self.ax.axvline(np.asarray(self.x)[self.upper_ind], color='grey', linestyle='--')

            self.ax.fill_between(np.asarray(self.x)[self.lower_ind:self.upper_ind+1], self.ax.get_ylim()[0], self.ax.get_ylim()[1], ec='grey', fc='none', hatch='/', alpha=0.5)

            self.click_num = 0



