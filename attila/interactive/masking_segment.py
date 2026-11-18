"""
Interactive editing of masks and segmentation maps with the mouse.

This module provides the matplotlib event handlers used from notebooks
to edit, by hand, the boolean mask and the segmentation map of a source
cutout before it is fitted. It sits at the end of the detection /
deblending part of the flow, just before the AstroPhot fits, and is
driven by the plotting classes of `attila.interactive.fitprep`.

Two families of classes are defined:

- `InteractiveMask` and its subclasses `InteractiveMaskDrawer` (draw
  polygons to be added to the mask) and `InteractiveMaskDilator` (grow
  an existing mask blob with the arrow keys).
- `InteractiveSegment` and its subclasses `InteractiveSegmentDrawer`
  (draw polygons that become new segments) and
  `InteractiveSegmentSelector` (pick a segment and delete it).

None of the classes connects itself to the figure: the caller must
register the handler, e.g.
`fig.canvas.mpl_connect('button_press_event', drawer.on_click_draw)`.
The mouse convention of the drawers is: left button adds a vertex,
right button closes and accepts the polygon, middle button undoes the
last vertex. The edited products are obtained afterwards with
`to_mask()` / `to_segment()`, which rasterise the accepted polygons
with `_project_polygon`.

All polygons live in the matplotlib data coordinates of an `imshow`
with `origin='lower'`, i.e. (x, y) = (column, row) with integer values
at the pixel centres.
"""
import shapely as sh

from shapely.plotting import plot_polygon

from scipy.ndimage import binary_dilation, label

import numpy as np

import matplotlib.pyplot as plt

from scipy.spatial import ConvexHull


def _pixel_from_centre(centre, size=1):
    """
    Return the square footprint of one pixel as a shapely polygon.

    Parameters
    ----------
    centre : sequence of float
        The pixel centre as (row, col), i.e. (y, x).
    size : float
        The side of the square in pixels.

    Returns
    -------
    shapely.Polygon
        The square with vertices in (x, y) order, centred on the pixel.
    """

    edges = [[centre[1] - size/2, centre[0] - size/2], 
             [centre[1] + size/2, centre[0] - size/2], 
             [centre[1] + size/2, centre[0] + size/2], 
             [centre[1] - size/2, centre[0] + size/2]]
    
    return sh.Polygon(edges)

def _project_polygon(polygon, shape, xrange, yrange):
    """
    Rasterise a polygon onto a boolean pixel grid.

    Every pixel whose unit-square footprint intersects the polygon (it
    is enough that they touch) is set to `True`, so the result is the
    outer pixelisation of the polygon. Every pixel of the grid is tested
    one by one with shapely, so the cost scales with the number of
    pixels of the grid and not with the size of the polygon.

    Parameters
    ----------
    polygon : shapely.Polygon
        The polygon in (x, y) pixel coordinates, with integer values at
        the pixel centres.
    shape : tuple of int
        The (n_rows, n_cols) shape of the output array.
    xrange : array_like of int
        The column indices to test, normally `np.arange(shape[1])`.
    yrange : array_like of int
        The row indices to test, normally `np.arange(shape[0])`.

    Returns
    -------
    numpy.ndarray
        The boolean array of shape `shape`, `True` on the pixels touched
        by the polygon, indexed as `[row, col]`.
    """

    arr = np.zeros(shape, dtype=bool)

    # Build the list of all (row, col) pixel centres: `centres` has
    # shape `(len(xrange), len(yrange), 2)` with the row index first.
    yxs = np.meshgrid(yrange, xrange)

    centres = np.concatenate(np.asarray(yxs)[..., np.newaxis], axis=-1)

    for centre in centres.reshape(centres.shape[1]*centres.shape[0], 2):

        pixel = _pixel_from_centre(centre)

        if sh.intersects(polygon, pixel):
            arr[centre[0], centre[1]] = True

    return arr       

def _grey_image_to_polygon(data):
    """
    Return the simplified convex hull of the non-zero image pixels.

    The one-pixel ring just outside the non-zero region is found with a
    binary dilation, the convex hull of the ring pixels is computed and
    then simplified with a tolerance of 1 pixel. Concavities of the
    region are therefore not preserved.

    Parameters
    ----------
    data : numpy.ndarray
        The 2D image, indexed as `[row, col]`; every pixel `> 0` belongs
        to the region.

    Returns
    -------
    shapely.Polygon
        The hull polygon in (x, y) pixel coordinates.
    """

    binary_data = np.where(data > 0, 1, 0)

    # The outline is the ring of pixels added by a one-pixel dilation.
    edge = binary_dilation(binary_data) - binary_data

    vertices_coords = np.array(np.where(edge)).T

    hull = ConvexHull(vertices_coords)
    vertices_coords = vertices_coords[hull.vertices]

    # Swap the (row, col) coordinates to (x, y) for shapely.
    vertices_coords = vertices_coords[:, ::-1]

    polygon = sh.geometry.Polygon(vertices_coords)

    return sh.simplify(polygon, 1) 

class InteractiveMask:
    """
    Base class holding a mask and the polygons to be added to it.

    The subclasses fill `polys` through mouse / keyboard events on
    `ax`; `to_mask` then merges them with the input mask.

    Attributes
    ----------
    fig : matplotlib.figure.Figure
        The figure whose canvas receives the events.
    ax : matplotlib.axes.Axes
        The axis on which the image is shown and the polygons are drawn.
    mask : numpy.ndarray
        The input mask, shape `(n_rows, n_cols)`; non-zero means masked.
    polys : list of shapely.Polygon
        The accepted polygons, in (x, y) pixel coordinates.
    """

    def __init__(self, fig, ax, mask=None):
        """
        Store the figure, the axis and the starting mask.

        Parameters
        ----------
        fig : matplotlib.figure.Figure
            The figure whose canvas receives the events.
        ax : matplotlib.axes.Axes
            The axis showing the image to be masked.
        mask : numpy.ndarray
            The existing mask, shape `(n_rows, n_cols)`. The default
            `None` is accepted here, but `to_mask` needs an array
            because it reads the output shape from it.
        """
        self.fig = fig
        self.ax = ax

        self.mask = mask

        self.polys = []

    def to_mask(self):
        """
        Return the input mask merged with all the accepted polygons.

        Each polygon in `polys` is rasterised with `_project_polygon`
        and combined with the input mask through a logical OR. The
        stored `mask` is not modified.

        Returns
        -------
        numpy.ndarray
            The boolean mask, shape `(n_rows, n_cols)`.
        """

        if self.mask is not None:
            mask = self.mask.astype(bool)
        else:
            mask = np.zeros(self.mask.shape)

        for poly in self.polys:

            mask_i = _project_polygon(poly, self.mask.shape, np.arange(self.mask.shape[1]), np.arange(self.mask.shape[0]))

            mask = np.logical_or(mask, mask_i)

        return mask


class InteractiveMaskDrawer(InteractiveMask):
    """
    Add regions to a mask by drawing polygons with the mouse.

    Connect `on_click_draw` to the `button_press_event` of the figure.
    Left click adds a vertex, right click accepts the current polygon
    (at least three vertices), middle click removes the last vertex.

    Attributes
    ----------
    poly_points : list of tuple of float
        The (x, y) vertices of the polygon being drawn.
    poly : shapely.Polygon
        The polygon built from `poly_points` at the last left click;
        it only exists once three vertices have been clicked.
    points_plots : list
        The matplotlib artists of the vertex markers being drawn.
    poly_plots : list
        The matplotlib artists of the dashed previews of the polygon
        being drawn, one per vertex after the second.
    """

    def __init__(self, fig, ax, mask):
        """
        Initialise the drawer with an empty polygon.

        Parameters
        ----------
        fig : matplotlib.figure.Figure
            The figure whose canvas receives the events.
        ax : matplotlib.axes.Axes
            The axis on which the polygons are drawn.
        mask : numpy.ndarray
            The existing mask, shape `(n_rows, n_cols)`.
        """

        super().__init__(fig, ax, mask)

        self.poly_points = []

        self.points_plots = []
        self.poly_plots = []

    def on_click_draw(self, event):
        """
        Handle a mouse click while drawing a mask polygon.

        Clicks outside `ax` are ignored. Left button (1): append the
        clicked (x, y) position as a vertex and, from the third vertex
        on, rebuild `poly` and draw it as a dashed preview. Right button
        (3): accept `poly`, append it to `polys`, redraw it with a solid
        edge and start a new polygon. Middle button (2): remove the last
        vertex and its marker; `poly` is not rebuilt from the remaining
        vertices, it is only reset to `None` when fewer than three
        vertices are left.

        Parameters
        ----------
        event : matplotlib.backend_bases.MouseEvent
            The `button_press_event` to process.

        Raises
        ------
        ValueError
            If the clicked vertices give an invalid (e.g. self-crossing)
            polygon, or if the polygon is accepted with fewer than three
            vertices.
        """

        if event.inaxes == self.ax:

            # Left click: add a vertex.
            if event.button == 1:

                point_plot = self.ax.plot(event.xdata, event.ydata, marker='o', mec='C1', mfc=None) 
                self.points_plots.append(point_plot)

                self.poly_points.append((event.xdata, event.ydata))

                # Three vertices are enough to preview the polygon.
                if len(self.poly_points) > 2:

                    self.ax.plot(self.poly_points[0][0], self.poly_points[0][1], marker='o', mec='C1', mfc=None)

                    self.poly = sh.geometry.Polygon(self.poly_points)

                    if not self.poly.is_valid:
                        raise ValueError("Polygon is not valid!")

                    poly_plot = plot_polygon(self.poly, ax=self.ax, add_points=False, ls='--', ec='C1', fc='none', hatch='//', alpha=1)
                    self.poly_plots.append(poly_plot)

                    self.fig.canvas.draw()

            # Right click: accept the polygon and start a new one.
            if event.button == 3:
                if len(self.poly_points) > 2:

                    # Remove the temporary markers and dashed previews.
                    [self.poly_plots[i].remove() for i in range(len(self.poly_plots))]
                    [self.points_plots[i][0].remove() for i in range(len(self.points_plots))]

                    self.polys.append(self.poly)

                    for point in self.poly_points:
                        self.ax.plot(point[0], point[1], marker='o', mec='C1', mfc=None)

                    plot_polygon(self.poly, ax=self.ax, add_points=False, ls='-', ec='C1', fc='none', hatch='//', alpha=1)

                    self.fig.canvas.draw()

                    self.poly_points = []
                    self.poly_plots = []
                    self.points_plots = []

                else:
                    raise ValueError("Not enough points to create a polygon.")

            # Middle click: undo the last vertex.
            if event.button == 2:  
                if len(self.poly_points) > 2:
                    
                    self.poly_points.pop()

                    self.points_plots[-1][0].remove()  
                    self.points_plots.pop()

                    self.poly_plots[-1].remove()  
                    self.poly_plots.pop()

                    self.fig.canvas.draw()

                elif len(self.poly_points) > 1:

                    self.poly_points.pop()
                    
                    self.points_plots[-1][0].remove()  
                    self.points_plots.pop()

                    [self.poly_plots[i].remove() for i in range(len(self.poly_plots))]

                    self.poly = None
                    self.poly_plots = []

                else:
                    self.points_plots = []
                    self.poly_plots = []
                    self.poly_points = []

class InteractiveMaskDilator(InteractiveMask):
    """
    Grow one connected blob of a mask with the arrow keys.

    Connect `on_click_dilate` to the `button_press_event` of the figure.
    A left click on a masked pixel selects the connected blob under the
    cursor and replaces it by its convex hull polygon; the up / down
    arrow keys then grow / shrink a buffer around the hull in steps of
    5 pixels, and a right click accepts the buffered polygon, which is
    added to `polys` (and hence to the output of `to_mask`).

    Attributes
    ----------
    splitted_mask : numpy.ndarray
        The mask split into connected components by
        `scipy.ndimage.label` (default 4-connectivity); 0 is unmasked.
    dilation_size : int
        The current buffer width in pixels; it is not reset when another
        blob is selected.
    to_update : bool
        Set to `True` at the first key press.
    mask_segment_polygon : shapely.Polygon
        The hull of the selected blob (set by the first left click).
    dilated_segment_polygon : shapely.Polygon
        The buffered hull (set by the first key press after a
        selection).
    """

    def __init__(self, fig, ax, mask):
        """
        Label the connected blobs of the mask and reset the buffer.

        Parameters
        ----------
        fig : matplotlib.figure.Figure
            The figure whose canvas receives the events.
        ax : matplotlib.axes.Axes
            The axis on which the mask outlines are drawn.
        mask : numpy.ndarray
            The existing mask, shape `(n_rows, n_cols)`.
        """
        super().__init__(fig, ax, mask)

        self.splitted_mask, nlab = label(self.mask)

        self.dilation_size = 0

        self.to_update = False

    def on_key_press_dilate_size(self, event):
        """
        Change the buffer width with the arrow keys and redraw.

        The up arrow adds 5 pixels to `dilation_size`, the down arrow
        subtracts 5 pixels with a floor at 0; any other key leaves it
        unchanged. If a blob is selected, its hull is buffered by
        `dilation_size` with mitred corners and the outlines are
        redrawn.

        Parameters
        ----------
        event : matplotlib.backend_bases.KeyEvent
            The `key_press_event` to process.
        """
        self.to_update = True

        if event.key == 'up':
            self.dilation_size += 5
     

        elif event.key == 'down':
            self.dilation_size -= 5

            if self.dilation_size <= 0:
                self.dilation_size = 0

        # Update the dilated polygon and re-plot.
        if hasattr(self, 'mask_segment_polygon'):
            self.dilated_segment_polygon = self.mask_segment_polygon.buffer(
                self.dilation_size, join_style='mitre', cap_style='square'
            )
            self.plot_masks()

    def on_click_dilate(self, event):
        """
        Handle a mouse click while dilating a mask blob.

        Clicks outside `ax` are ignored. Left button (1): read the blob
        label under the cursor (the click position is truncated to
        integers) and, if it is a masked pixel, store the hull of that
        blob in `mask_segment_polygon`, connect
        `on_key_press_dilate_size` to the key press events and draw the
        outlines. Right button (3): append `dilated_segment_polygon` to
        `polys`, disconnect the key handler and redraw.

        Parameters
        ----------
        event : matplotlib.backend_bases.MouseEvent
            The `button_press_event` to process.
        """

        if event.inaxes == self.ax:

            # Left click: select the blob under the cursor.
            if event.button == 1:
                mask_value = self.splitted_mask[int(event.ydata), int(event.xdata)]

                if mask_value != 0:
                    self.mask_segment_polygon = _grey_image_to_polygon(np.where(self.splitted_mask == mask_value, 1, 0))

                    self.cid_key = self.fig.canvas.mpl_connect('key_press_event', self.on_key_press_dilate_size)

                    # self.dilated_segment_polygon = self.mask_segment_polygon.buffer(self.dilation_size, join_style='mitre', cap_style='square')

                    self.plot_masks()

                    if self.to_update:
                        self.plot_masks()

            # Right click: accept the dilated polygon.
            if event.button == 3:

                self.polys.append(self.dilated_segment_polygon)

                self.cid_key = self.fig.canvas.mpl_disconnect(self.cid_key)

                self.plot_masks()

    def plot_masks(self):
        """
        Draw the selected blob, the accepted polygons and the buffer.

        The hull of the selected blob and the accepted polygons are
        drawn in orange (`C1`); the current buffered polygon is drawn in
        red (`C3`), replacing the one drawn at the previous call.
        """
        plot_polygon(self.mask_segment_polygon, ax=self.ax, add_points=False, ls='-', ec='C1', fc='none', hatch='/', alpha=1)

        if len(self.polys) > 0:
            for poly in self.polys:
                plot_polygon(poly, ax=self.ax, add_points=False, ls='-', ec='C1', fc='none', hatch='/', alpha=1)

        if hasattr(self, 'dil_poly_plot'):
            self.dil_poly_plot.remove()

        if hasattr(self, 'dilated_segment_polygon'):
            self.dil_poly_plot = plot_polygon(self.dilated_segment_polygon, ax=self.ax, add_points=False, ls='-', ec='C3', fc='none', hatch='///', alpha=1)

        self.fig.canvas.draw()



class InteractiveSegment:
    """
    Base class holding a segmentation map and polygons to add to it.

    The subclasses fill `polys` (or edit `segment`) through mouse events
    on `ax`; `to_segment` then paints the polygons as new labels.

    Attributes
    ----------
    fig : matplotlib.figure.Figure
        The figure whose canvas receives the events.
    ax : matplotlib.axes.Axes
        The axis on which the events are accepted.
    segment : numpy.ndarray
        The segmentation map, shape `(n_rows, n_cols)`; 0 is background
        and positive integers are segment labels.
    polys : list of shapely.Polygon
        The accepted polygons, in (x, y) pixel coordinates.
    max_label : int
        The largest label of the input map (set by `to_segment`).
    """

    def __init__(self, fig, main_ax, segment=None):
        """
        Store the figure, the axis and the starting segmentation map.

        Parameters
        ----------
        fig : matplotlib.figure.Figure
            The figure whose canvas receives the events.
        main_ax : matplotlib.axes.Axes
            The axis on which the events are accepted.
        segment : numpy.ndarray
            The existing segmentation map, shape `(n_rows, n_cols)`. The
            default `None` is accepted here, but `to_segment` needs an
            array because it reads the output shape from it.
        """

        self.fig = fig
        self.ax = main_ax

        self.segment = segment

        self.polys = []

    def to_segment(self):
        """
        Return the segmentation map with the polygons added as segments.

        The map is copied as `int32`. The polygon with index `i` in
        `polys` is rasterised with `_project_polygon` and painted with
        the label `max_label + i + 1`, overwriting whatever label the
        covered pixels had (including pixels painted by an earlier
        polygon). The stored `segment` is not modified.

        Returns
        -------
        numpy.ndarray
            The new segmentation map, shape `(n_rows, n_cols)`.
        """

        if self.segment is not None:
            segment = self.segment.astype(np.int32)
        else:
            segment = np.zeros(self.segment.shape)

        self.max_label = np.max(segment)

        for i, poly in enumerate(self.polys):

            segment_i = (self.max_label + i + 1) * _project_polygon(
                poly, 
                self.segment.shape, 
                np.arange(self.segment.shape[1]), 
                np.arange(self.segment.shape[0])
            )

            segment[segment_i > 0] = segment_i[segment_i > 0]

        return segment

class InteractiveSegmentDrawer(InteractiveSegment):
    """
    Add segments to a segmentation map by drawing polygons.

    Connect `on_click_draw` to the `button_press_event` of the figure.
    The mouse convention is the one of `InteractiveMaskDrawer`: left
    click adds a vertex, right click accepts the polygon, middle click
    removes the last vertex.

    Attributes
    ----------
    poly_points : list of tuple of float
        The (x, y) vertices of the polygon being drawn.
    poly : shapely.Polygon
        The polygon built from `poly_points` at the last left click;
        it only exists once three vertices have been clicked.
    points_plots : list
        The matplotlib artists of the vertex markers being drawn.
    poly_plots : list
        The matplotlib artists of the dashed previews of the polygon
        being drawn.
    """

    def __init__(self, fig, main_ax, segment):
        """
        Initialise the drawer with an empty polygon.

        Parameters
        ----------
        fig : matplotlib.figure.Figure
            The figure whose canvas receives the events.
        main_ax : matplotlib.axes.Axes
            The axis on which the polygons are drawn.
        segment : numpy.ndarray
            The existing segmentation map, shape `(n_rows, n_cols)`.
        """

        super().__init__(fig, main_ax, segment)

        self.poly_points = []

        self.points_plots = []
        self.poly_plots = []

    def on_click_draw(self, event):
        """
        Handle a mouse click while drawing a segment polygon.

        Clicks outside `ax` are ignored. Left button (1): print the
        clicked position, append it as a vertex and, from the third
        vertex on, rebuild `poly` and draw it as a dashed preview. Right
        button (3): accept `poly`, redraw it with a solid edge and start
        a new polygon; the accepted polygon is appended to `polys` twice
        (before and after the redraw). Middle button (2): remove the
        last vertex and its marker; `poly` is not rebuilt from the
        remaining vertices, it is only reset to `None` when fewer than
        three vertices are left.

        Parameters
        ----------
        event : matplotlib.backend_bases.MouseEvent
            The `button_press_event` to process.

        Raises
        ------
        ValueError
            If the clicked vertices give an invalid (e.g. self-crossing)
            polygon, or if the polygon is accepted with fewer than three
            vertices.
        """

        if event.inaxes == self.ax:

            # Left click: add a vertex.
            if event.button == 1:

                print(f"Clicked at: ({event.xdata}, {event.ydata})")

                point_plot = self.ax.plot(event.xdata, event.ydata, marker='o', mec='C1', mfc=None) 
                self.points_plots.append(point_plot)

                self.poly_points.append((event.xdata, event.ydata))

                # Three vertices are enough to preview the polygon.
                if len(self.poly_points) > 2:

                    self.ax.plot(self.poly_points[0][0], self.poly_points[0][1], marker='o', mec='C1', mfc=None)

                    self.poly = sh.geometry.Polygon(self.poly_points)

                    if not self.poly.is_valid:
                        raise ValueError("Polygon is not valid!")

                    poly_plot = plot_polygon(self.poly, ax=self.ax, add_points=False, ls='--', ec='C1', fc='none', hatch='//', alpha=1)
                    self.poly_plots.append(poly_plot)

                    self.fig.canvas.draw()

            # Right click: accept the polygon and start a new one.
            if event.button == 3:
                if len(self.poly_points) > 2:

                    # Remove the temporary markers and dashed previews.
                    [self.poly_plots[i].remove() for i in range(len(self.poly_plots))]
                    [self.points_plots[i][0].remove() for i in range(len(self.points_plots))]

                    self.polys.append(self.poly)

                    for point in self.poly_points:
                        self.ax.plot(point[0], point[1], marker='o', mec='C1', mfc=None)

                    plot_polygon(self.poly, ax=self.ax, add_points=False, ls='-', ec='C1', fc='none', hatch='//', alpha=1)

                    self.fig.canvas.draw()

                    self.poly_points = []
                    self.poly_plots = []
                    self.points_plots = []
                    
                    self.polys.append(self.poly)

                else:
                    raise ValueError("Not enough points to create a polygon.")

            # Middle click: undo the last vertex.
            if event.button == 2:  
                if len(self.poly_points) > 2:
                        
                    self.poly_points.pop()

                    self.points_plots[-1][0].remove()  
                    self.points_plots.pop()

                    self.poly_plots[-1].remove()  
                    self.poly_plots.pop()

                    self.fig.canvas.draw()

                elif len(self.poly_points) > 1:

                    self.poly_points.pop()
                        
                    self.points_plots[-1][0].remove()  
                    self.points_plots.pop()

                    [self.poly_plots[i].remove() for i in range(len(self.poly_plots))]

                    self.poly = None
                    self.poly_plots = []

                else:
                    self.points_plots = []
                    self.poly_plots = []
                    self.poly_points = []

class InteractiveSegmentSelector(InteractiveSegment):
    """
    Select a segment with the mouse and remove it from the map.

    Connect `on_click_remove` to the `button_press_event` of the figure.
    A left click selects the segment under the cursor, a right click
    deletes it and a middle click drops the selection. The deletion is
    done in place on the array given at construction.

    Attributes
    ----------
    segment_value : int
        The label of the selected segment; 0 means no selection.
    """

    def __init__(self, fig, main_ax, segment):
        """
        Initialise the selector with no segment selected.

        Parameters
        ----------
        fig : matplotlib.figure.Figure
            The figure whose canvas receives the events.
        main_ax : matplotlib.axes.Axes
            The axis on which the clicks are accepted.
        segment : numpy.ndarray
            The segmentation map to edit in place, shape
            `(n_rows, n_cols)`.
        """

        super().__init__(fig, main_ax, segment)

        self.segment_value = 0

    def on_click_remove(self, event):
        """
        Handle a mouse click while selecting / removing a segment.

        Clicks outside `ax` are ignored. Left button (1): store in
        `segment_value` the label under the cursor (the click position
        is truncated to integers) and mark the position. Middle button
        (2): if a segment is selected, clear the selection; the map is
        left unchanged. Right button (3): if a segment is selected, set
        all its pixels to 0 in `segment` (in place) and redraw the
        canvas; the displayed outlines are not rebuilt and
        `segment_value` is kept.

        Parameters
        ----------
        event : matplotlib.backend_bases.MouseEvent
            The `button_press_event` to process.
        """

        if event.inaxes == self.ax:

            # Left click: select the segment under the cursor.
            if event.button == 1:  

                self.segment_value = self.segment[int(event.ydata), int(event.xdata)]

                self.ax.plot(event.xdata, event.ydata, marker='o', mec='C1', mfc=None)


            # Middle click: drop the selection (the assignment below
            # leaves the map unchanged).
            if (event.button == 2) and (self.segment_value > 0):
                if self.segment_value > 0:

                    self.segment[self.segment == self.segment_value] = self.segment_value

                    self.segment_value = 0

            # Right click: delete the selected segment.
            if (event.button == 3) and (self.segment_value > 0):
                self.segment[self.segment == self.segment_value] = 0

                self.fig.canvas.draw()