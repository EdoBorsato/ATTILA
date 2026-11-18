import shapely as sh

from shapely.plotting import plot_polygon

from scipy.ndimage import binary_dilation, label

import numpy as np

import matplotlib.pyplot as plt

from scipy.spatial import ConvexHull


def _pixel_from_centre(centre, size=1):

    edges = [[centre[1] - size/2, centre[0] - size/2], 
             [centre[1] + size/2, centre[0] - size/2], 
             [centre[1] + size/2, centre[0] + size/2], 
             [centre[1] - size/2, centre[0] + size/2]]
    
    return sh.Polygon(edges)

def _project_polygon(polygon, shape, xrange, yrange):

    arr = np.zeros(shape, dtype=bool)

    yxs = np.meshgrid(yrange, xrange)

    centres = np.concatenate(np.asarray(yxs)[..., np.newaxis], axis=-1)

    for centre in centres.reshape(centres.shape[1]*centres.shape[0], 2):

        pixel = _pixel_from_centre(centre)

        if sh.intersects(polygon, pixel):
            arr[centre[0], centre[1]] = True

    return arr       

def _grey_image_to_polygon(data):

    binary_data = np.where(data > 0, 1, 0)

    edge = binary_dilation(binary_data) - binary_data

    vertices_coords = np.array(np.where(edge)).T

    hull = ConvexHull(vertices_coords)
    vertices_coords = vertices_coords[hull.vertices]

    # Swap x and y coordinates
    vertices_coords = vertices_coords[:, ::-1]

    polygon = sh.geometry.Polygon(vertices_coords)

    return sh.simplify(polygon, 1) 

class InteractiveMask:

    def __init__(self, fig, ax, mask=None):
        self.fig = fig
        self.ax = ax

        self.mask = mask

        self.polys = []

    def to_mask(self):

        if self.mask is not None:
            mask = self.mask.astype(bool)
        else:
            mask = np.zeros(self.mask.shape)

        for poly in self.polys:

            mask_i = _project_polygon(poly, self.mask.shape, np.arange(self.mask.shape[1]), np.arange(self.mask.shape[0]))

            mask = np.logical_or(mask, mask_i)

        return mask


class InteractiveMaskDrawer(InteractiveMask):

    def __init__(self, fig, ax, mask):

        super().__init__(fig, ax, mask)

        self.poly_points = []

        self.points_plots = []
        self.poly_plots = []

    def on_click_draw(self, event):

        if event.inaxes == self.ax:

            if event.button == 1:

                point_plot = self.ax.plot(event.xdata, event.ydata, marker='o', mec='C1', mfc=None) 
                self.points_plots.append(point_plot)

                self.poly_points.append((event.xdata, event.ydata))

                if len(self.poly_points) > 2:

                    self.ax.plot(self.poly_points[0][0], self.poly_points[0][1], marker='o', mec='C1', mfc=None)

                    self.poly = sh.geometry.Polygon(self.poly_points)

                    if not self.poly.is_valid:
                        raise ValueError("Polygon is not valid!")

                    poly_plot = plot_polygon(self.poly, ax=self.ax, add_points=False, ls='--', ec='C1', fc='none', hatch='//', alpha=1)
                    self.poly_plots.append(poly_plot)

                    self.fig.canvas.draw()

            if event.button == 3:
                if len(self.poly_points) > 2:

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

    def __init__(self, fig, ax, mask):
        super().__init__(fig, ax, mask)

        self.splitted_mask, nlab = label(self.mask)

        self.dilation_size = 0

        self.to_update = False

    def on_key_press_dilate_size(self, event):
        self.to_update = True

        if event.key == 'up':
            self.dilation_size += 5
     

        elif event.key == 'down':
            self.dilation_size -= 5

            if self.dilation_size <= 0:
                self.dilation_size = 0

        # Update the dilated polygon and re-plot
        if hasattr(self, 'mask_segment_polygon'):
            self.dilated_segment_polygon = self.mask_segment_polygon.buffer(
                self.dilation_size, join_style='mitre', cap_style='square'
            )
            self.plot_masks()

    def on_click_dilate(self, event):

        if event.inaxes == self.ax:

            if event.button == 1:
                mask_value = self.splitted_mask[int(event.ydata), int(event.xdata)]

                if mask_value != 0:
                    self.mask_segment_polygon = _grey_image_to_polygon(np.where(self.splitted_mask == mask_value, 1, 0))

                    self.cid_key = self.fig.canvas.mpl_connect('key_press_event', self.on_key_press_dilate_size)

                    # self.dilated_segment_polygon = self.mask_segment_polygon.buffer(self.dilation_size, join_style='mitre', cap_style='square')

                    self.plot_masks()

                    if self.to_update:
                        self.plot_masks()

            if event.button == 3:

                self.polys.append(self.dilated_segment_polygon)

                self.cid_key = self.fig.canvas.mpl_disconnect(self.cid_key)

                self.plot_masks()

    def plot_masks(self):
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

    def __init__(self, fig, main_ax, segment=None):

        self.fig = fig
        self.ax = main_ax

        self.segment = segment

        self.polys = []

    def to_segment(self):

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

    def __init__(self, fig, main_ax, segment):

        super().__init__(fig, main_ax, segment)

        self.poly_points = []

        self.points_plots = []
        self.poly_plots = []

    def on_click_draw(self, event):

        if event.inaxes == self.ax:

            if event.button == 1:

                print(f"Clicked at: ({event.xdata}, {event.ydata})")

                point_plot = self.ax.plot(event.xdata, event.ydata, marker='o', mec='C1', mfc=None) 
                self.points_plots.append(point_plot)

                self.poly_points.append((event.xdata, event.ydata))

                if len(self.poly_points) > 2:

                    self.ax.plot(self.poly_points[0][0], self.poly_points[0][1], marker='o', mec='C1', mfc=None)

                    self.poly = sh.geometry.Polygon(self.poly_points)

                    if not self.poly.is_valid:
                        raise ValueError("Polygon is not valid!")

                    poly_plot = plot_polygon(self.poly, ax=self.ax, add_points=False, ls='--', ec='C1', fc='none', hatch='//', alpha=1)
                    self.poly_plots.append(poly_plot)

                    self.fig.canvas.draw()

            if event.button == 3:
                if len(self.poly_points) > 2:

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

    def __init__(self, fig, main_ax, segment):

        super().__init__(fig, main_ax, segment)

        self.segment_value = 0

    def on_click_remove(self, event):

        if event.inaxes == self.ax:

            if event.button == 1:  

                self.segment_value = self.segment[int(event.ydata), int(event.xdata)]

                self.ax.plot(event.xdata, event.ydata, marker='o', mec='C1', mfc=None)


            if (event.button == 2) and (self.segment_value > 0):
                if self.segment_value > 0:

                    self.segment[self.segment == self.segment_value] = self.segment_value

                    self.segment_value = 0

            if (event.button == 3) and (self.segment_value > 0):
                self.segment[self.segment == self.segment_value] = 0

                self.fig.canvas.draw()