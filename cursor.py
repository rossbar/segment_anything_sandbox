import numpy as np
from matplotlib.backend_bases import MouseEvent, MouseButton

class PointSelector:
    """A cursor object for selecting points in a maptlotlib canvas."""
    def __init__(self, ax):
        self.ax = ax
        self._points = []
        self._labels = []

        # Attach to the matplotlib event loop
        self.ax.figure.canvas.mpl_connect("button_press_event", self.on_click)

    @property
    def points(self):
        return np.asarray(self._points)

    @property
    def labels(self):
        return np.asarray(self._labels)

    def on_click(self, event):
        if not event.inaxes:
            return
        # Add to points
        x, y = event.xdata, event.ydata
        self._points.append((x, y))
        # Left button == foreground, right button == background
        if event.button is MouseButton.LEFT:
            self._labels.append(1)
            self.ax.scatter(x, y, color="tab:blue")
        elif event.button is MouseButton.RIGHT:
            self._labels.append(0)
            self.ax.scatter(x, y, color="tab:red")
        self.ax.figure.canvas.draw()
