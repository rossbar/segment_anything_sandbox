import time
import numpy as np
from matplotlib.backend_bases import MouseEvent, MouseButton

class InteractivePredictor:
    """A cursor object for selecting points in a maptlotlib canvas."""
    def __init__(self, ax, predictor):
        self.ax = ax
        self.predictor = predictor  # The model
        self._points = []
        self._labels = []
        self._mask = None

        # Attach to the matplotlib event loop
        self.ax.figure.canvas.mpl_connect("button_press_event", self.on_click)
        self.ax.figure.canvas.mpl_connect("key_press_event", self.on_press)

    @property
    def points(self):
        return np.asarray(self._points)

    @property
    def labels(self):
        return np.asarray(self._labels)

    def predict(self):
        print("Running prediction...")
        tic = time.time()
        self._mask, _, _ = self.predictor.predict(self.points, self.labels)
        toc = time.time()
        print(f"Done: {(toc - tic) * 1e3:.2f} ms")

    def show_mask(self):
        if self._mask is None:
            raise ValueError("No mask - run prediction first.")
        self.ax.imshow(self._mask[0], alpha=0.2)
        self.ax.figure.canvas.draw()


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

    def on_press(self, event):
        print("Press", event.key)
        if event.key == "r":
            self.predict()
            self.show_mask()
        if event.key == "c":
            self._points = []
            self._labels = []
