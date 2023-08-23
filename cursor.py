import time
from tqdm import tqdm
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
        self._centroids = None

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
        pred, _, _ = self.predictor.predict(self.points, self.labels)
        self._mask = pred[0]
        toc = time.time()
        print(f"Done: {(toc - tic) * 1e3:.2f} ms")

    def show_mask(self):
        if self._mask is None:
            raise ValueError("No mask - run prediction first.")
        self.ax.imshow(self._mask, alpha=0.2)
        self.ax.figure.canvas.draw()

    def clear_canvas(self):
        # NOTE: Creating/destroying new artists/collections every time is not
        # efficient. Refactor if we want to push further
        while len(self.ax.images) > 1:
            self.ax.images[-1].remove()
        for pt in self.ax.collections:
            pt.remove()
        self.ax.figure.canvas.draw()

    def load_centroids(self, centroids):
        # NOTE: In principle, better models won't need so much prompting - demo
        # purposes only
        self._centroids = centroids

    def predict_over_all(self):
        # Like `load_centroids`, exact interface to be defined depending on how
        # updated model works
        if self._centroids is None:
            raise ValueError("Need centroids to prompt full image prediction")
        # Start with an empty mask
        self._mask = np.zeros(
            self.ax.images[0].get_array().shape[:-1], dtype=np.int32
        )
        # Loop over and run a prediction for each centroid (using all others
        # as negative [i.e. "not object"] inputs)
        print("Segmenting entire image...")
        for idx, centroid in enumerate(tqdm(self._centroids)):
            pred, C, logits = self.predictor.predict(
                np.array([centroid]), np.array([1])
            )
            pred, C = pred[0], C[0]  # take "best" mask
            if C > 0.9:  # Arbitrary
                self._mask[pred > 0] = idx
        self.show_mask()

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
        if event.key == "ctrl+r":
            self.predict_over_all()
        if event.key == "c":
            self._points = []
            self._labels = []
            self._mask = None
            self.clear_canvas()
