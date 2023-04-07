from matplotlib.backend_bases import MouseEvent, MouseButton

class PointSelector:
    """A cursor object for selecting points in a maptlotlib canvas."""
    def __init__(self, ax):
        self.ax = ax
        self.points = []
        self.labels = []
        self.ax.figure.canvas.mpl_connect("button_press_event", self.on_click)

    def on_click(self, event):
        if not event.inaxes:
            return
        # Add to points
        x, y = event.xdata, event.ydata
        self.points.append((x, y))
        # Left button == foreground, right button == background
        if event.button is MouseButton.LEFT:
            self.labels.append(1)
            self.ax.scatter(x, y, color="tab:blue")
        elif event.button is MouseButton.RIGHT:
            self.labels.append(0)
            self.ax.scatter(x, y, color="tab:red")
        self.ax.figure.canvas.draw()

