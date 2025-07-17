import numpy as np
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


class LivePlot:
    def __init__(self, parent, title, x_axis, y_axis, window_size=1000, figsize=(10,5), y_lim=(-1500,2000)):
        self.window_size = window_size
        self.y_data = [0]
        self.x_data = [0]

        self.figure = Figure(figsize, dpi=100)
        self.ax = self.figure.add_subplot(111)
        self.ax.set_ylim(y_lim[0], y_lim[1])
        self.line, = self.ax.plot(self.x_data, self.y_data, color='green')
        self.scatter = self.ax.scatter([], [], color='red', label="R-peaks", s=20)
        self.scatter_x = []
        self.scatter_y = []

        self.ax.set_title(title)
        self.ax.set_xlabel(x_axis)
        self.ax.set_ylabel(y_axis)
        self.ax.grid(True)

        self.canvas = FigureCanvasTkAgg(self.figure, master=parent)
        self.canvas.get_tk_widget().pack()

    def add_data(self, value):
        self.y_data.append(value)
        self.y_data = self.y_data[-self.window_size:]

        last_x = self.x_data[-1] + 1 if self.x_data else 0
        self.x_data.append(last_x)
        self.x_data = self.x_data[-self.window_size:]

        # Scroll scatter points if needed
        min_visible_x = self.x_data[0]
        max_visible_x = self.x_data[-1]

        visible_indices = [
            i for i, x in enumerate(self.scatter_x) if min_visible_x <= x <= max_visible_x
        ]
        self.scatter_x = [self.scatter_x[i] for i in visible_indices]
        self.scatter_y = [self.scatter_y[i] for i in visible_indices]

        # Update line and scatter
        self.line.set_data(self.x_data, self.y_data)
        if self.scatter_x and self.scatter_y:
            offsets = np.column_stack((self.scatter_x, self.scatter_y))
        else:
            offsets = np.empty((0, 2))

        self.scatter.set_offsets(offsets)

        self.ax.set_xlim(min_visible_x, max_visible_x)
        self.canvas.draw_idle()

    def add_scatter_points(self, x_points, y_points):
        """
        Add new scatter points; they persist until they scroll out of view.
        """
        self.scatter_x.extend(x_points)
        self.scatter_y.extend(y_points)


    def reset(self):
        self.x_data = list(range(self.window_size))
        self.y_data = [0] * self.window_size


