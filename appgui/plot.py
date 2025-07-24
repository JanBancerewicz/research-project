import numpy as np
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


class LivePlot:
    def __init__(self, parent, title, x_axis, y_axis, window_size=1000, figsize=(10, 5), y_lim=(-1500, 2000), fs=130):
        self.window_size = window_size
        self.x_data = [0]
        self.y_data = [0]
        self.fs = fs


        self.y_data2 = None  # opcjonalna druga linia
        self.line2 = None

        self.figure = Figure(figsize, dpi=100)
        self.ax = self.figure.add_subplot(111)
        # self.ax.set_ylim(y_lim[0], y_lim[1])

        self.line, = self.ax.plot(self.x_data, self.y_data, color='green', label="Raw")
        self.scatter = self.ax.scatter([], [], color='red', label="R-peaks", s=20)
        self.scatter_x = []
        self.scatter_y = []

        self.ax.set_title(title)
        self.ax.set_xlabel(x_axis)
        self.ax.set_ylabel(y_axis)
        self.ax.grid(True)
        self.ax.legend(loc='upper right')

        self.canvas = FigureCanvasTkAgg(self.figure, master=parent)
        self.canvas.get_tk_widget().pack()

    # def add_data(self, value, value2=None):
    #     # --- główna linia ---
    #     self.y_data.append(value)
    #     self.y_data = self.y_data[-self.window_size:]
    #
    #     last_x = self.x_data[-1] + (1.0/self.fs) if self.x_data else 0
    #     self.x_data.append(last_x)
    #     self.x_data = self.x_data[-self.window_size:]
    #
    #     # --- druga linia (opcjonalna) ---
    #     if value2 is not None:
    #         if self.y_data2 is None:
    #             self.y_data2 = [value2]
    #         else:
    #             self.y_data2.append(value2)
    #         self.y_data2 = self.y_data2[-self.window_size:]
    #
    #         # Utwórz line2 tylko, gdy x i y są równej długości
    #         if self.line2 is None and len(self.y_data2) == len(self.x_data):
    #             self.line2, = self.ax.plot(self.x_data, self.y_data2, color='orange', label="Filtered")
    #             self.ax.legend(loc='upper right')
    #
    #     # --- aktualizacja wykresu ---
    #     self.line.set_data(self.x_data, self.y_data)
    #     if self.line2 and self.y_data2 and len(self.x_data) == len(self.y_data2):
    #         self.line2.set_data(self.x_data, self.y_data2)
    #
    #     # --- scatter przesuwany ---
    #     min_visible_x = self.x_data[0]
    #     max_visible_x = self.x_data[-1]
    #     visible_indices = [i for i, x in enumerate(self.scatter_x) if min_visible_x <= x <= max_visible_x]
    #
    #     self.scatter_x = [self.scatter_x[i] for i in visible_indices]
    #     self.scatter_y = [self.scatter_y[i] for i in visible_indices]
    #
    #     if self.scatter_x and self.scatter_y:
    #         offsets = np.column_stack((self.scatter_x, self.scatter_y))
    #     else:
    #         offsets = np.empty((0, 2))
    #     self.scatter.set_offsets(offsets)
    #
    #     self.ax.set_xlim(min_visible_x, max_visible_x)
    #     self.canvas.draw_idle()

    def add_data(self, value, value2=None):
        # --- główna linia ---
        self.y_data.append(value)
        self.y_data = self.y_data[-self.window_size:]

        last_x = self.x_data[-1] + (1.0 / self.fs) if self.x_data else 0
        self.x_data.append(last_x)
        self.x_data = self.x_data[-self.window_size:]

        # --- druga linia (opcjonalna) ---
        if value2 is not None:
            if self.y_data2 is None:
                self.y_data2 = [value2]
            else:
                self.y_data2.append(value2)
            self.y_data2 = self.y_data2[-self.window_size:]

            # Utwórz line2 tylko, gdy x i y są równej długości
            if self.line2 is None and len(self.y_data2) == len(self.x_data):
                self.line2, = self.ax.plot(self.x_data, self.y_data2, color='orange', label="Filtered")
                self.ax.legend(loc='upper right')

        # --- aktualizacja wykresu ---
        self.line.set_data(self.x_data, self.y_data)
        if self.line2 and self.y_data2 and len(self.x_data) == len(self.y_data2):
            self.line2.set_data(self.x_data, self.y_data2)

        # --- dynamiczne skalowanie osi Y ---
        all_y_data = self.y_data + (self.y_data2 if self.y_data2 else [])
        min_y = min(all_y_data) - 10  # Add margin
        max_y = max(all_y_data) + 10  # Add margin
        self.ax.set_ylim(min_y, max_y)

        # --- scatter przesuwany ---
        min_visible_x = self.x_data[0]
        max_visible_x = self.x_data[-1]
        visible_indices = [i for i, x in enumerate(self.scatter_x) if min_visible_x <= x <= max_visible_x]

        self.scatter_x = [self.scatter_x[i] for i in visible_indices]
        self.scatter_y = [self.scatter_y[i] for i in visible_indices]

        if self.scatter_x and self.scatter_y:
            offsets = np.column_stack((self.scatter_x, self.scatter_y))
        else:
            offsets = np.empty((0, 2))
        self.scatter.set_offsets(offsets)

        self.ax.set_xlim(min_visible_x, max_visible_x)
        self.canvas.draw_idle()

    def add_scatter_points(self, x_points, y_points):
        """Dodaje scatter (np. R-peaks), pozostają do momentu przewinięcia."""
        x = []
        for i in x_points:
            x.append(i)
        self.scatter_x.extend(x)
        self.scatter_y.extend(y_points)

    def reset(self):
        """Reset wykresu."""
        self.x_data = list(range(self.window_size))
        self.y_data = [0] * self.window_size
        if self.y_data2 is not None:
            self.y_data2 = [0] * self.window_size
