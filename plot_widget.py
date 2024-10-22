import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from PyQt5 import QtWidgets

class MatplotlibWidget(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.layout = QtWidgets.QVBoxLayout(self)
        self.figure = plt.figure()
        self.canvas = FigureCanvas(self.figure)
        self.layout.addWidget(self.canvas)
        self.setLayout(self.layout)

    def plot(self, x, y):
        """Plot data on the canvas."""
        self.figure.clear()  # Clear the previous figure
        ax = self.figure.add_subplot(111)
        ax.plot(x, y, label='Line Plot')
        ax.set_title('Sample Plot')
        ax.set_xlabel('X Axis')
        ax.set_ylabel('Y Axis')
        ax.legend()
        self.canvas.draw()  # Refresh the canvas
