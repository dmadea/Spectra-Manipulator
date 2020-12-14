from PyQt5 import QtCore, QtGui
from PyQt5.QtWidgets import QApplication

import pyqtgraph.opengl as gl
import pyqtgraph as pg
import numpy as np
from .spectrum import group2mat
import sys


class GLPlotter:
    def __init__(self, group_item=None):

        self.x, self.y, self.mat = group2mat(group_item)
        if self.y is None:
            return

        self.x -= self.x.mean()
        self.y -= self.y.mean()
        self.mat *= 50

        self.traces = []
        self.w = gl.GLViewWidget()
        self.w.setCameraPosition(distance=500)
        self.w.setWindowTitle(f'Plot of {group_item.name}')
        self.w.setGeometry(0, 110, 1400, 700)
        self.w.show()

        gax = gl.GLAxisItem(size=QtGui.QVector3D(self.x.mean(), self.y.mean(), self.mat.min()))
        self.w.addItem(gax)

        self.plot_data()

    def plot_data(self):
        # for i in range(self.y.shape[0]):
        #     pts = np.vstack((self.x, np.ones_like(self.x) * self.y[i], self.mat[:, i])).T
        #     self.traces.append(gl.GLLinePlotItem(pos=pts, color=pg.glColor(
        #         (i, self.y.shape[0] * 1.3)), width=2, antialias=True))
        #     self.w.addItem(self.traces[i])

        surface = gl.GLSurfacePlotItem(x=self.x, y=self.y, z=self.mat, shader='heightColor', drawEdges=True)
        surface.shader()['colorMap'] = np.array([0.2, 2, 0.5])
        self.w.addItem(surface)

    #
    # def set_plotdata(self, name, points, color, width):
    #     self.traces[name].setData(pos=points, color=color, width=width)
    #
    # def update(self):
    #     for i in range(self.n):
    #         yi = np.array([self.y[i]] * self.m)
    #         d = np.sqrt(self.x ** 2 + yi ** 2)
    #         z = 10 * np.cos(d + self.phase) / (d + 1)
    #         pts = np.vstack([self.x, yi, z]).transpose()
    #         self.set_plotdata(
    #             name=i, points=pts,
    #             color=pg.glColor((255, 0, 0, 255*i/self.n)),
    #             width=(i + 1) / 10
    #         )
    #         self.phase -= .001


# Start Qt event loop unless running in interactive mode.
if __name__ == '__main__':
    v = GLPlotter()
