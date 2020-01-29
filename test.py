
#
# from PyQt5 import QtCore, QtGui
#
# import pyqtgraph.opengl as gl
# import pyqtgraph as pg
# import numpy as np
# import sys
# import time
#
#
# class Visualizer(object):
#     def __init__(self):
#         self.traces = dict()
#         self.app = QtGui.QApplication(sys.argv)
#         self.w = gl.GLViewWidget()
#         self.w.opts['distance'] = 40
#         self.w.setWindowTitle('pyqtgraph example: GLLinePlotItem')
#         self.w.setGeometry(0, 110, 1920, 1080)
#         self.w.show()
#
#         self.phase = 0
#         self.lines = 50
#         self.points = 1000
#         self.y = np.linspace(-10, 10, self.lines)
#         self.x = np.linspace(-10, 10, self.points)
#
#         for i, line in enumerate(self.y):
#             y = np.array([line] * self.points)
#             d = np.sqrt(self.x ** 2 + y ** 2)
#             sine = 10 * np.sin(d + self.phase)
#             pts = np.vstack([self.x, y, sine]).transpose()
#             self.traces[i] = gl.GLLinePlotItem(
#                 pos=pts,
#                 color=pg.glColor((i, self.lines * 1.3)),
#                 width=(i + 1) / 10,
#                 antialias=True
#             )
#             self.w.addItem(self.traces[i])
#
#     def start(self):
#         if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
#             QtGui.QApplication.instance().exec_()
#
#     def set_plotdata(self, name, points, color, width):
#         self.traces[name].setData(pos=points, color=color, width=width)
#
#     def update(self):
#         stime = time.time()
#         for i, line in enumerate(self.y):
#             y = np.array([line] * self.points)
#
#             amp = 10 / (i + 1)
#             phase = self.phase * (i + 1) - 10
#             freq = self.x * (i + 1) / 10
#
#             sine = amp * np.sin(freq - phase)
#             pts = np.vstack([self.x, y, sine]).transpose()
#
#             self.set_plotdata(
#                 name=i, points=pts,
#                 color=pg.glColor((i, self.lines * 1.3)),
#                 width=3
#             )
#             self.phase -= .00002
#
#         print('{:.0f} FPS'.format(1 / (time.time() - stime)))
#
#     def animation(self):
#         timer = QtCore.QTimer()
#         timer.timeout.connect(self.update)
#         timer.start(10)
#         self.start()
#

# # Start event loop.
# if __name__ == '__main__':
# #     v = Visualizer()
# #     v.animation()
# #
# from sympy import *
# # x, y = Symbol('x'), Function('y')
# # sol = dsolve(y(x).diff(x) - 1/y(x), y(x))
# #
# # print(sol)
#
#
#
#
# t, k1, k2 = symbols('t, k1, k2')
#
# A, B = Function('A'), Function('B')
#
# eq1 = Eq(A(t).diff(t), -k1*A(t))
# eq2 = Eq(B(t).diff(t), k1*A(t) -k2*B(t))
#
# print(eq1, eq2)
#
# sol1 = dsolve(eq1, A(t), ics={A(0): 2})
# sol2 = dsolve(Eq(B(t).diff(t), k1*2*exp(-k1*t) - k2*B(t)), B(t), ics={B(0): 0})
#
# print(sol2)
#
#
#
# import os
#


from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
from matplotlib import cm

X, Y, Z = axes3d.get_test_data(0.2)

# Normalize to [0,1]
norm = plt.Normalize(Z.min(), Z.max())
colors = cm.seismic(norm(Z))
rcount, ccount, _ = colors.shape

fig = plt.figure()
ax = fig.gca(projection='3d')
surf = ax.plot_surface(X, Y, Z, rcount=rcount, ccount=ccount,
                       facecolors=colors, shade=False, linewidth=0.5)
surf.set_facecolor((0,0,0,0))
plt.show()

 # spocita pocet radku kodu v py souborech
# def countlines(start, lines=0, header=True, begin_start=None):
#     if header:
#         print('{:>10} |{:>10} | {:<20}'.format('ADDED', 'TOTAL', 'FILE'))
#         print('{:->11}|{:->11}|{:->20}'.format('', '', ''))
#
#     for thing in os.listdir(start):
#         thing = os.path.join(start, thing)
#         if 'pyqtgraph' in start and 'pyqtgraphmodif' not in start:
#             continue
#         if 'testdocstring.py' in thing or 'test.py' in thing:
#             continue
#
#         if os.path.isfile(thing):
#             if thing.endswith('.py') or thing.endswith('.pyw'):
#                 with open(thing, 'r') as f:
#                     newlines = f.readlines()
#                     newlines = len(newlines)
#                     lines += newlines
#
#                     if begin_start is not None:
#                         reldir_of_thing = '.' + thing.replace(begin_start, '')
#                     else:
#                         reldir_of_thing = '.' + thing.replace(start, '')
#
#                     print('{:>10} |{:>10} | {:<20}'.format(
#                             newlines, lines, reldir_of_thing))
#
#
#     for thing in os.listdir(start):
#         thing = os.path.join(start, thing)
#         if os.path.isdir(thing):
#             lines = countlines(thing, lines, header=False, begin_start=start)
#
#     return lines
#
#
# countlines(".\\")



#
#
#
#
#
# import numba as nb
# import numpy as np
#
# @nb.njit(fastmath=True)
# def sum(n=2, A0 = 0):
#     sum = 0
#     for i in range(0, n):
#         sum += np.power(10, -i/(n - 1) * A0)
#
#     return sum / n
#
#
# # print(sum(1000, 0.1))
#
# def intA(A, classical=False):
#     if classical:
#         return np.power(10, -A)
#     else:
#         result = (1 - np.power(10, -A)) / (A * np.log(10))
#         result[np.isnan(result)] = 1
#         return result
#
# from matplotlib import pyplot as plt
#
# x = np.linspace(0, 10, 10000)
# T = np.power(10, -x)
# Ttrue = intA(x, classical=False)
# # Ts10 = [sum(10, A) for A in x]
# # Ts100 = [sum(100, A) for A in x]
# # Ts1000 = [sum(1000, A) for A in x]
# # Ts10000 = [sum(10000, A) for A in x]
# # Ts100000 = [sum(100000, A) for A in x]
#
#
#
# plt.plot(x, T, label='$T=10^{-A}$')
# # plt.pyqtgraph-modif(x, Ts10, label='10')
# # plt.pyqtgraph-modif(x, Ts100, label='100')
# # plt.pyqtgraph-modif(x, Ts1000, label='1000')
# plt.plot(x, Ttrue, label='True')
# # plt.pyqtgraph-modif(x, Ts100000, label='100000')
#
#
# plt.legend()
# plt.show()


# import timeit
#
# a = np.linspace(0, 1000, num=1000)
# b = np.linspace(-1000, 0, num=10000)
#
# ar = np.zeros((1000, 10000), dtype=np.float64)
#
#
# @nb.jit(fastmath=True, nopython=True)
# def test(a, b, ar):
#     for i in range(a.shape[0]):
#         for j in range(b.shape[0]):
#             ar[i, j] = a[i] * b[j]
#
#     return ar
#
# # print('statrt')
# #
# # ar = test()
# #
# # print(ar)

























# import numpy as np



# for k1=k2
# B(x) = A2 * k1 * x * exp(-k1 * x)

# (k1 / (k2 - k1)) * (exp(-k1 * x) - exp(-k2 * x))
#
# def exp(x):
#     return np.exp(x)
#
#
# def ABC(x, A1, A2, A3, k1, k2, k3, y0):
#     _ = 1e-8
#     if abs(k1 - k2) < _ and abs(k1 - k3) < _ and abs(k2 - k3) < _:
#         print("k1=k2=k3")
#         return A1 * np.exp(-k1 * x) + A2 * k1 * x * exp(-k1 * x) + A3 * k1 * k1 * x * x * exp(-k1 * x) / 2 + y0
#
#     if abs(k1 - k2) < _:
#         print("k1=k2")
#         return A1 * np.exp(-k1 * x) + A2 * k1 * x * exp(-k1 * x) + A3 * k1 * k1 * exp(-x * (k3 + k1)) * (exp(k3 * x) * (k3 * x - k1 * x - 1) + exp(k1 * x)) / (k3 - k1) ** 2 + y0
#
#     if abs(k1 - k3) < _:
#         print("k1=k3")
#         return A1 * np.exp(-k1 * x) + A2 * (k1 / (k2 - k1)) * (exp(-k1 * x) - exp(-k2 * x)) + A3 * k2 * k1 * exp(-x * (k2 + k1)) * (exp(k2 * x) * (k2 * x - k1 * x - 1) + exp(k1 * x)) / (k2 - k1) ** 2 + y0
#
#     if abs(k2 - k3) < _:
#         print("k2=k3")
#         return A1 * np.exp(-k1 * x) + A2 * (k1 / (k2 - k1)) * (exp(-k1 * x) - exp(-k2 * x)) + A3 * k1 * k2 * exp(-x * (k1 + k2)) * (exp(k1 * x) * (k1 * x - k2 * x - 1) + exp(k2 * x)) / (k1 - k2) ** 2 + y0
#     print("different")
#     return A1 * np.exp(-k1 * x) + A2 * (k1 / (k2 - k1)) * (exp(-k1 * x) - exp(-k2 * x)) + A3 * k1 * k2 * exp(-x * (k1 + k2 + k3)) * ((k1 - k2) * exp(x * (k1 + k2)) + (k3 - k1) * exp(x * (k1 + k3)) + (k2 - k3) * exp(x * (k2 + k3))) / ((k1 - k2) * (k1 - k3) * (k2 - k3)) + y0
#
#
#
# k1, k2, k3 = 1.001, 1.001, 1.001
# y0 = 0.1
# A1 = 1
# A2 = 0.5
# A3 = 1
#
#
# print(ABC(1.5, A1, A2, A3, k1, k2, k3, y0))
#





# def ApC(x, A1, A2, k1, k2, k3, y0):
#     _ = 1e-8
#     if abs(k1 - k2) < _ and abs(k1 - k3) < _ and abs(k2 - k3) < _:
#         print("k1=k2=k3")
#         return A1 * np.exp(-k1 * x) + A2 * k1 * k1 * x * x * np.exp(-k1 * x) / 2 + y0
#
#     if abs(k1 - k2) < _:
#         print("k1=k2")
#         return A1 * np.exp(-k1 * x) + A2 * k1 * k1 * np.exp(-x * (k3 + k1)) * (np.exp(k3 * x) * (k3 * x - k1 * x - 1) + np.exp(k1 * x)) / (k3 - k1) ** 2 + y0
#
#     if abs(k1 - k3) < _:
#         print("k1=k3")
#         return A1 * np.exp(-k1 * x) + A2 * k2 * k1 * np.exp(-x * (k2 + k1)) * (np.exp(k2 * x) * (k2 * x - k1 * x - 1) + np.exp(k1 * x)) / (k2 - k1) ** 2 + y0
#
#     if abs(k2 - k3) < _:
#         print("k2=k3")
#         return A1 * np.exp(-k1 * x) + A2 * k1 * k2 * np.exp(-x * (k1 + k2)) * (np.exp(k1 * x) * (k1 * x - k2 * x - 1) + np.exp(k2 * x)) / (k1 - k2) ** 2 + y0
#     print("different")
#     return A1 * np.exp(-k1 * x) + A2 * k1 * k2 * np.exp(-x * (k1 + k2 + k3)) * ((k1 - k2) * np.exp(x * (k1 + k2)) + (k3 - k1) * np.exp(x * (k1 + k3)) + (k2 - k3) * np.exp(x * (k2 + k3))) / ((k1 - k2) * (k1 - k3) * (k2 - k3)) + y0
#
#
#
#
#
# def C(x, A, k1, k2, k3, y0):
#     _ = 1e-8
#     if abs(k1 - k2) < _ and abs(k1 - k3) < _ and abs(k2 - k3) < _:
#         print("k1=k2=k3")
#         return A * k1 * k1 * x * x * np.exp(-k1 * x) / 2 + y0
#
#     if abs(k1 - k2) < _:
#         print("k1=k2")
#         return A * k1 * k1 * np.exp(-x * (k3 + k1)) * (np.exp(k3 * x) * (k3 * x - k1 * x - 1) + np.exp(k1 * x)) / (k3 - k1) ** 2 + y0
#
#     if abs(k1 - k3) < _:
#         print("k1=k3")
#         return A * k2 * k1 * np.exp(-x * (k2 + k1)) * (np.exp(k2 * x) * (k2 * x - k1 * x - 1) + np.exp(k1 * x)) / (k2 - k1) ** 2 + y0
#
#     if abs(k2 - k3) < _:
#         print("k2=k3")
#         return A * k1 * k2 * np.exp(-x * (k1 + k2)) * (np.exp(k1 * x) * (k1 * x - k2 * x - 1) + np.exp(k2 * x)) / (k1 - k2) ** 2 + y0
#     print("different")
#     return A * k1 * k2 * np.exp(-x * (k1 + k2 + k3)) * ((k1 - k2) * np.exp(x * (k1 + k2)) + (k3 - k1) * np.exp(x * (k1 + k3)) + (k2 - k3) * np.exp(x * (k2 + k3))) / ((k1 - k2) * (k1 - k3) * (k2 - k3)) + y0



# def BpC(x, A1, A2, k1, k2, k3, y0):
#     _ = 1e-8
#     if abs(k1 - k2) < _ and abs(k1 - k3) < _ and abs(k2 - k3) < _:
#         #print("k1=k2=k3")
#         return A1 * k1 * x * exp(-k1 * x) + A2 * k1 * k1 * x * x * exp(-k1 * x) / 2 + y0
#
#     if abs(k1 - k2) < _:
#         #print("k1=k2")
#         return A1 * k1 * x * exp(-k1 * x) + A2 * k1 * k1 * exp(-x * (k3 + k1)) * (exp(k3 * x) * (k3 * x - k1 * x - 1) + exp(k1 * x)) / (k3 - k1) ** 2 + y0
#
#     if abs(k1 - k3) < _:
#         #print("k1=k3")
#         return A1 * (k1 / (k2 - k1)) * (exp(-k1 * x) - exp(-k2 * x)) + A2 * k2 * k1 * exp(-x * (k2 + k1)) * (exp(k2 * x) * (k2 * x - k1 * x - 1) + exp(k1 * x)) / (k2 - k1) ** 2 + y0
#
#     if abs(k2 - k3) < _:
#         #print("k2=k3")
#         return A1 * (k1 / (k2 - k1)) * (exp(-k1 * x) - exp(-k2 * x)) + A2 * k1 * k2 * exp(-x * (k1 + k2)) * (exp(k1 * x) * (k1 * x - k2 * x - 1) + exp(k2 * x)) / (k1 - k2) ** 2 + y0
#     #print("different")
#     return A1 * (k1 / (k2 - k1)) * (exp(-k1 * x) - exp(-k2 * x)) + A2 * k1 * k2 * exp(-x * (k1 + k2 + k3)) * ((k1 - k2) * exp(x * (k1 + k2)) + (k3 - k1) * exp(x * (k1 + k3)) + (k2 - k3) * exp(x * (k2 + k3))) / ((k1 - k2) * (k1 - k3) * (k2 - k3)) + y0
#





# import sys
# import cProfile
# import pstats
#
# from PyQt5.QtCore import Qt, QAbstractItemModel, QVariant, QModelIndex
# from PyQt5.QtWidgets import QApplication, QTreeView
#
# # 200 root nodes with 10 subnodes each
#
# class TreeNode(object):
#     def __init__(self, parent, row, text):
#         self.parent = parent
#         self.row = row
#         self.text = text
#         if parent is None: # root node, create subnodes
#             self.children = [TreeNode(self, i, str(i)) for i in range(10)]
#         else:
#             self.children = []
#
# class TreeModel(QAbstractItemModel):
#     def __init__(self):
#         QAbstractItemModel.__init__(self)
#         self.nodes = [TreeNode(None, i, str(i)) for i in range(200)]
#
#     def index(self, row, column, parent):
#         if not self.nodes:
#             return QModelIndex()
#         if not parent.isValid():
#             return self.createIndex(row, column, self.nodes[row])
#         node = parent.internalPointer()
#         return self.createIndex(row, column, node.children[row])
#
#     def parent(self, index):
#         if not index.isValid():
#             return QModelIndex()
#         node = index.internalPointer()
#         if node.parent is None:
#             return QModelIndex()
#         else:
#             return self.createIndex(node.parent.row, 0, node.parent)
#
#     def columnCount(self, parent):
#         return 1
#
#     def rowCount(self, parent):
#         if not parent.isValid():
#             return len(self.nodes)
#         node = parent.internalPointer()
#         return len(node.children)
#
#     def data(self, index, role):
#         if not index.isValid():
#             return QVariant()
#         node = index.internalPointer()
#         if role == Qt.DisplayRole:
#             return QVariant(node.text)
#         return QVariant()
#
#
# app = QApplication(sys.argv)
# treemodel = TreeModel()
# treeview = QTreeView()
# treeview.setSelectionMode(QTreeView.ExtendedSelection)
# treeview.setSelectionBehavior(QTreeView.SelectRows)
# treeview.setModel(treemodel)
# treeview.expandAll()
# treeview.show()
# cProfile.run('app.exec_()', 'profdata')
# p = pstats.Stats('profdata')
# p.sort_stats('time').print_stats()
