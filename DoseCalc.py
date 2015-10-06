__author__ = 'Florian Laible'
#!/usr/bin/python
import numpy as np
import scipy as sci
import numpy.linalg as nplin
import scipy.special as sps
import scipy.ndimage as sciim

from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from PyQt5.QtCore import pyqtSlot, QTimer, QSocketNotifier, QAbstractTableModel, Qt, QVariant, QModelIndex
from PyQt5.QtWidgets import QApplication, QMainWindow, QSizePolicy, QVBoxLayout, QFileDialog, QInputDialog
from matplotlib.figure import Figure

import matplotlib
matplotlib.use('Qt5Agg')
from PyQt5 import QtGui, QtCore, QtWidgets
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure


from matplotlib import cm
import matplotlib.pyplot as plt
import seaborn as sb
import os
import re
import sys

filedir = 'C:/Users/Florian Laible/Desktop/'

sb.set_style("ticks")
sb.set_context("talk", font_scale=1.4)

alpha = 14
beta = 2180
Delta = 1
eta = 0.92

#fig = plt.figure()
#ax1 = fig.add_axes([0.1, 0.12, 0.8, 0.8])

AbstandX = 10
AbstandY = 10

from PyQt5 import uic
Ui_MainWindow = uic.loadUiType('DoseCalcGUI.ui')[0]

class DoseCalc(QMainWindow):
    _window_title = 'DoseCalcGUI'

    def __init__(self, parent=None):
        super(DoseCalc, self).__init__(parent)
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)

        self.fig = Figure()
        self.axes = self.fig.add_subplot(2, 2, 2)
        self.axes.hold(False)

        self.fig_erg = Figure()
        self.axes_erg = self.fig.add_subplot(2, 2, 1, projection='3d')
        #self.axes_erg.axis('off')
        self.axes_erg.hold(False)


        self.Canvas = FigureCanvas(self.fig)
        self.Canvas.setParent(self.ui.plot_widget)

        self.Canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.Canvas.updateGeometry()

        l = QVBoxLayout(self.ui.plot_widget)
        l.addWidget(self.Canvas)

        #plt.subplots_adjust(left=0.125, bottom=0.1, right=0.9, top=0.9, wspace=0.2, hspace=0.5)
        #left  = 0.125  # the left side of the subplots of the figure
        #right = 0.9    # the right side of the subplots of the figure
        #bottom = 0.1   # the bottom of the subplots of the figure
        #top = 0.9      # the top of the subplots of the figure
        #wspace = 0.2   # the amount of width reserved for blank space between subplots
        #hspace = 0.5   # the amount of height reserved for white space between subplots



# ##---------------- START button connect functions ----------
    @pyqtSlot()
    def on_Load_PB_clicked(self):
        buf = self._load_pic_from_file()
        if not buf is None:
            self.data = buf
            self._plot()

    @pyqtSlot()
    def on_Calc_PB_clicked(self):
        self._calc()

# ##---------------- END button connect functions ----------

    def _load_pic_from_file(self):
        save_dir = QFileDialog.getOpenFileName(self, "Load pic", filedir, 'PNG Files (*.png)')
        if len(save_dir[0])>1:
            save_dir = save_dir[0]
            data = sci.misc.imread(save_dir)
            data = data[:,:,0]
            for i in range(data.size):
                if data.item(i)<190:
                    data.itemset(i,1)
                else:
                    data.itemset(i,0)
            self.ASum = sum(sum(data))
            return np.array(data)
        return None



    def _calc(self):
        A = self.data
        x0 = np.zeros((self.ASum,1))
        y0 = np.zeros((self.ASum,1))
        Erg = np.zeros((self.ASum,1))

        x = np.linspace(-1*A.shape[0], 10*A.shape[0], 100)
        y = np.linspace(-1*A.shape[1], 10*A.shape[1], 100)

        self.xv, self.yv = np.meshgrid(x, y, sparse=False, indexing='ij')
        xvv = self.xv
        yvv = self.yv
        m = 0
        for k in range(A.shape[0]):
            for l in range(A.shape[1]):
                if A[k,l] == 1:
                    x0[m] = k * AbstandX
                    y0[m] = l * AbstandY
                    m += 1

        zges = np.zeros((len(self.xv),len(self.yv)))
        z = np.zeros((len(x0),len(y0)))

        def rechenknecht(xi,yi,xj,yj,Delta,alpha,beta,eta):
            z1 = 1/(np.pi*(1+eta))
            z2 = np.pi/(2*Delta**2)
            z3 = sps.erf((xi-xj+(Delta/2))/alpha)-sps.erf((xi-xj-(Delta/2))/alpha)
            z4 = sps.erf((yi-yj+(Delta/2))/alpha)-sps.erf((yi-yj-(Delta/2))/alpha)
            z5 = eta/(beta**2)
            z6 = np.exp(-(((xi-xj)**2)+((yi-yj)**2))/(beta**2))

            z = z1*((z2*(z3*z4))+(z5*z6))
            return(z)

        def rechenknecht2(x,y,xi,yi,Delta,alpha,beta,eta,D):
            z1 = 1/(np.pi*(1+eta))
            z2 = np.pi/(2*Delta**2)
            z3 = sps.erf((x-xi+(Delta/2))/alpha)-sps.erf((x-xi-(Delta/2))/alpha)
            z4 = sps.erf((y-yi+(Delta/2))/alpha)-sps.erf((y-yi-(Delta/2))/alpha)
            z5 = eta/(beta**2)
            z6 = np.exp(-(((x-xi)**2)+((y-yi)**2))/(beta**2))

            z = z1*((z2*(z3*z4))+(z5*z6))
            z = D * z
            return(z)

        BadCount = 1

        while BadCount != 0:
            BadCount = 0
            PS = np.zeros((len(x0),len(y0)))
            for k in range(len(x0)):
                for l in range(len(y0)):
                    PS[k, l] = rechenknecht(x0[k], y0[k], x0[l], y0[l],Delta,alpha,beta,eta)

            PSinv = nplin.inv(PS)
            Test = PS * PSinv
            Test = np.round(Test)
            EinerVek = np.ones(len(x0))
            Erg = nplin.solve(PS,EinerVek)
            #Erg = sp.linalg.solve_triangular(PS,EinerVek)
            #Erg = sp.sparse.linalg.spsolve(PS,EinerVek)
            #Erg = sp.sparse.linalg.minres(PS,EinerVek)

            Entries2Delete = []
            for m in range(len(x0)):
                if Erg[m] < 0:
                    Entries2Delete.append(m)
                    BadCount += 1
            x0neu = np.delete(x0,Entries2Delete)
            y0neu = np.delete(y0,Entries2Delete)
            x0 = x0neu
            y0 = y0neu

        for i in range(len(x0)):
            print(i)
            z = rechenknecht2(x0[i],y0[i],xvv,yvv,Delta,alpha,beta,eta,Erg[i])
            self.zges = zges + z

        print(Test)
        self._plot_erg()
        return None

    @pyqtSlot(np.ndarray)
    def _plot(self):
        self.axes.imshow(self.data)
        self.Canvas.draw()
        return True

    @pyqtSlot(np.ndarray)
    def _plot_erg(self):
        print('Check')
        self.axes_erg.plot_surface(self.xv, self.yv, self.zges, rstride=1, cstride=1, cmap=cm.coolwarm, linewidth=0, antialiased=False)
        self.Canvas.draw()
        return True



if __name__ == '__main__':
    app = QApplication(sys.argv)
    main = DoseCalc()
    main.show()
    sys.exit(app.exec_())
#
#
#
# #
# #
#
# #
# # x0 = np.zeros((ASum,1))
# # y0 = np.zeros((ASum,1))
# # Erg = np.zeros((ASum,1))
# # #print(A)
# # #ax1.imshow(A)
# # #plt.show()
# #
# # x = np.linspace(-1*A.shape[0], AbstandX*A.shape[0], 100)
# # y = np.linspace(-1*A.shape[1], AbstandY*A.shape[1], 100)
# #
# # xv, yv = np.meshgrid(x, y, sparse=False, indexing='ij')
# # m = 0
# # for k in range(A.shape[0]):
# #     for l in range(A.shape[1]):
# #         if A[k,l] == 1:
# #             x0[m] = k * AbstandX
# #             y0[m] = l * AbstandY
# #             m += 1
# #
# #
# # zges = np.zeros((len(xv),len(yv)))
# # z = np.zeros((len(x0),len(y0)))
# #
# # def rechenknecht(xi,yi,xj,yj,Delta,alpha,beta,eta):
# #     z1 = 1/(np.pi*(1+eta))
# #     z2 = np.pi/(2*Delta**2)
# #     z3 = sps.erf((xi-xj+(Delta/2))/alpha)-sps.erf((xi-xj-(Delta/2))/alpha)
# #     z4 = sps.erf((yi-yj+(Delta/2))/alpha)-sps.erf((yi-yj-(Delta/2))/alpha)
# #     z5 = eta/(beta**2)
# #     z6 = np.exp(-(((xi-xj)**2)+((yi-yj)**2))/(beta**2))
# #
# #     z = z1*((z2*(z3*z4))+(z5*z6))
# #     return(z)
# #
# # def rechenknecht2(x,y,xi,yi,Delta,alpha,beta,eta,D):
# #     z1 = 1/(np.pi*(1+eta))
# #     z2 = np.pi/(2*Delta**2)
# #     z3 = sps.erf((x-xi+(Delta/2))/alpha)-sps.erf((x-xi-(Delta/2))/alpha)
# #     z4 = sps.erf((y-yi+(Delta/2))/alpha)-sps.erf((y-yi-(Delta/2))/alpha)
# #     z5 = eta/(beta**2)
# #     z6 = np.exp(-(((x-xi)**2)+((y-yi)**2))/(beta**2))
# #
# #     z = z1*((z2*(z3*z4))+(z5*z6))
# #     z = D * z
# #     return(z)
# #
# # BadCount = 1
# #
# # while BadCount != 0:
# #     BadCount = 0
# #     print(len(x0))
# #     PS = np.zeros((len(x0),len(y0)))
# #     for k in range(len(x0)):
# #         for l in range(len(y0)):
# #             PS[k, l] = rechenknecht(x0[k], y0[k], x0[l], y0[l],Delta,alpha,beta,eta)
# #
# #     PSinv = nplin.inv(PS)
# #     Test = PS * PSinv
# #     Test = np.round(Test)
# #     EinerVek = np.ones(len(x0))
# #     Erg = nplin.solve(PS,EinerVek)
# #     #Erg = sp.linalg.solve_triangular(PS,EinerVek)
# #     #Erg = sp.sparse.linalg.spsolve(PS,EinerVek)
# #     #Erg = sp.sparse.linalg.minres(PS,EinerVek)
# #     print(Erg)
# #
# #     Entries2Delete = []
# #     for m in range(len(x0)):
# #         if Erg[m] < 0:
# #             Entries2Delete.append(m)
# #             BadCount += 1
# #
# #     x0neu = np.delete(x0,Entries2Delete)
# #     y0neu = np.delete(y0,Entries2Delete)
# #     x0 = x0neu
# #     y0 = y0neu
# #
# #
# #
# #
# # for i in range(len(x0)):
# #     z = rechenknecht2(x0[i],y0[i],xv,yv,Delta,alpha,beta,eta,Erg[i])
# #     zges = zges + z
# #
# # print(Test)
# #
# #
# # fig = plt.figure()
# # ax = fig.gca(projection='3d')
# # surf = ax.plot_surface(xv, yv, zges, rstride=1, cstride=1, cmap=cm.coolwarm, linewidth=0, antialiased=False)
# #
# # plt.show()
