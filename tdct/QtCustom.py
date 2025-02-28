#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Custom Qt classes. Some widgets in QT Designer are promoted to these classes:
    - QTableView
    - QSortFilterProxyModel
    - QStandardItemModel
    - QGraphicsScene
    - a custom QWidget for Matplotlib integration

@Title         : QtCustom
@Project       : 3DCTv2
@Description   : Custom Qt classes
@Author        : Jan Arnold
@Email         : jan.arnold (at) coraxx.net
@Copyright     : Copyright (C) 2016  Jan Arnold
@License       : GPLv3 (see LICENSE file)
@Version       : 3DCT 2.3.0 module rev. 47
@Status        : stable
@Usage         : part of 3D Correlation Toolbox
@Python_version: 2.7.11 (converted for Python 3)
"""

import sys
import math
import os
import time
import re
import tempfile
import numpy as np

from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QSplitter, QAbstractItemView, QGraphicsView, QGraphicsSimpleTextItem, QGraphicsItem, QGraphicsLineItem, QGraphicsEllipseItem, QVBoxLayout, QWidget, QGraphicsPathItem

# Use Qt5Agg backend for matplotlib
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
from matplotlib import style
import matplotlib.patches as patches
from mpl_toolkits.axes_grid1 import make_axes_locatable

import beadPos
import clrmsg
import TDCT_debug

debug = TDCT_debug.debug
style.use('fivethirtyeight')

##############################
# QTableViewCustom

class QTableViewCustom(QtWidgets.QTableView):
    def __init__(self, parent=None):
        super().__init__(parent)
        # If parent is a QSplitter, get the main parent accordingly.
        if isinstance(self.parent(), QSplitter):
            self.mainParent = self.parent().parent().parent()
        self._drop = False
        # Enable Drag'n'Drop
        self.setDragDropOverwriteMode(False)
        self.setDragEnabled(True)
        self.setDragDropMode(QAbstractItemView.InternalMove)

    def mouseMoveEvent(self, event):
        super().mouseMoveEvent(event)
        if self._drop:
            self.updateItems()
            self._drop = False

    def dropEvent(self, event):
        self._drop = True
        super().dropEvent(event)

    def updateItems(self):
        items = [item for item in self._scene.items() if isinstance(item, QGraphicsEllipseItem)]
        if debug:
            print(clrmsg.DEBUG + "Update items check - Nr. of items/rows:", len(items), self._model.rowCount())
        if len(items) == self._model.rowCount():
            for row, item in enumerate(items):
                if debug:
                    print(clrmsg.DEBUG + "Row:", row, "|",
                          self._model.data(self._model.index(row, 0)),
                          self._model.data(self._model.index(row, 1)),
                          self._model.data(self._model.index(row, 2)))
                item.setPos(
                    float(self._model.data(self._model.index(row, 0))),
                    float(self._model.data(self._model.index(row, 1)))
                )
                self._scene.zValuesDict[item] = [
                    self._model.data(self._model.index(row, 2)),
                    self._model.itemFromIndex(self._model.index(row, 2)).foreground().color().getRgb()
                ]
        self.mainParent.colorModels()

    def showSelectedItem(self):
        indices = self.selectedIndexes()
        activeitems = []
        for item in self._scene.items():
            if isinstance(item, QGraphicsEllipseItem):
                item.setPen(self._scene.pen)
                activeitems.append(item)
        if indices:
            rows = {index.row() for index in indices}
            for row in rows:
                activeitems[row].setPen(QtGui.QPen(QtCore.Qt.green))

    def deleteItem(self):
        indices = self.selectedIndexes()
        activeitems = [item for item in self._scene.items() if isinstance(item, QGraphicsEllipseItem)]
        if indices:
            rows = {index.row() for index in indices}
            for row in rows:
                self._scene.removeItem(activeitems[row])
                self._scene.enumeratePoints()
            self._scene.itemsToModel()

    def contextMenuEvent(self, event):
        indices = self.selectedIndexes()
        if indices:
            cmDelete = QtGui.QAction('Delete', self)
            cmDelete.triggered.connect(self.deleteItem)

            # Actions for different layers
            cmGetZgaussL1 = QtGui.QAction('Get z gauss layer 1', self)
            cmGetZgaussL1.triggered.connect(lambda: self.getz(self.img1, gauss=True))
            cmGetZgaussOptL1 = QtGui.QAction('Get x,y,z gauss layer 1', self)
            cmGetZgaussOptL1.triggered.connect(lambda: self.getz(self.img1, gauss=True, optimize=True))
            cmGetZgaussL2 = QtGui.QAction('Get z gauss layer 2', self)
            cmGetZgaussL2.triggered.connect(lambda: self.getz(self.img2, gauss=True))
            cmGetZgaussOptL2 = QtGui.QAction('Get x,y,z gauss layer 2', self)
            cmGetZgaussOptL2.triggered.connect(lambda: self.getz(self.img2, gauss=True, optimize=True))
            cmGetZgaussL3 = QtGui.QAction('Get z gauss layer 3', self)
            cmGetZgaussL3.triggered.connect(lambda: self.getz(self.img3, gauss=True))
            cmGetZgaussOptL3 = QtGui.QAction('Get x,y,z gauss layer 3', self)
            cmGetZgaussOptL3.triggered.connect(lambda: self.getz(self.img3, gauss=True, optimize=True))

            if self.img1 is None:
                cmGetZgaussL1.setEnabled(False)
                cmGetZgaussOptL1.setEnabled(False)
            if self.img2 is None:
                cmGetZgaussL2.setEnabled(False)
                cmGetZgaussOptL2.setEnabled(False)
            if self.img3 is None:
                cmGetZgaussL3.setEnabled(False)
                cmGetZgaussOptL3.setEnabled(False)

            self.contextMenu = QtGui.QMenu(self)
            self.contextMenu.addAction(cmDelete)
            self.contextMenu.addSeparator()
            self.contextMenu.addAction(cmGetZgaussL1)
            self.contextMenu.addAction(cmGetZgaussL2)
            self.contextMenu.addAction(cmGetZgaussL3)
            self.contextMenu.addSeparator()
            self.contextMenu.addAction(cmGetZgaussOptL1)
            self.contextMenu.addAction(cmGetZgaussOptL2)
            self.contextMenu.addAction(cmGetZgaussOptL3)
            self.contextMenu.popup(QtGui.QCursor.pos())


##############################
# QStandardItemModelCustom

class QStandardItemModelCustom(QtGui.QStandardItemModel):
    def __init__(self, parent=None):
        super().__init__(parent)

    def dropMimeData(self, data, action, row, column, parent):
        return super().dropMimeData(data, action, row, 0, parent)


##############################
# QGraphicsSceneCustom

class QGraphicsSceneCustom(QtWidgets.QGraphicsScene):
    def __init__(self, parent=None, mainWidget=None, side=None, model=None):
        super().__init__(parent)
        self.mainWidget = mainWidget
        self.side = side
        self._model = model
        self.parent().setDragMode(QGraphicsView.NoDrag)
        self.pen = QtGui.QPen(QtCore.Qt.red)
        self.lastScreenPos = QtCore.QPoint(0, 0)
        self.lastScenePos = 0
        self.selectionmode = False
        self.pointidx = 1
        self.rotangle = 0
        self.markerSize = 10
        self.zValuesDict = {}

    def wheelEvent(self, event):
        scalingFactor = 1.15 if event.delta() > 0 else 1 / 1.15
        self.parent().scale(scalingFactor, scalingFactor)
        if (event.screenPos() - self.lastScreenPos).manhattanLength() > 25:
            self.parent().centerOn(event.scenePos().x(), event.scenePos().y())
            self.lastScenePos = event.scenePos()
        else:
            self.parent().centerOn(self.lastScenePos.x(), self.lastScenePos.y())
        self.lastScreenPos = event.screenPos()

    def mousePressEvent(self, event):
        modifiers = QtGui.QGuiApplication.keyboardModifiers()
        if event.button() == QtCore.Qt.LeftButton:
            if modifiers != QtCore.Qt.ControlModifier:
                self.parent().setDragMode(QGraphicsView.ScrollHandDrag)
                return
            else:
                self.parent().setDragMode(QGraphicsView.RubberBandDrag)
                self.selectionmode = True
                return
        elif event.button() == QtCore.Qt.RightButton:
            if self.mainWidget.checkBox_MIP.isChecked():
                self.addCircle(event.scenePos().x(), event.scenePos().y())
            else:
                self.addCircle(event.scenePos().x(), event.scenePos().y(), z=self.mainWidget.spinBox_slice.value())
        elif event.button() == QtCore.Qt.MiddleButton:
            item = self.itemAt(event.scenePos())
            if isinstance(item, QGraphicsEllipseItem):
                self.removeItem(item)
                self.enumeratePoints()
        self.itemsToModel()

    def mouseReleaseEvent(self, event):
        super().mouseReleaseEvent(event)
        if self.selectedItems() and not self.selectionmode:
            if debug:
                print(clrmsg.DEBUG + "New pos:", self.selectedItems()[0].x(), self.selectedItems()[0].y())
            if '{0:b}'.format(self.imagetype)[-1] == '0':
                for item in self.selectedItems():
                    if isinstance(item, QGraphicsEllipseItem):
                        self.zValuesDict[item] = [self.zValuesDict[item][0], (255, 190, 0)]
            self.clearSelection()
            self.itemsToModel()
        self.parent().setDragMode(QGraphicsView.NoDrag)
        self.selectionmode = False

    def keyPressEvent(self, event):
        if event.key() == QtCore.Qt.Key_Delete:
            for item in self.selectedItems():
                self.removeItem(item)
            self.itemsToModel()
        elif event.key() == QtCore.Qt.Key_Plus:
            self.parent().scale(1.15, 1.15)
        elif event.key() == QtCore.Qt.Key_Minus:
            self.parent().scale(1 / 1.15, 1 / 1.15)

    def addCircle(self, x, y, z=None):
        circle = self.addEllipse(-self.markerSize, -self.markerSize,
                                 self.markerSize * 2, self.markerSize * 2, self.pen)
        circle.setPos(x, y)
        circle.setFlag(QGraphicsItem.ItemIsMovable, True)
        circle.setFlag(QGraphicsItem.ItemIsSelectable, True)
        if self._z and z is None:
            self.zValuesDict[circle] = [0.0, (255, 190, 0)]
        elif self._z and z is not None:
            self.zValuesDict[circle] = [float(z), (0, 0, 0)]
        else:
            self.zValuesDict[circle] = [0.0, (0, 0, 0)]
        QGraphicsItem.stackBefore(circle, self.items()[-2])
        self.enumeratePoints()

    def addArrow(self, start, end, arrowangle=45, color=QtCore.Qt.red):
        dx, dy = end[0] - start[0], end[1] - start[1]
        length = math.hypot(dx, dy)
        angle = -math.asin(dy / length) if length != 0 else 0
        if dx < 0:
            angle = math.radians(180) - angle
        if debug:
            print(clrmsg.DEBUG + "Radians:", angle, "Degree", math.degrees(angle))
        path = QtGui.QPainterPath()
        path.moveTo(*start)
        path.lineTo(*end)
        path.arcMoveTo(end[0] - 0.25 * length, end[1] - 0.25 * length, 0.5 * length, 0.5 * length,
                        180 - arrowangle + math.degrees(angle))
        path.lineTo(*end)
        path.arcMoveTo(end[0] - 0.25 * length, end[1] - 0.25 * length, 0.5 * length, 0.5 * length,
                        180 + arrowangle + math.degrees(angle))
        path.lineTo(*end)
        self.addPath(path, QtGui.QPen(color))

    def deleteArrows(self):
        for item in self.items():
            if isinstance(item, QGraphicsPathItem):
                self.removeItem(item)

    def enumeratePoints(self):
        for item in self.items():
            if isinstance(item, (QGraphicsSimpleTextItem, QGraphicsLineItem)):
                self.removeItem(item)
        pointidx = 1
        for item in self.items():
            if isinstance(item, QGraphicsEllipseItem):
                item.setRect(-self.markerSize, -self.markerSize, self.markerSize * 2, self.markerSize * 2)
                nr = self.addSimpleText(str(pointidx), QtGui.QFont("Helvetica", int(1.5 * self.markerSize)))
                nr.setParentItem(item)
                nr.setRotation(-self.rotangle)
                radangle = math.radians(390 - self.rotangle)
                nr.setPos(math.cos(radangle) * self.markerSize, math.sin(radangle) * self.markerSize)
                nr.setBrush(QtCore.Qt.cyan)
                hline = self.addLine(-self.markerSize - 2, 0, self.markerSize + 2, 0)
                hline.setPen(QtGui.QPen(QtGui.QColor(255, 255, 255, 128)))
                hline.setParentItem(item)
                hline.setRotation(-self.rotangle)
                vline = self.addLine(0, -self.markerSize - 2, 0, self.markerSize + 2)
                vline.setPen(QtGui.QPen(QtGui.QColor(255, 255, 255, 128)))
                vline.setParentItem(item)
                vline.setRotation(-self.rotangle)
                pointidx += 1

    def itemsToModel(self):
        self._model.removeRows(0, self._model.rowCount())
        for item in self.items():
            if isinstance(item, QGraphicsEllipseItem):
                x_item = QtGui.QStandardItem(str(item.x()))
                y_item = QtGui.QStandardItem(str(item.y()))
                z_item = QtGui.QStandardItem(str(self.zValuesDict[item][0]))
                z_item.setForeground(QtGui.QColor(*self.zValuesDict[item][1]))
                for it in (x_item, y_item, z_item):
                    it.setFlags(it.flags() & ~QtCore.Qt.ItemIsDropEnabled)
                self._model.appendRow([x_item, y_item, z_item])
                self._model.setHeaderData(0, QtCore.Qt.Horizontal, 'x')
                self._model.setHeaderData(1, QtCore.Qt.Horizontal, 'y')
                self._model.setHeaderData(2, QtCore.Qt.Horizontal, 'z')
        self.mainWidget.colorModels()


##############################
# NumberSortModel

class NumberSortModel(QtCore.QSortFilterProxyModel):
    def lessThan(self, left, right):
        lvalue = left.data().toDouble()[0]
        rvalue = right.data().toDouble()[0]
        return lvalue < rvalue


##############################
# QStandardItemModelCustom

class QStandardItemModelCustom(QtGui.QStandardItemModel):
    def __init__(self, parent=None):
        super().__init__(parent)

    def dropMimeData(self, data, action, row, column, parent):
        return super().dropMimeData(data, action, row, 0, parent)


##############################
# MatplotlibWidgetCustom

class MatplotlibWidgetCustom(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._setup = False

    def setupScatterCanvas(self, width=5, height=5, dpi=72, toolbar=False):
        if not self._setup:
            self.figure = Figure(figsize=(width, height), dpi=dpi)
            self.canvas = FigureCanvas(self.figure)
            layout = QVBoxLayout()
            layout.addWidget(self.canvas)
            if toolbar:
                self.figure.set_figheight(height + 0.5)
                self.toolbar = NavigationToolbar(self.canvas, self)
                layout.addWidget(self.toolbar)
            self.setLayout(layout)
            self._setup = True
        else:
            self.clearAll()
            self._setup = False
            self.setupScatterCanvas(width, height, dpi, toolbar)

    def clearAll(self):
        QWidget().setLayout(self.layout())
        self._setup = False

    def scatterPlot(self, x='random', y='random', frame=False, framesize=None, xlabel="", ylabel=""):
        if (isinstance(x, str) and x == 'random') or (isinstance(y, str) and y == 'random'):
            x = np.random.randn(1000)
            y = np.random.randn(1000)
        self.subplotScatter = self.figure.add_subplot(111)
        self.subplotScatter.clear()
        self.subplotScatter.scatter(x, y)
        self.subplotScatter.set_aspect(1.0)
        limit = max(abs(x.min()), abs(x.max()), abs(y.min()), abs(y.max())) + 0.2
        self.subplotScatter.set_xlim(-limit, limit)
        self.subplotScatter.set_ylim(-limit, limit)
        self.subplotScatter.set_xlabel(xlabel)
        self.subplotScatter.set_ylabel(ylabel)
        self.subplotScatter.xaxis.set_label_coords(0.1, 0.08)
        self.subplotScatter.yaxis.set_label_coords(0.08, 0.12)
        self.subplotScatter.plot([0], '+', mew=1, ms=10, c="red")
        if frame and framesize is not None:
            self.subplotScatter.add_patch(patches.Rectangle(
                (-framesize * 0.5, -framesize * 0.5), framesize, framesize,
                fill=False, edgecolor="red"
            ))
        else:
            print("Please specify frame size in px as e.g. framesize=1.86")
        self.divider = make_axes_locatable(self.subplotScatter)
        self.axHistx = self.divider.append_axes("top", size="25%", pad=0.1)
        self.axHisty = self.divider.append_axes("right", size="25%", pad=0.1)
        for tl in self.axHistx.get_xticklabels():
            tl.set_visible(False)
        self.axHistx.set_yticks([])
        for tl in self.axHisty.get_yticklabels():
            tl.set_visible(False)
        self.axHisty.set_xticks([])
        binwidth = 0.25
        xymax = np.max(np.abs(x)) if np.max(np.abs(x)) > np.max(np.abs(y)) else np.max(np.abs(y))
        lim = (int(xymax / binwidth) + 1) * binwidth
        bins = np.arange(-lim, lim + binwidth, binwidth)
        self.axHistx.hist(x, bins=bins)
        self.axHisty.hist(y, bins=bins, orientation='horizontal')
        for tl in self.axHistx.get_xticklabels():
            tl.set_visible(False)
        self.canvas.draw()

    def xyPlot(self, *args, **kwargs):
        self.subplotXY = self.figure.add_subplot(111)
        clear = kwargs.pop('clear', False)
        if clear:
            self.subplotXY.clear()
        self.subplotXY.plot(*args, **kwargs)
        leg = self.subplotXY.legend(fontsize='small')
        leg.get_frame().set_alpha(0.5)
        self.canvas.draw()

    def matshowPlot(self, mat=None, contour=None, labelContour=''):
        n = len(self.figure.axes)
        if n < 2:
            for i in range(n):
                self.figure.axes[i].change_geometry(n + 1, 1, i + 1)
            self.figure.tight_layout()
            self.subplotMat = self.figure.add_subplot(n + 1, 1, n + 1)
        self.subplotMat.clear()
        self.subplotMat.matshow(mat)
        self.subplotMat.contour(contour, cmap='Greys', linewidths=1)
        self.subplotMat.grid(False)
        self.subplotMat.text(0.95, 0.03, labelContour, fontsize=12,
                             horizontalalignment='right', verticalalignment='bottom',
                             transform=self.figure.transFigure)
        self.subplotMat.set_anchor('W')
        self.canvas.draw()


##############################
# QLineEditFilePath

class QLineEditFilePath(QtWidgets.QLineEdit):
    def __init__(self, parent):
        super().__init__(parent)
        self.setDragEnabled(True)
        if sys.platform == 'darwin':
            try:
                import objc
                import CoreFoundation as CF
                if debug:
                    print(clrmsg.DEBUG + '"objc" and "CoreFoundation" import successful | Drag’n’Drop supported on macOS')
            except Exception as e:
                if debug:
                    print(clrmsg.ERROR + str(e))
                self.objc = None

    def dragEnterEvent(self, event):
        urls = event.mimeData().urls()
        if urls and urls[0].scheme() == 'file':
            event.acceptProposedAction()

    def dragMoveEvent(self, event):
        urls = event.mimeData().urls()
        if urls and urls[0].scheme() == 'file':
            event.acceptProposedAction()

    def dropEvent(self, event):
        data = event.mimeData()
        urls = data.urls()
        if urls and urls[0].scheme() == 'file':
            filepath = str(urls[0].path())[1:]
            if filepath.startswith('.file/id=') and hasattr(self, 'objc'):
                if debug:
                    print(clrmsg.DEBUG + 'File id bug detected, converting:', filepath)
                filepath = str(self.getUrlFromLocalFileID(urls[0]))
                if debug:
                    print(clrmsg.DEBUG + 'Converted to:', filepath)
            if sys.platform in ['linux2', 'darwin'] and not filepath.startswith('/'):
                self.setText('/' + filepath)
            else:
                self.setText(filepath)

    def getUrlFromLocalFileID(self, localFileID):
        localFileQString = QtCore.QString(localFileID.toLocalFile())
        relCFStringRef = CF.CFStringCreateWithCString(
            CF.kCFAllocatorDefault,
            localFileQString.toUtf8(),
            CF.kCFStringEncodingUTF8
        )
        relCFURL = CF.CFURLCreateWithFileSystemPath(
            CF.kCFAllocatorDefault,
            relCFStringRef,
            CF.kCFURLPOSIXPathStyle,
            False
        )
        absCFURL = CF.CFURLCreateFilePathURL(
            CF.kCFAllocatorDefault,
            relCFURL,
            objc.NULL
        )
        return QtCore.QUrl(str(absCFURL[0])).toLocalFile()
