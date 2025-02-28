#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Extracting 2D and 3D points with subsequent 2D to 3D correlation.
This module can be run as a standalone python application, but is best paired
with the preceding data processing (cubing voxels, merging single image files
into one single stack file, etc.).

@Title          : TDCT_correlation
@Project        : 3DCTv2
@Description    : Extracting 2D and 3D points for 2D to 3D correlation
@Author         : Jan Arnold
@Email          : jan.arnold (at) coraxx.net
@Copyright      : Copyright (C) 2016  Jan Arnold
@License        : GPLv3 (see LICENSE file)
@Version        : 3DCT 2.3.0 module rev. 30
@Status         : stable
@Usage          : Part of 3D Correlation Toolbox
@Python_version : 2.7.11 (Converted to work with Python 3)
"""

import sys
import os
import time
import re
import tempfile

from PyQt5 import QtCore, QtGui, uic, QtWidgets
from PyQt5.QtWidgets import QSplashScreen, QApplication, QGraphicsPixmapItem, QFileDialog, QGraphicsItem, QMessageBox
import numpy as np
import cv2
import tifffile as tf
import qimage2ndarray

# Custom modules
from tdct import clrmsg, TDCT_debug, QtCustom, csvHandler, correlation

__version__ = 'v2.3.0'

# Set execution directory
if getattr(sys, 'frozen', False):
    execdir = sys._MEIPASS  # running as a bundle (pyinstaller)
else:
    execdir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(execdir)

qtCreatorFile_main = os.path.join(execdir, "TDCT_correlation.ui")
Ui_WidgetWindow, QtBaseClass = uic.loadUiType(qtCreatorFile_main)

debug = TDCT_debug.debug
if debug:
    print(clrmsg.DEBUG + "Execdir =", execdir)


class MainWidget(QtWidgets.QMainWindow, Ui_WidgetWindow):
    def __init__(self, parent=None, leftImage=None, rightImage=None, workingdir=None):
        if debug:
            print(clrmsg.DEBUG + "Debug messages enabled")
        super(MainWidget, self).__init__()
        self.setupUi(self)

        self.parent = parent
        self.counter = 0  # Loop counter for testing
        self.refreshUI = QtGui.QGuiApplication.processEvents
        self.currentFocusedWidgetName = QApplication.focusWidget()

        self.workingdir = workingdir if workingdir is not None else execdir
        self.lineEdit_workingDir.setText(self.workingdir)

        # Define stylesheet colors
        self.stylesheet_orange = "color: rgb(255, 120, 0);"
        self.stylesheet_green = "color: rgb(0, 200, 0);"
        self.stylesheet_blue = "color: rgb(0, 190, 255);"
        self.stylesheet_red = "color: rgb(255, 0, 0);"

        # Marker and POI colors
        self.markerColor = (0, 255, 0)
        self.poiColor = (0, 0, 255)

        # TableViews and models
        self.modelLleft = QtCustom.QStandardItemModelCustom(self)
        self.tableView_left.setModel(self.modelLleft)
        self.modelLleft.tableview = self.tableView_left

        self.modelRight = QtCustom.QStandardItemModelCustom(self)
        self.tableView_right.setModel(self.modelRight)
        self.modelRight.tableview = self.tableView_right

        self.modelResults = QtGui.QStandardItemModel(self)
        self.modelResultsProxy = QtCustom.NumberSortModel()
        self.modelResultsProxy.setSourceModel(self.modelResults)
        self.tableView_results.setModel(self.modelResultsProxy)

        # Store parameters for resizing and images
        self.parent = parent
        self.size = 500
        self.leftImage = leftImage
        self.rightImage = rightImage

        # Initialize parameters for left image
        self.selectedLayer_left = 1
        self.brightness_left_layer1 = 0
        self.brightness_left_layer2 = 0
        self.brightness_left_layer3 = 0
        self.contrast_left_layer1 = 10
        self.contrast_left_layer2 = 10
        self.contrast_left_layer3 = 10
        self.slice_left = 0
        self.mipCHKbox_left = True
        self.layer1CHKbox_left = True
        self.layer2CHKbox_left = False
        self.layer3CHKbox_left = False
        self.layer1Color_left = 0
        self.layer2Color_left = 0
        self.layer3Color_left = 0
        self.layer1CustomColor_left = [255, 0, 255]
        self.layer2CustomColor_left = [255, 0, 255]
        self.layer3CustomColor_left = [255, 0, 255]
        self.img_left_overlay = None
        self.img_left_layer2 = None
        self.imgstack_left_layer2 = None
        self.img_left_layer3 = None
        self.imgstack_left_layer3 = None

        # Initialize parameters for right image
        self.selectedLayer_right = 1
        self.brightness_right_layer1 = 0
        self.brightness_right_layer2 = 0
        self.brightness_right_layer3 = 0
        self.contrast_right_layer1 = 10
        self.contrast_right_layer2 = 10
        self.contrast_right_layer3 = 10
        self.slice_right = 0
        self.mipCHKbox_right = True
        self.layer1CHKbox_right = True
        self.layer2CHKbox_right = False
        self.layer3CHKbox_right = False
        self.layer1Color_right = 0
        self.layer2Color_right = 0
        self.layer3Color_right = 0
        self.layer1CustomColor_right = [255, 0, 255]
        self.layer2CustomColor_right = [255, 0, 255]
        self.layer3CustomColor_right = [255, 0, 255]
        self.img_right_overlay = None
        self.img_right_layer2 = None
        self.imgstack_right_layer2 = None
        self.img_right_layer3 = None
        self.imgstack_right_layer3 = None

        # Connect image load/reset buttons
        self.toolButton_loadLeftImage.clicked.connect(self.openImageLeft)
        self.toolButton_loadRightImage.clicked.connect(self.openImageRight)
        self.toolButton_resetLeftImage.clicked.connect(lambda: self.resetImageLeft(img=None))
        self.toolButton_resetRightImage.clicked.connect(lambda: self.resetImageRight(img=None))

        if leftImage is not None and rightImage is not None:
            self.initImageLeft()
            self.initImageRight()

        # Connect model and table signals
        self.modelLleft.itemChanged.connect(self.tableView_left.updateItems)
        self.modelRight.itemChanged.connect(self.tableView_right.updateItems)
        self.tableView_left.selectionModel().selectionChanged.connect(self.tableView_left.showSelectedItem)
        self.tableView_right.selectionModel().selectionChanged.connect(self.tableView_right.showSelectedItem)
        self.tableView_results.selectionModel().selectionChanged.connect(self.showSelectedResidual)
        self.tableView_results.doubleClicked.connect(lambda: self.showSelectedResidual(doubleclick=True))

        # Connect spinboxes and checkboxes
        self.spinBox_rot.valueChanged.connect(self.rotateImage)
        self.spinBox_markerSize.valueChanged.connect(self.changeMarkerSize)
        self.spinBox_slice.valueChanged.connect(self.selectSlice)
        self.doubleSpinBox_scatterPlotFrameSize.valueChanged.connect(
            lambda: self.displayResults(
                frame=self.checkBox_scatterPlotFrame.isChecked(),
                framesize=self.doubleSpinBox_scatterPlotFrameSize.value())
        )
        self.checkBox_scatterPlotFrame.stateChanged.connect(
            lambda: self.displayResults(
                frame=self.checkBox_scatterPlotFrame.isChecked(),
                framesize=self.doubleSpinBox_scatterPlotFrameSize.value())
        )
        self.checkBox_resultsAbsolute.stateChanged.connect(
            lambda: self.displayResults(
                frame=self.checkBox_scatterPlotFrame.isChecked(),
                framesize=self.doubleSpinBox_scatterPlotFrameSize.value())
        )
        self.checkBox_MIP.stateChanged.connect(self.selectSlice)
        self.checkBox_layer1.toggled.connect(lambda: self.layerCtrl('layer1'))
        self.checkBox_layer2.toggled.connect(lambda: self.layerCtrl('layer2'))
        self.checkBox_layer3.toggled.connect(lambda: self.layerCtrl('layer3'))

        # Connect comboboxes and radio buttons
        self.comboBox_channelColorLayer1.currentIndexChanged.connect(self.changeColorChannel)
        self.comboBox_channelColorLayer2.currentIndexChanged.connect(self.changeColorChannel)
        self.comboBox_channelColorLayer3.currentIndexChanged.connect(self.changeColorChannel)
        self.radioButton_layer1.clicked.connect(self.setSliders)
        self.radioButton_layer2.clicked.connect(self.setSliders)
        self.radioButton_layer3.clicked.connect(self.setSliders)

        # Connect other buttons
        self.toolButton_rotcw.clicked.connect(lambda: self.rotateImage45(direction='cw'))
        self.toolButton_rotccw.clicked.connect(lambda: self.rotateImage45(direction='ccw'))
        self.toolButton_brightness_reset.clicked.connect(lambda: self.horizontalSlider_brightness.setValue(0))
        self.toolButton_contrast_reset.clicked.connect(lambda: self.horizontalSlider_contrast.setValue(10))
        self.toolButton_importPoints.clicked.connect(self.importPoints)
        self.toolButton_exportPoints.clicked.connect(self.exportPoints)
        self.toolButton_selectWorkingDir.clicked.connect(self.selectWorkingDir)
        self.toolButton_selectMarkerColor.clicked.connect(self.getMarkerColor)
        self.toolButton_selectPoiColor.clicked.connect(self.getPoiColor)
        self.toolButton_saveImage_left.clicked.connect(lambda: self.displayImage(side='left', save=True))
        self.toolButton_saveImage_right.clicked.connect(lambda: self.displayImage(side='right', save=True))
        self.toolButton_loadLayer2.clicked.connect(lambda: self.layerCtrl('layer2', load=True))
        self.toolButton_loadLayer3.clicked.connect(lambda: self.layerCtrl('layer3', load=True))
        self.commandLinkButton_correlate.clicked.connect(self.correlate)

        # Connect sliders
        self.horizontalSlider_brightness.valueChanged.connect(self.setBrightCont)
        self.horizontalSlider_contrast.valueChanged.connect(self.setBrightCont)

        # Connect focus change events
        QApplication.instance().focusChanged.connect(self.changedFocusSlot)

        # Pass models and scenes to tableViews
        self.tableView_left._model = self.modelLleft
        self.tableView_right._model = self.modelRight
        self.tableView_left._scene = self.sceneLeft
        self.tableView_right._scene = self.sceneRight

        self.tableView_results.setContextMenuPolicy(QtCore.Qt.CustomContextMenu)
        self.tableView_results.customContextMenuRequested.connect(self.cmTableViewResults)
        self.lineEdit_workingDir.textChanged.connect(self.updateWorkingDir)

        self.activateWindow()

    def keyPressEvent(self, event):
        """Filter key press events (e.g. Delete key to remove table rows)."""
        if event.key() == QtCore.Qt.Key_Delete:
            if self.currentFocusedWidgetName == 'tableView_left':
                if debug:
                    print(clrmsg.DEBUG + "Deleting item(s) on the left side")
                self.tableView_left.deleteItem()
                self.tableView_left.updateItems()
            elif self.currentFocusedWidgetName == 'tableView_right':
                if debug:
                    print(clrmsg.DEBUG + "Deleting item(s) on the right side")
                self.tableView_right.deleteItem()
                self.tableView_right.updateItems()

    def closeEvent(self, event):
        """Warn the user before exiting."""
        quit_msg = ("Are you sure you want to exit the\n"
                    "3DCT Correlation?\n\n"
                    "Unsaved data will be lost!")
        reply = QMessageBox.question(
            self, 'Message', quit_msg,
            QMessageBox.Yes, QMessageBox.No
        )
        if reply == QMessageBox.Yes:
            event.accept()
            if self.parent:
                self.parent.cleanUp()
                self.parent.exitstatus = 0
        else:
            event.ignore()
            if self.parent:
                self.parent.exitstatus = 1

    def selectWorkingDir(self):
        path = str(QFileDialog.getExistingDirectory(self, "Select working directory", self.workingdir))
        self.activateWindow()
        if path:
            workingdir = self.checkWorkingDirPrivileges(path)
            if workingdir:
                self.workingdir = workingdir
            self.lineEdit_workingDir.setText(self.workingdir)

    def updateWorkingDir(self):
        if os.path.isdir(self.lineEdit_workingDir.text()):
            workingdir = self.checkWorkingDirPrivileges(self.lineEdit_workingDir.text())
            if workingdir:
                self.workingdir = workingdir
            self.lineEdit_workingDir.setText(self.workingdir)
            print("updated working dir to:", self.workingdir)
        else:
            self.lineEdit_workingDir.setText(self.workingdir)
            print(clrmsg.ERROR + "Dropped object is not a valid path. Returning to {0} as working directory.".format(self.workingdir))

    def checkWorkingDirPrivileges(self, path):
        try:
            testfile = tempfile.TemporaryFile(dir=path)
            testfile.close()
            return path
        except Exception:
            QMessageBox.critical(
                self, "Warning",
                "I cannot write to this folder: {0}\nFalling back to {1} as the working directory".format(path, self.workingdir)
            )
            return None

    def changedFocusSlot(self, former, current):
        if debug:
            print(clrmsg.DEBUG + "focus changed from/to:",
                  former.objectName() if former else former,
                  current.objectName() if current else current)
        if current:
            self.currentFocusedWidgetName = current.objectName()
            self.currentFocusedWidget = current
        if former:
            self.formerFocusedWidgetName = former.objectName()
            self.formerFocusedWidget = former

        # Update label and border based on focused widget
        if self.currentFocusedWidgetName not in [
            'spinBox_rot', 'spinBox_markerSize', 'spinBox_slice',
            'horizontalSlider_brightness', 'horizontalSlider_contrast',
            'doubleSpinBox_custom_rot_center_x', 'doubleSpinBox_custom_rot_center_y', 'doubleSpinBox_custom_rot_center_z',
            'checkBox_MIP', 'checkBox_layer1', 'checkBox_layer2', 'checkBox_layer3',
            'comboBox_channelColorLayer1', 'comboBox_channelColorLayer2', 'comboBox_channelColorLayer3',
            'radioButton_layer1', 'radioButton_layer2', 'radioButton_layer3', ''
        ]:
            if self.currentFocusedWidgetName not in ['graphicsView_left', 'graphicsView_right']:
                self.borderControl(None, 1)
                self.label_selimg.setStyleSheet(self.stylesheet_orange)
                self.label_selimg.setText('none')
                self.label_markerSizeNano.setText('')
                self.label_markerSizeNanoUnit.setText('')
                self.label_imgpxsize.setText('')
                self.label_imgpxsizeUnit.setText('')
                self.label_imagetype.setText('')
                self.ctrlEnDisAble(False)
            elif self.currentFocusedWidgetName == 'graphicsView_left':
                self.borderControl(current)
                self.label_selimg.setStyleSheet(self.stylesheet_green)
                self.label_selimg.setText('left')
                self.label_imagetype.setStyleSheet(self.stylesheet_green)
                if '{0:b}'.format(self.sceneLeft.imagetype)[-1] == '1':
                    self.label_imagetype.setText('(2D)')
                    self.widget_sliceSelector.setVisible(False)
                else:
                    self.label_imagetype.setText('(3D)')
                    self.widget_sliceSelector.setVisible(True)
                self.ctrlEnDisAble(True)
                if not self.mipCHKbox_left:
                    self.spinBox_slice.setEnabled(True)
            elif self.currentFocusedWidgetName == 'graphicsView_right':
                self.borderControl(current)
                self.label_selimg.setStyleSheet(self.stylesheet_blue)
                self.label_selimg.setText('right')
                self.label_imagetype.setStyleSheet(self.stylesheet_blue)
                if '{0:b}'.format(self.sceneRight.imagetype)[-1] == '1':
                    self.label_imagetype.setText('(2D)')
                    self.widget_sliceSelector.setVisible(False)
                else:
                    self.label_imagetype.setText('(3D)')
                    self.widget_sliceSelector.setVisible(True)
                self.ctrlEnDisAble(True)
                if not self.mipCHKbox_right:
                    self.spinBox_slice.setEnabled(True)

        # Update selected table label
        if self.currentFocusedWidgetName not in ['tableView_left', 'tableView_right']:
            self.borderControl(None, 2)
            self.label_selectedTable.setStyleSheet(self.stylesheet_orange)
            self.label_selectedTable.setText('none')
        elif self.currentFocusedWidgetName == 'tableView_left':
            self.borderControl(current)
            self.label_selectedTable.setStyleSheet(self.stylesheet_green)
            self.label_selectedTable.setText('left')
            self.ctrlEnDisAble(False)
        elif self.currentFocusedWidgetName == 'tableView_right':
            self.borderControl(current)
            self.label_selectedTable.setStyleSheet(self.stylesheet_blue)
            self.label_selectedTable.setText('right')
            self.ctrlEnDisAble(False)

        # Block signals while updating slider values
        self.horizontalSlider_brightness.blockSignals(True)
        self.horizontalSlider_contrast.blockSignals(True)
        self.spinBox_slice.blockSignals(True)
        self.checkBox_MIP.blockSignals(True)
        self.checkBox_layer1.blockSignals(True)
        self.checkBox_layer2.blockSignals(True)
        self.checkBox_layer3.blockSignals(True)
        self.comboBox_channelColorLayer1.blockSignals(True)
        self.comboBox_channelColorLayer2.blockSignals(True)
        self.comboBox_channelColorLayer3.blockSignals(True)

        if self.currentFocusedWidgetName == 'graphicsView_left':
            self.spinBox_rot.setValue(self.sceneLeft.rotangle)
            self.spinBox_markerSize.setValue(self.sceneLeft.markerSize)
            self.spinBox_slice.setValue(self.slice_left)
            self.checkBox_MIP.setChecked(self.mipCHKbox_left)
            self.checkBox_layer1.setChecked(self.layer1CHKbox_left)
            self.comboBox_channelColorLayer1.setEnabled(self.layer1CHKbox_left)
            self.comboBox_channelColorLayer1.setCurrentIndex(self.layer1Color_left)
            self.checkBox_layer2.setChecked(self.layer2CHKbox_left)
            self.comboBox_channelColorLayer2.setEnabled(self.layer2CHKbox_left)
            self.comboBox_channelColorLayer2.setCurrentIndex(self.layer2Color_left)
            self.checkBox_layer3.setChecked(self.layer3CHKbox_left)
            self.comboBox_channelColorLayer3.setEnabled(self.layer3CHKbox_left)
            self.comboBox_channelColorLayer3.setCurrentIndex(self.layer3Color_left)
            if self.selectedLayer_left == 1:
                self.radioButton_layer1.setChecked(True)
            elif self.selectedLayer_left == 2:
                self.radioButton_layer2.setChecked(True)
            elif self.selectedLayer_left == 3:
                self.radioButton_layer3.setChecked(True)
            self.radioButton_layer1.setEnabled(self.layer1CHKbox_left)
            self.radioButton_layer2.setEnabled(self.layer2CHKbox_left)
            self.radioButton_layer3.setEnabled(self.layer3CHKbox_left)
            if self.radioButton_layer1.isChecked():
                self.horizontalSlider_brightness.setValue(self.brightness_left_layer1)
                self.horizontalSlider_contrast.setValue(self.contrast_left_layer1)
            elif self.radioButton_layer2.isChecked():
                self.horizontalSlider_brightness.setValue(self.brightness_left_layer2)
                self.horizontalSlider_contrast.setValue(self.contrast_left_layer2)
            elif self.radioButton_layer3.isChecked():
                self.horizontalSlider_brightness.setValue(self.brightness_left_layer3)
                self.horizontalSlider_contrast.setValue(self.contrast_left_layer3)
            self.label_imgpxsize.setText(str(self.sceneLeft.pixelSize))
            if self.sceneLeft.pixelSize:
                self.label_imgpxsizeUnit.setText('um')
            else:
                self.label_imgpxsizeUnit.setText('')
        elif self.currentFocusedWidgetName == 'graphicsView_right':
            self.spinBox_rot.setValue(self.sceneRight.rotangle)
            self.spinBox_markerSize.setValue(self.sceneRight.markerSize)
            self.spinBox_slice.setValue(self.slice_right)
            self.checkBox_MIP.setChecked(self.mipCHKbox_right)
            self.checkBox_layer1.setChecked(self.layer1CHKbox_right)
            self.comboBox_channelColorLayer1.setEnabled(self.layer1CHKbox_right)
            self.comboBox_channelColorLayer1.setCurrentIndex(self.layer1Color_right)
            self.checkBox_layer2.setChecked(self.layer2CHKbox_right)
            self.comboBox_channelColorLayer2.setEnabled(self.layer2CHKbox_right)
            self.comboBox_channelColorLayer2.setCurrentIndex(self.layer2Color_right)
            self.checkBox_layer3.setChecked(self.layer3CHKbox_right)
            self.comboBox_channelColorLayer3.setEnabled(self.layer3CHKbox_right)
            self.comboBox_channelColorLayer3.setCurrentIndex(self.layer3Color_right)
            if self.selectedLayer_right == 1:
                self.radioButton_layer1.setChecked(True)
            elif self.selectedLayer_right == 2:
                self.radioButton_layer2.setChecked(True)
            elif self.selectedLayer_right == 3:
                self.radioButton_layer3.setChecked(True)
            self.radioButton_layer1.setEnabled(self.layer1CHKbox_right)
            self.radioButton_layer2.setEnabled(self.layer2CHKbox_right)
            self.radioButton_layer3.setEnabled(self.layer3CHKbox_right)
            if self.radioButton_layer1.isChecked():
                self.horizontalSlider_brightness.setValue(self.brightness_right_layer1)
                self.horizontalSlider_contrast.setValue(self.contrast_right_layer1)
            elif self.radioButton_layer2.isChecked():
                self.horizontalSlider_brightness.setValue(self.brightness_right_layer2)
                self.horizontalSlider_contrast.setValue(self.contrast_right_layer2)
            elif self.radioButton_layer3.isChecked():
                self.horizontalSlider_brightness.setValue(self.brightness_right_layer3)
                self.horizontalSlider_contrast.setValue(self.contrast_right_layer3)
            self.label_imgpxsize.setText(str(self.sceneRight.pixelSize))
            if self.sceneRight.pixelSize:
                self.label_imgpxsizeUnit.setText('um')
            else:
                self.label_imgpxsizeUnit.setText('')
        # Unblock signals
        self.horizontalSlider_brightness.blockSignals(False)
        self.horizontalSlider_contrast.blockSignals(False)
        self.spinBox_slice.blockSignals(False)
        self.checkBox_MIP.blockSignals(False)
        self.checkBox_layer1.blockSignals(False)
        self.checkBox_layer2.blockSignals(False)
        self.checkBox_layer3.blockSignals(False)
        self.comboBox_channelColorLayer1.blockSignals(False)
        self.comboBox_channelColorLayer2.blockSignals(False)
        self.comboBox_channelColorLayer3.blockSignals(False)

        self.changeMarkerSize()

    def borderControl(self, widget, idx=0):
        base_style = "border: 1px solid rgb(223, 223, 223)"
        if idx == 0:
            self.graphicsView_left.setStyleSheet(base_style)
            self.graphicsView_right.setStyleSheet(base_style)
            self.tableView_left.setStyleSheet(base_style)
            self.tableView_right.setStyleSheet(base_style)
        elif idx == 1:
            self.graphicsView_left.setStyleSheet(base_style)
            self.graphicsView_right.setStyleSheet(base_style)
        elif idx == 2:
            self.tableView_left.setStyleSheet(base_style)
            self.tableView_right.setStyleSheet(base_style)
        if widget is not None:
            widget.setStyleSheet("border: 1px solid rgb(100, 140, 220)")

    def ctrlEnDisAble(self, status):
        self.spinBox_rot.setEnabled(status)
        self.spinBox_markerSize.setEnabled(status)
        self.spinBox_slice.setEnabled(False)
        self.checkBox_MIP.setEnabled(status)
        self.checkBox_layer1.setEnabled(status)
        self.checkBox_layer2.setEnabled(status)
        self.checkBox_layer3.setEnabled(status)
        self.comboBox_channelColorLayer1.setEnabled(False)
        self.comboBox_channelColorLayer2.setEnabled(False)
        self.comboBox_channelColorLayer3.setEnabled(False)
        self.radioButton_layer1.setEnabled(False)
        self.radioButton_layer2.setEnabled(False)
        self.radioButton_layer3.setEnabled(False)
        self.horizontalSlider_brightness.setEnabled(status)
        self.horizontalSlider_contrast.setEnabled(status)
        self.toolButton_brightness_reset.setEnabled(status)
        self.toolButton_contrast_reset.setEnabled(status)
        self.toolButton_rotcw.setEnabled(status)
        self.toolButton_rotccw.setEnabled(status)
        self.toolButton_importPoints.setEnabled(not status)
        self.toolButton_exportPoints.setEnabled(not status)
        self.toolButton_loadLayer2.setEnabled(status)
        self.toolButton_loadLayer3.setEnabled(status)

    def setSliders(self):
        self.horizontalSlider_brightness.blockSignals(True)
        self.horizontalSlider_contrast.blockSignals(True)
        if self.label_selimg.text() == 'left':
            if self.radioButton_layer1.isChecked():
                self.horizontalSlider_brightness.setValue(self.brightness_left_layer1)
                self.horizontalSlider_contrast.setValue(self.contrast_left_layer1)
                self.selectedLayer_left = 1
            elif self.radioButton_layer2.isChecked():
                self.horizontalSlider_brightness.setValue(self.brightness_left_layer2)
                self.horizontalSlider_contrast.setValue(self.contrast_left_layer2)
                self.selectedLayer_left = 2
            elif self.radioButton_layer3.isChecked():
                self.horizontalSlider_brightness.setValue(self.brightness_left_layer3)
                self.horizontalSlider_contrast.setValue(self.contrast_left_layer3)
                self.selectedLayer_left = 3
        elif self.label_selimg.text() == 'right':
            if self.radioButton_layer1.isChecked():
                self.horizontalSlider_brightness.setValue(self.brightness_right_layer1)
                self.horizontalSlider_contrast.setValue(self.contrast_right_layer1)
                self.selectedLayer_right = 1
            if self.radioButton_layer2.isChecked():
                self.horizontalSlider_brightness.setValue(self.brightness_right_layer2)
                self.horizontalSlider_contrast.setValue(self.contrast_right_layer2)
                self.selectedLayer_right = 2
            if self.radioButton_layer3.isChecked():
                self.horizontalSlider_brightness.setValue(self.brightness_right_layer3)
                self.horizontalSlider_contrast.setValue(self.contrast_right_layer3)
                self.selectedLayer_right = 3
        self.horizontalSlider_brightness.blockSignals(False)
        self.horizontalSlider_contrast.blockSignals(False)

    def colorModels(self):
        rowsLeft = self.modelLleft.rowCount()
        rowsRight = self.modelRight.rowCount()
        alpha = 100
        for row in range(min(rowsLeft, rowsRight)):
            color_correlate = (50, 220, 175, alpha)
            try:
                self.modelLleft.item(row, 0).setBackground(QtGui.QColor(*color_correlate))
                self.modelLleft.item(row, 1).setBackground(QtGui.QColor(*color_correlate))
                self.modelLleft.item(row, 2).setBackground(QtGui.QColor(*color_correlate))
            except Exception:
                if debug:
                    print(clrmsg.DEBUG + "Model item is None")
            try:
                self.modelRight.item(row, 0).setBackground(QtGui.QColor(*color_correlate))
                self.modelRight.item(row, 1).setBackground(QtGui.QColor(*color_correlate))
                self.modelRight.item(row, 2).setBackground(QtGui.QColor(*color_correlate))
            except Exception:
                if debug:
                    print(clrmsg.DEBUG + "Model item is None")
        # Handle extra rows
        if rowsLeft > rowsRight:
            if '{0:b}'.format(self.sceneLeft.imagetype)[-1] == '0' or \
               '{0:b}'.format(self.sceneLeft.imagetype)[-1] == '{0:b}'.format(self.sceneRight.imagetype)[-1]:
                color_overflow = (105, 220, 0, alpha)
            else:
                color_overflow = (220, 25, 105, alpha)
            for row in range(rowsRight, rowsLeft):
                try:
                    self.modelLleft.item(row, 0).setBackground(QtGui.QColor(*color_overflow))
                    self.modelLleft.item(row, 1).setBackground(QtGui.QColor(*color_overflow))
                    self.modelLleft.item(row, 2).setBackground(QtGui.QColor(*color_overflow))
                except Exception:
                    if debug:
                        print(clrmsg.DEBUG + "Model item is None")
        elif rowsLeft < rowsRight:
            if '{0:b}'.format(self.sceneRight.imagetype)[-1] == '0' or \
               '{0:b}'.format(self.sceneLeft.imagetype)[-1] == '{0:b}'.format(self.sceneRight.imagetype)[-1]:
                color_overflow = (105, 220, 0, alpha)
            else:
                color_overflow = (220, 25, 105, alpha)
            for row in range(rowsLeft, rowsRight):
                try:
                    self.modelRight.item(row, 0).setBackground(QtGui.QColor(*color_overflow))
                    self.modelRight.item(row, 1).setBackground(QtGui.QColor(*color_overflow))
                    self.modelRight.item(row, 2).setBackground(QtGui.QColor(*color_overflow))
                except Exception:
                    if debug:
                        print(clrmsg.DEBUG + "Model item is None")

    def getMarkerColor(self):
        color = QtGui.QColorDialog.getColor()
        self.activateWindow()
        if color.isValid():
            self.markerColor = (color.blue(), color.green(), color.red())
            self.label_markerColor.setStyleSheet("background-color: rgb{0};".format(
                (color.red(), color.green(), color.blue())
            ))

    def getPoiColor(self):
        color = QtGui.QColorDialog.getColor()
        self.activateWindow()
        if color.isValid():
            self.poiColor = (color.blue(), color.green(), color.red())
            self.label_poiColor.setStyleSheet("background-color: rgb{0};".format(
                (color.red(), color.green(), color.blue())
            ))

    def getCustomChannelColor(self):
        color = QtGui.QColorDialog.getColor()
        self.activateWindow()
        if color.isValid():
            return [color.red(), color.green(), color.blue()]

    def initImageLeft(self):
        if self.leftImage is not None:
            self.sceneLeft = QtCustom.QGraphicsSceneCustom(
                self.graphicsView_left, mainWidget=self, side='left', model=self.modelLleft
            )
            self.sceneLeft.pen = QtGui.QPen(QtCore.Qt.red)
            try:
                splashscreen.splash.showMessage("Loading images... " + self.leftImage, color=QtCore.Qt.white)
            except Exception as e:
                print(clrmsg.WARNING, e)
            QtGui.QGuiApplication.processEvents()
            self.sceneLeft.pixelSize = self.pxSize(self.leftImage)
            self.sceneLeft.pixelSizeUnit = 'um'
            self.img_left_layer1, self.sceneLeft.imagetype, self.imgstack_left_layer1 = self.imread(self.leftImage)
            self.img_left_displayed_layer1 = np.copy(self.img_left_layer1)
            self.img_adj_left_layer1 = np.copy(self.img_left_layer1)
            if self.imgstack_left_layer1 is not None:
                self.spinBox_slice.setValue(0)
                self.slice_left = 0
                self.spinBox_slice.setMaximum(self.imgstack_left_layer1.shape[0] - 1)
            self.tableView_left.img1 = self.imgstack_left_layer1
            self.tableView_left.img2 = self.imgstack_left_layer2
            self.tableView_left.img3 = self.imgstack_left_layer3
            if self.imgstack_left_layer1 is None:
                self.sceneLeft._z = False
            else:
                self.sceneLeft._z = True
                self.setCustomRotCenter(max(self.imgstack_left_layer1.shape))
            self.pixmap_left = self.cv2Qimage(self.img_left_displayed_layer1)
            self.pixmap_item_left = QGraphicsPixmapItem(self.pixmap_left)
            self.sceneLeft.addItem(self.pixmap_item_left)
            self.graphicsView_left.setScene(self.sceneLeft)
            self.graphicsView_left.resetTransform()
            scaling_factor = float(self.size) / max(self.pixmap_left.width(), self.pixmap_left.height())
            self.graphicsView_left.scale(scaling_factor, scaling_factor)

    def initImageRight(self):
        if self.rightImage is not None:
            self.sceneRight = QtCustom.QGraphicsSceneCustom(
                self.graphicsView_right, mainWidget=self, side='right', model=self.modelRight
            )
            self.sceneRight.pen = QtGui.QPen(QtCore.Qt.yellow)
            try:
                splashscreen.splash.showMessage("Loading images... " + self.rightImage, color=QtCore.Qt.white)
            except Exception as e:
                print(clrmsg.WARNING, e)
            QtGui.QGuiApplication.processEvents()
            self.sceneRight.pixelSize = self.pxSize(self.rightImage)
            self.sceneRight.pixelSizeUnit = 'um'
            self.img_right_layer1, self.sceneRight.imagetype, self.imgstack_right_layer1 = self.imread(self.rightImage)
            self.img_right_displayed_layer1 = np.copy(self.img_right_layer1)
            self.img_adj_right_layer1 = np.copy(self.img_right_layer1)
            if self.imgstack_right_layer1 is not None:
                self.spinBox_slice.setValue(0)
                self.slice_left = 0
                self.spinBox_slice.setMaximum(self.imgstack_right_layer1.shape[0] - 1)
            self.tableView_right.img1 = self.imgstack_right_layer1
            self.tableView_right.img2 = self.imgstack_right_layer2
            self.tableView_right.img3 = self.imgstack_right_layer3
            if self.imgstack_right_layer1 is None:
                self.sceneRight._z = False
            else:
                self.sceneRight._z = True
                self.setCustomRotCenter(max(self.imgstack_right_layer1.shape))
            self.pixmap_right = self.cv2Qimage(self.img_right_displayed_layer1)
            self.pixmap_item_right = QGraphicsPixmapItem(self.pixmap_right)
            self.sceneRight.addItem(self.pixmap_item_right)
            self.graphicsView_right.setScene(self.sceneRight)
            self.graphicsView_right.resetTransform()
            scaling_factor = float(self.size) / max(self.pixmap_right.width(), self.pixmap_right.height())
            self.graphicsView_right.scale(scaling_factor, scaling_factor)

    def openImageLeft(self):
        path = str(QFileDialog.getOpenFileName(
            None, "Select image file for correlation", self.workingdir, "Image Files (*.tif *.tiff);; All (*.*)"
        ))
        self.activateWindow()
        if path:
            self.graphicsView_left.setFocus()
            self.brightness_left_layer1 = 0
            self.brightness_left_layer2 = 0
            self.brightness_left_layer3 = 0
            self.contrast_left_layer1 = 10
            self.contrast_left_layer2 = 10
            self.contrast_left_layer3 = 10
            self.radioButton_layer1.setChecked(True)
            self.horizontalSlider_brightness.setValue(0)
            self.horizontalSlider_contrast.setValue(10)
            self.spinBox_slice.setValue(0)
            self.checkBox_MIP.setChecked(True)
            self.img_left_layer2 = None
            self.img_adj_left_layer2 = None
            self.sceneLeft.imagetype_layer2 = None
            self.imgstack_left_layer2 = None
            self.img_left_layer3 = None
            self.img_adj_left_layer3 = None
            self.sceneLeft.imagetype_layer3 = None
            self.imgstack_left_layer3 = None
            self.tableView_left.img2 = self.imgstack_left_layer2
            self.tableView_left.img3 = self.imgstack_left_layer3
            self.layer2CHKbox_left = False
            self.layer3CHKbox_left = False
            self.checkBox_layer2.setChecked(False)
            self.checkBox_layer3.setChecked(False)
            self.comboBox_channelColorLayer1.setCurrentIndex(0)
            self.comboBox_channelColorLayer2.setCurrentIndex(0)
            self.comboBox_channelColorLayer3.setCurrentIndex(0)
            self.leftImage = path
            self.sceneLeft.clear()
            self.initImageLeft()
            self.tableView_left._scene = self.sceneLeft
            for i in range(self.tableView_left._model.rowCount()):
                self.sceneLeft.addCircle(0.0, 0.0, 0.0)
            self.tableView_left.updateItems()
            self.tableView_left.setFocus()
            self.graphicsView_left.setFocus()

    def openImageRight(self):
        path = str(QFileDialog.getOpenFileName(
            None, "Select image file for correlation", self.workingdir, "Image Files (*.tif *.tiff);; All (*.*)"
        ))
        self.activateWindow()
        if path:
            self.graphicsView_right.setFocus()
            self.brightness_right_layer1 = 0
            self.brightness_right_layer2 = 0
            self.brightness_right_layer3 = 0
            self.contrast_right_layer1 = 10
            self.contrast_right_layer2 = 10
            self.contrast_right_layer3 = 10
            self.radioButton_layer1.setChecked(True)
            self.horizontalSlider_brightness.setValue(0)
            self.horizontalSlider_contrast.setValue(0)
            self.spinBox_slice.setValue(0)
            self.checkBox_MIP.setChecked(True)
            self.img_right_layer2 = None
            self.img_adj_right_layer2 = None
            self.sceneRight.imagetype_layer2 = None
            self.imgstack_right_layer2 = None
            self.img_right_layer3 = None
            self.img_adj_right_layer3 = None
            self.sceneRight.imagetype_layer3 = None
            self.imgstack_right_layer3 = None
            self.tableView_right.img2 = self.imgstack_right_layer2
            self.tableView_right.img3 = self.imgstack_right_layer3
            self.layer2CHKbox_right = False
            self.layer3CHKbox_right = False
            self.checkBox_layer2.setChecked(False)
            self.checkBox_layer3.setChecked(False)
            self.comboBox_channelColorLayer1.setCurrentIndex(0)
            self.comboBox_channelColorLayer2.setCurrentIndex(0)
            self.comboBox_channelColorLayer3.setCurrentIndex(0)
            self.rightImage = path
            self.sceneRight.clear()
            self.initImageRight()
            self.tableView_right._scene = self.sceneRight
            for i in range(self.tableView_right._model.rowCount()):
                self.sceneRight.addCircle(0.0, 0.0, 0.0)
            self.tableView_right.updateItems()
            self.tableView_right.setFocus()
            self.graphicsView_right.setFocus()

    def resetImageLeft(self, img=None):
        if img is None:
            img = self.imgstack_left_layer1[self.slice_left, :] if not self.mipCHKbox_left else self.img_left_layer1
            self.brightness_left_layer1 = 0
            self.contrast_left_layer1 = 10
            self.horizontalSlider_brightness.setValue(0)
            self.horizontalSlider_contrast.setValue(10)
        self.img_left_overlay = None
        self.img_left_displayed_layer1 = np.copy(img)
        self.img_adj_left_layer1 = np.copy(img)
        self.displayImage(side='left')
        self.sceneLeft.deleteArrows()

    def resetImageRight(self, img=None):
        if img is None:
            img = self.imgstack_right_layer1[self.slice_right, :] if not self.mipCHKbox_right else self.img_right_layer1
            self.brightness_right_layer1 = 0
            self.contrast_right_layer1 = 10
            self.horizontalSlider_brightness.setValue(0)
            self.horizontalSlider_contrast.setValue(10)
        self.img_right_overlay = None
        self.img_right_displayed_layer1 = np.copy(img)
        self.img_adj_right_layer1 = np.copy(img)
        self.displayImage(side='right')
        self.sceneRight.deleteArrows()

    def rotateImage(self):
        current_value = int(self.spinBox_rot.value())
        if self.label_selimg.text() == 'left':
            if current_value == 360:
                self.spinBox_rot.setValue(0)
            elif current_value == -1:
                self.spinBox_rot.setValue(359)
            self.graphicsView_left.rotate(current_value - self.sceneLeft.rotangle)
            self.sceneLeft.rotangle = current_value
            self.sceneLeft.enumeratePoints()
        elif self.label_selimg.text() == 'right':
            if current_value == 360:
                self.spinBox_rot.setValue(0)
            elif current_value == -1:
                self.spinBox_rot.setValue(359)
            self.graphicsView_right.rotate(current_value - self.sceneRight.rotangle)
            self.sceneRight.rotangle = current_value
            self.sceneRight.enumeratePoints()

    def rotateImage45(self, direction=None):
        if direction is None:
            print(clrmsg.ERROR + "Please specify direction ('cw' or 'ccw').")
        elif direction == 'cw':
            if self.label_selimg.text() == 'left':
                self.sceneLeft.rotangle += 45
                self.graphicsView_left.rotate(45)
                self.sceneLeft.rotangle = self.anglectrl(self.sceneLeft.rotangle)
                self.spinBox_rot.setValue(self.sceneLeft.rotangle)
                self.sceneLeft.enumeratePoints()
            elif self.label_selimg.text() == 'right':
                self.sceneRight.rotangle += 45
                self.graphicsView_right.rotate(45)
                self.sceneRight.rotangle = self.anglectrl(self.sceneRight.rotangle)
                self.spinBox_rot.setValue(self.sceneRight.rotangle)
                self.sceneRight.enumeratePoints()
        elif direction == 'ccw':
            if self.label_selimg.text() == 'left':
                self.sceneLeft.rotangle -= 45
                self.graphicsView_left.rotate(-45)
                self.sceneLeft.rotangle = self.anglectrl(self.sceneLeft.rotangle)
                self.spinBox_rot.setValue(self.sceneLeft.rotangle)
            elif self.label_selimg.text() == 'right':
                self.sceneRight.rotangle -= 45
                self.graphicsView_right.rotate(-45)
                self.sceneRight.rotangle = self.anglectrl(self.sceneRight.rotangle)
                self.spinBox_rot.setValue(self.sceneRight.rotangle)

    def anglectrl(self, angle=None):
        if angle is None:
            print(clrmsg.ERROR + "Please specify an angle (e.g. self.sceneLeft.rotangle)")
        if angle >= 360:
            angle -= 360
        elif angle < 0:
            angle += 360
        return angle

    def changeMarkerSize(self):
        if self.label_selimg.text() == 'left':
            self.sceneLeft.markerSize = int(self.spinBox_markerSize.value())
            self.sceneLeft.enumeratePoints()
            if self.sceneLeft.pixelSize:
                try:
                    self.label_markerSizeNano.setText(str(self.sceneLeft.markerSize * 2 * self.sceneLeft.pixelSize))
                    self.label_markerSizeNanoUnit.setText(self.sceneLeft.pixelSizeUnit)
                except Exception as e:
                    if debug:
                        print(clrmsg.DEBUG + "Image pixel size is not a number:", self.label_imgpxsize.text())
                    self.label_markerSizeNano.setText("NaN")
                    self.label_markerSizeNanoUnit.setText('')
            else:
                self.label_markerSizeNano.setText('')
                self.label_markerSizeNanoUnit.setText('')
        elif self.label_selimg.text() == 'right':
            self.sceneRight.markerSize = int(self.spinBox_markerSize.value())
            self.sceneRight.enumeratePoints()
            if self.sceneRight.pixelSize:
                try:
                    self.label_markerSizeNano.setText(str(self.sceneRight.markerSize * 2 * self.sceneRight.pixelSize))
                    self.label_markerSizeNanoUnit.setText(self.sceneRight.pixelSizeUnit)
                except Exception as e:
                    if debug:
                        print(clrmsg.DEBUG + "Image pixel size is not a number:", self.label_imgpxsize.text())
                    self.label_markerSizeNano.setText("NaN")
                    self.label_markerSizeNanoUnit.setText('')
            else:
                self.label_markerSizeNano.setText('')
                self.label_markerSizeNanoUnit.setText('')

    def setCustomRotCenter(self, maxdim):
        halfmaxdim = 0.5 * maxdim
        self.doubleSpinBox_custom_rot_center_x.setValue(halfmaxdim)
        self.doubleSpinBox_custom_rot_center_y.setValue(halfmaxdim)
        self.doubleSpinBox_custom_rot_center_z.setValue(halfmaxdim)

    def imread(self, path, normalize=True):
        if debug:
            print(clrmsg.DEBUG + "===== imread")
        img = tf.imread(path)
        if debug:
            print(clrmsg.DEBUG + "Image shape/dtype:", img.shape, img.dtype)
        if img.dtype == 'uint16':
            img = img * (255.0 / img.max())
            img = img.astype(np.uint8)
            if debug:
                print(clrmsg.DEBUG + "Image dtype converted to:", img.shape, img.dtype)
        if img.ndim == 4:
            if debug:
                print(clrmsg.DEBUG + "Calculating multichannel MIP")
            return np.amax(img, axis=1), 26, img
        elif (img.ndim == 3 and any(dim <= 4 for dim in img.shape)) or img.ndim == 2:
            if debug:
                print(clrmsg.DEBUG + "Loading regular 2D image... multicolor/normalize:", [img.ndim == 3], "/", [normalize])
            if normalize:
                return self.norm_img(img), (25 if img.ndim == 3 else 21), None
            else:
                return img, (9 if img.ndim == 3 else 5), None
        elif img.ndim == 3:
            if debug:
                print(clrmsg.DEBUG + "Calculating MIP")
            return np.amax(img, axis=0), 22, img

    def pxSize(self, img_path, z=False):
        with tf.TiffFile(img_path) as tif:
            for page in tif.pages:
                for tag in page.tags.values():
                    if isinstance(tag.value, str):
                        keywords = ['PhysicalSizeX', 'PixelWidth', 'PixelSize'] if not z else ['PhysicalSizeZ', 'FocusStepSize']
                        for keyword in keywords:
                            tagposs = [m.start() for m in re.finditer(keyword, tag.value)]
                            for tagpos in tagposs:
                                if keyword in ['PhysicalSizeX', 'PhysicalSizeZ']:
                                    for piece in tag.value[tagpos:tagpos+30].split('"'):
                                        try:
                                            pixelSize = float(piece)
                                            if debug:
                                                print(clrmsg.DEBUG + "Pixel size from exif metakey:", keyword)
                                            if z:
                                                pixelSize *= 1000
                                            return pixelSize
                                        except Exception as e:
                                            if debug:
                                                print(clrmsg.DEBUG + "Pixel size parser:", e)
                                elif keyword == 'PixelWidth':
                                    for piece in tag.value[tagpos:tagpos+30].split('='):
                                        try:
                                            pixelSize = float(piece.strip().split('\r\n')[0])
                                        except Exception:
                                            pixelSize = float(piece.strip().split(r'\r\n')[0])
                                        if debug:
                                            print(clrmsg.DEBUG + "Pixel size from exif metakey:", keyword)
                                        return pixelSize * 1E6
                                elif keyword in ['PixelSize', 'FocusStepSize']:
                                    for piece in tag.value[tagpos:tagpos+30].split('"'):
                                        try:
                                            pixelSize = float(piece)
                                            if debug:
                                                print(clrmsg.DEBUG + "Pixel size from exif metakey:", keyword)
                                            return pixelSize
                                        except Exception as e:
                                            if debug:
                                                print(clrmsg.DEBUG + "Pixel size parser:", e)
        return None

    def cv2Qimage(self, img, combobox=None):
        if debug:
            print(clrmsg.DEBUG + "===== cv2Qimage")
        if img.shape[0] <= 4:
            if debug:
                print(clrmsg.DEBUG + "Swapping image axes from c,y,x to y,x,c.")
            img = img.swapaxes(0, 2).swapaxes(0, 1)
        if debug:
            print(clrmsg.DEBUG + "Image shape:", img.shape)
        return QtGui.QPixmap.fromImage(qimage2ndarray.array2qimage(img))

    def setBrightCont(self):
        if self.label_selimg.text() == 'left':
            if self.radioButton_layer1.isChecked():
                self.brightness_left_layer1 = self.horizontalSlider_brightness.value()
                self.contrast_left_layer1 = self.horizontalSlider_contrast.value()
                self.img_adj_left_layer1 = self.adjustBrightCont(
                    self.img_left_displayed_layer1, self.img_adj_left_layer1,
                    self.brightness_left_layer1, self.contrast_left_layer1
                )
            elif self.radioButton_layer2.isChecked():
                self.brightness_left_layer2 = self.horizontalSlider_brightness.value()
                self.contrast_left_layer2 = self.horizontalSlider_contrast.value()
                self.img_adj_left_layer2 = self.adjustBrightCont(
                    self.img_left_displayed_layer2, self.img_adj_left_layer2,
                    self.brightness_left_layer2, self.contrast_left_layer2
                )
            elif self.radioButton_layer3.isChecked():
                self.brightness_left_layer3 = self.horizontalSlider_brightness.value()
                self.contrast_left_layer3 = self.horizontalSlider_contrast.value()
                self.img_adj_left_layer3 = self.adjustBrightCont(
                    self.img_left_displayed_layer3, self.img_adj_left_layer3,
                    self.brightness_left_layer3, self.contrast_left_layer3
                )
        elif self.label_selimg.text() == 'right':
            if self.radioButton_layer1.isChecked():
                self.brightness_right_layer1 = self.horizontalSlider_brightness.value()
                self.contrast_right_layer1 = self.horizontalSlider_contrast.value()
                self.img_adj_right_layer1 = self.adjustBrightCont(
                    self.img_right_displayed_layer1, self.img_adj_right_layer1,
                    self.brightness_right_layer1, self.contrast_right_layer1
                )
            elif self.radioButton_layer2.isChecked():
                self.brightness_right_layer2 = self.horizontalSlider_brightness.value()
                self.contrast_right_layer2 = self.horizontalSlider_contrast.value()
                self.img_adj_right_layer2 = self.adjustBrightCont(
                    self.img_right_displayed_layer2, self.img_adj_right_layer2,
                    self.brightness_right_layer2, self.contrast_right_layer2
                )
            elif self.radioButton_layer3.isChecked():
                self.brightness_right_layer3 = self.horizontalSlider_brightness.value()
                self.contrast_right_layer3 = self.horizontalSlider_contrast.value()
                self.img_adj_right_layer3 = self.adjustBrightCont(
                    self.img_right_displayed_layer3, self.img_adj_right_layer3,
                    self.brightness_right_layer3, self.contrast_right_layer3
                )
        self.displayImage()

    def adjustBrightCont(self, img_displayed, img_adjusted, brightness, contrast):
        if debug:
            ping = time.time()
            print(clrmsg.DEBUG + "===== adjustBrightCont")

        # Copy image and convert to int16 for safe arithmetic
        img_adjusted = np.copy(img_displayed).astype(np.int16)

        # Apply contrast adjustment
        contr = contrast * 0.1
        # Scale and cap at 255 using int16 arithmetic
        img_adjusted = np.where(img_adjusted * contr >= 255, 255, img_adjusted * contr)

        # Apply brightness adjustment on the int16 image
        if brightness > 0:
            # For positive brightness, prevent overflow
            img_adjusted = np.where(255 - img_adjusted <= brightness, 255, img_adjusted + brightness)
        else:
            # For negative brightness, prevent underflow
            img_adjusted = np.where(img_adjusted + brightness < 0, 0, img_adjusted + brightness)

        # Clip the values and convert back to uint8
        img_adjusted = np.clip(img_adjusted, 0, 255).astype(np.uint8)

        if debug:
            pong = time.time()
            print(clrmsg.DEBUG + "adjusting brightness/contrast in s:", pong - ping)

        return img_adjusted

    def norm_img(self, img, copy=False):
        if debug:
            print(clrmsg.DEBUG + "===== norm_img")
        if copy:
            img = np.copy(img)
        dtype = str(img.dtype)
        if dtype in ["uint16", "int16"]:
            typesize = 65535
            out_dtype = np.uint16
        elif dtype in ["uint8", "int8"]:
            typesize = 255
            out_dtype = np.uint8
        elif dtype in ["float32", "float64"]:
            typesize = 1
            out_dtype = img.dtype
        else:
            print(clrmsg.ERROR + "Sorry, I don't know this file type yet:", dtype)
            return img

        if img.ndim == 2:
            # Convert to float, scale, then cast back.
            img = img.astype(np.float64)
            img = (img * (typesize / img.max())).astype(out_dtype)
        elif img.ndim == 3:
            if img.shape[-1] > 4:
                # Assume a stack with shape (z, y, x)
                for i in range(img.shape[0]):
                    slice_img = img[i, :, :].astype(np.float64)
                    img[i, :, :] = (slice_img * (typesize / slice_img.max())).astype(out_dtype)
            else:
                # Assume a multichannel image with shape (y, x, c)
                for i in range(img.shape[2]):
                    channel = img[:, :, i].astype(np.float64)
                    img[:, :, i] = (channel * (typesize / channel.max())).astype(out_dtype)
        return img

    def selectSlice(self):
        if self.label_selimg.text() == 'left':
            self.mipCHKbox_left = self.checkBox_MIP.isChecked()
        elif self.label_selimg.text() == 'right':
            self.mipCHKbox_right = self.checkBox_MIP.isChecked()

        if self.checkBox_MIP.isChecked():
            if self.label_selimg.text() == 'left' and '{0:b}'.format(self.sceneLeft.imagetype)[-1] == '0':
                self.img_left_displayed_layer1 = self.img_left_layer1
                self.img_adj_left_layer1 = self.img_left_layer1
                self.img_left_displayed_layer2 = self.img_left_layer2
                self.img_adj_left_layer2 = self.img_left_layer2
                self.img_left_displayed_layer3 = self.img_left_layer3
                self.displayImage('left')
                if self.brightness_left_layer1 != 0 and self.contrast_left_layer1 != 10:
                    self.setBrightCont()
                else:
                    self.img_adj_left_layer3 = self.img_left_layer3
            elif self.label_selimg.text() == 'right' and '{0:b}'.format(self.sceneRight.imagetype)[-1] == '0':
                self.img_right_displayed_layer1 = self.img_right_layer1
                self.img_adj_right_layer1 = self.img_right_layer1
                self.img_right_displayed_layer2 = self.img_right_layer2
                self.img_adj_right_layer2 = self.img_right_layer2
                self.img_right_displayed_layer3 = self.img_right_layer3
                self.img_adj_right_layer3 = self.img_right_layer3
                if self.brightness_right_layer1 != 0 or self.contrast_right_layer1 != 10:
                    self.setBrightCont()
                else:
                    self.displayImage('right')
        else:
            if self.label_selimg.text() == 'left' and '{0:b}'.format(self.sceneLeft.imagetype)[-1] == '0':
                self.slice_left = int(self.spinBox_slice.value())
                self.img_left_displayed_layer1 = self.imgstack_left_layer1[self.slice_left, :]
                self.img_adj_left_layer1 = self.imgstack_left_layer1[self.slice_left, :]
                if self.img_left_layer2 is not None:
                    self.img_left_displayed_layer2 = self.imgstack_left_layer2[self.slice_left, :]
                    self.img_adj_left_layer2 = self.imgstack_left_layer2[self.slice_left, :]
                if self.img_left_layer3 is not None:
                    self.img_left_displayed_layer3 = self.imgstack_left_layer3[self.slice_left, :]
                    self.img_adj_left_layer3 = self.imgstack_left_layer3[self.slice_left, :]
                if self.brightness_left_layer1 != 0 and self.contrast_left_layer1 != 10:
                    self.setBrightCont()
                else:
                    self.displayImage('left')
            elif self.label_selimg.text() == 'right' and '{0:b}'.format(self.sceneRight.imagetype)[-1] == '0':
                self.slice_right = int(self.spinBox_slice.value())
                self.img_right_displayed_layer1 = self.imgstack_right_layer1[self.slice_right, :]
                self.img_adj_right_layer1 = self.imgstack_right_layer1[self.slice_right, :]
                if self.img_right_layer2 is not None:
                    self.img_right_displayed_layer2 = self.imgstack_right_layer2[self.slice_right, :]
                    self.img_adj_right_layer2 = self.imgstack_right_layer2[self.slice_right, :]
                if self.img_right_layer3 is not None:
                    self.img_right_displayed_layer3 = self.imgstack_right_layer3[self.slice_right, :]
                    self.img_adj_right_layer3 = self.imgstack_right_layer3[self.slice_right, :]
                if self.brightness_right_layer1 != 0 or self.contrast_right_layer1 != 10:
                    self.setBrightCont()
                else:
                    self.displayImage('right')

    def changeColorChannel(self):
        if self.label_selimg.text() == 'left':
            if self.layer1Color_left != self.comboBox_channelColorLayer1.currentIndex():
                self.layer1Color_left = self.comboBox_channelColorLayer1.currentIndex()
                if self.layer1Color_left == 4:
                    self.layer1CustomColor_left = self.getCustomChannelColor()
            if self.layer2Color_left != self.comboBox_channelColorLayer2.currentIndex():
                self.layer2Color_left = self.comboBox_channelColorLayer2.currentIndex()
                if self.layer2Color_left == 4:
                    self.layer2CustomColor_left = self.getCustomChannelColor()
            if self.layer3Color_left != self.comboBox_channelColorLayer3.currentIndex():
                self.layer3Color_left = self.comboBox_channelColorLayer3.currentIndex()
                if self.layer3Color_left == 4:
                    self.layer3CustomColor_left = self.getCustomChannelColor()
            self.displayImage(side='left')
        elif self.label_selimg.text() == 'right':
            if self.layer1Color_right != self.comboBox_channelColorLayer1.currentIndex():
                self.layer1Color_right = self.comboBox_channelColorLayer1.currentIndex()
                if self.layer1Color_right == 4:
                    self.layer1CustomColor_right = self.getCustomChannelColor()
            if self.layer2Color_right != self.comboBox_channelColorLayer2.currentIndex():
                self.layer2Color_right = self.comboBox_channelColorLayer2.currentIndex()
                if self.layer2Color_right == 4:
                    self.layer2CustomColor_right = self.getCustomChannelColor()
            if self.layer3Color_right != self.comboBox_channelColorLayer3.currentIndex():
                self.layer3Color_right = self.comboBox_channelColorLayer3.currentIndex()
                if self.layer3Color_right == 4:
                    self.layer3CustomColor_right = self.getCustomChannelColor()
            self.displayImage(side='right')

    def colorizeImage(self, img, color=None):
        if debug:
            ping = time.time()
        if color is None and all(c.lower() == 'none' for c in [
                self.comboBox_channelColorLayer1.currentText(),
                self.comboBox_channelColorLayer2.currentText(),
                self.comboBox_channelColorLayer3.currentText()]):
            if debug:
                pong = time.time()
                print(clrmsg.DEBUG + "colorize image in s:", pong - ping)
            return img
        elif color is None:
            color = [255, 255, 255]
        if img.ndim == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        imgC = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
        imgC[:, :, 0] = img * (color[0] / 255.0)
        imgC[:, :, 1] = img * (color[1] / 255.0)
        imgC[:, :, 2] = img * (color[2] / 255.0)
        if debug:
            pong = time.time()
            print(clrmsg.DEBUG + "colorize image in s:", pong - ping)
        return imgC.astype(np.uint8)

    def colorCoder(self, code, side, layer):
        if code == 0:
            if side == 'left' and self.img_left_overlay is not None:
                return [255, 255, 255]
            elif side == 'right' and self.img_right_overlay is not None:
                return [255, 255, 255]
            else:
                return None
        elif code == 1:
            return [255, 0, 0]
        elif code == 2:
            return [0, 255, 0]
        elif code == 3:
            return [0, 0, 255]
        elif code == 4:
            if side == 'left':
                if layer == 1:
                    return self.layer1CustomColor_left
                elif layer == 2:
                    return self.layer2CustomColor_left
                elif layer == 3:
                    return self.layer3CustomColor_left
            elif side == 'right':
                if layer == 1:
                    return self.layer1CustomColor_right
                elif layer == 2:
                    return self.layer2CustomColor_right
                elif layer == 3:
                    return self.layer3CustomColor_right

    def displayImage(self, side=None, save=False, keepRGB=False):
        if debug:
            ping = time.time()
        if side is None:
            side = self.label_selimg.text()
        if side == 'left':
            if self.layer1CHKbox_left:
                if keepRGB:
                    image_list = [self.img_adj_left_layer1]
                else:
                    image_list = [self.colorizeImage(self.img_adj_left_layer1,
                                                      color=self.colorCoder(self.layer1Color_left, 'left', 1))]
            else:
                image_list = []
            if self.img_left_layer2 is not None and self.layer2CHKbox_left:
                image_list.append(self.colorizeImage(self.img_adj_left_layer2,
                                                     color=self.colorCoder(self.layer2Color_left, 'left', 2)))
            if self.img_left_layer3 is not None and self.layer3CHKbox_left:
                image_list.append(self.colorizeImage(self.img_adj_left_layer3,
                                                     color=self.colorCoder(self.layer3Color_left, 'left', 3)))
            if self.img_left_overlay is not None:
                image_list.append(self.img_left_overlay)
            img_blend = self.blendImages(image_list)
            self.sceneLeft.removeItem(self.pixmap_item_left)
            self.pixmap_left = self.cv2Qimage(img_blend)
            self.pixmap_item_left = QGraphicsPixmapItem(self.pixmap_left, None, self.sceneLeft)
            QGraphicsItem.stackBefore(self.pixmap_item_left, self.sceneLeft.items()[-1])
            self.pixmap_item_left.setZValue(-10)
        elif side == 'right':
            if self.layer1CHKbox_right:
                image_list = [self.colorizeImage(self.img_adj_right_layer1,
                                                  color=self.colorCoder(self.layer1Color_right, 'right', 1))]
            else:
                image_list = []
            if self.img_right_layer2 is not None and self.layer2CHKbox_right:
                image_list.append(self.colorizeImage(self.img_adj_right_layer2,
                                                     color=self.colorCoder(self.layer2Color_right, 'right', 2)))
            if self.img_right_layer3 is not None and self.layer3CHKbox_right:
                image_list.append(self.colorizeImage(self.img_adj_right_layer3,
                                                     color=self.colorCoder(self.layer3Color_right, 'right', 3)))
            if self.img_right_overlay is not None:
                image_list.append(self.img_right_overlay)
            img_blend = self.blendImages(image_list)
            self.sceneRight.removeItem(self.pixmap_item_right)
            self.pixmap_right = self.cv2Qimage(img_blend)
            self.pixmap_item_right = QGraphicsPixmapItem(self.pixmap_right)
            self.sceneRight.addItem(self.pixmap_item_right)
            QGraphicsItem.stackBefore(self.pixmap_item_right, self.sceneRight.items()[-1])
            self.pixmap_item_right.setZValue(-10)
        if save:
            timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
            cv2.imwrite(os.path.join(self.workingdir, timestamp + "_image.tif"),
                        cv2.cvtColor(img_blend, cv2.COLOR_RGB2BGR))
        if debug:
            pong = time.time()
            print(clrmsg.DEBUG + "displaying image in s:", pong - ping)

    def blendImages(self, images, blendmode='screen'):
        if debug:
            ping = time.time()
        if len(images) == 0:
            return np.zeros((10, 10), dtype=np.uint8) - 1
        if len(images) == 1:
            return images[0].astype(np.uint8)
        blend = images[0]
        for i in range(1, len(images)):
            if blendmode == 'screen':
                blend = blend + images[i] - (blend * images[i].astype(np.float32) / 255.0)
            elif blendmode == 'minimum':
                blend = np.minimum(blend, images[i])
        if debug:
            pong = time.time()
            print(clrmsg.DEBUG + "blending images in s:", pong - ping)
        return blend.astype(np.uint8)

    def layerCtrl(self, layer, load=False):
        if layer == 'layer1':
            if self.label_selimg.text() == 'left':
                self.layer1CHKbox_left = self.checkBox_layer1.isChecked()
            else:
                self.layer1CHKbox_right = self.checkBox_layer1.isChecked()
        elif layer == 'layer2':
            if self.label_selimg.text() == 'left':
                self.layer2CHKbox_left = self.checkBox_layer2.isChecked()
                if (self.img_left_layer2 is None and self.checkBox_layer2.isChecked()) or load:
                    path = str(QFileDialog.getOpenFileName(
                        None, "Select image file", self.workingdir, "Image Files (*.tif *.tiff);; All (*.*)"
                    ))
                    self.activateWindow()
                    if not path:
                        self.layer2CHKbox_left = False
                        self.checkBox_layer2.setChecked(False)
                        return
                    self.img_left_layer2, self.sceneLeft.imagetype_layer2, self.imgstack_left_layer2 = self.imread(path)
                    self.img_adj_left_layer2 = np.copy(self.img_left_layer2)
                    if self.sceneLeft.imagetype_layer2 != self.sceneLeft.imagetype or \
                       (self.imgstack_left_layer1 is not None and self.imgstack_left_layer2.shape != self.imgstack_left_layer1.shape) or \
                       (self.imgstack_left_layer1 is None and self.img_left_layer2.shape != self.img_left_layer1.shape):
                        QMessageBox.critical(self, "Warning", "This image file does not match the first image!")
                        self.img_left_layer2 = self.sceneLeft.imagetype_layer2 = self.imgstack_left_layer2 = None
                        self.layer2CHKbox_left = False
                        self.checkBox_layer2.setChecked(False)
                        return
                    else:
                        if load:
                            self.checkBox_layer2.blockSignals(True)
                            self.layer2CHKbox_left = True
                            self.checkBox_layer2.setChecked(True)
                            self.comboBox_channelColorLayer2.setEnabled(True)
                            self.radioButton_layer2.setEnabled(True)
                            self.checkBox_layer2.blockSignals(False)
                        self.img_left_displayed_layer2 = self.img_left_layer2
                        self.selectSlice()
            else:
                self.layer2CHKbox_right = self.checkBox_layer2.isChecked()
                if (self.img_right_layer2 is None and self.checkBox_layer2.isChecked()) or load:
                    path = str(QFileDialog.getOpenFileName(
                        None, "Select image file", self.workingdir, "Image Files (*.tif *.tiff);; All (*.*)"
                    ))
                    self.activateWindow()
                    if not path:
                        self.layer2CHKbox_right = False
                        self.checkBox_layer2.setChecked(False)
                        return
                    self.img_right_layer2, self.sceneRight.imagetype_layer2, self.imgstack_right_layer2 = self.imread(path)
                    self.img_adj_right_layer2 = np.copy(self.img_right_layer2)
                    if self.sceneRight.imagetype_layer2 != self.sceneRight.imagetype or \
                       (self.imgstack_right_layer1 is not None and self.imgstack_right_layer2.shape != self.imgstack_right_layer1.shape) or \
                       (self.imgstack_right_layer1 is None and self.img_right_layer2.shape != self.img_right_layer1.shape):
                        QMessageBox.critical(self, "Warning", "This image file does not match the first image!")
                        self.img_right_layer2 = self.sceneRight.imagetype_layer2 = self.imgstack_right_layer2 = None
                        self.layer2CHKbox_right = False
                        self.checkBox_layer2.setChecked(False)
                        return
                    else:
                        if load:
                            self.checkBox_layer2.blockSignals(True)
                            self.layer2CHKbox_right = True
                            self.checkBox_layer2.setChecked(True)
                            self.comboBox_channelColorLayer2.setEnabled(True)
                            self.radioButton_layer2.setEnabled(True)
                            self.checkBox_layer2.blockSignals(False)
                        self.img_right_displayed_layer2 = self.img_right_layer2
                        self.selectSlice()
        elif layer == 'layer3':
            if self.label_selimg.text() == 'left':
                self.layer3CHKbox_left = self.checkBox_layer3.isChecked()
                if (self.img_left_layer3 is None and self.checkBox_layer3.isChecked()) or load:
                    path = str(QFileDialog.getOpenFileName(
                        None, "Select image file", self.workingdir, "Image Files (*.tif *.tiff);; All (*.*)"
                    ))
                    self.activateWindow()
                    if not path:
                        self.layer3CHKbox_left = False
                        self.checkBox_layer3.setChecked(False)
                        return
                    self.img_left_layer3, self.sceneLeft.imagetype_layer3, self.imgstack_left_layer3 = self.imread(path)
                    self.img_adj_left_layer3 = np.copy(self.img_left_layer3)
                    if self.sceneLeft.imagetype_layer3 != self.sceneLeft.imagetype or \
                       (self.imgstack_left_layer1 is not None and self.imgstack_left_layer3.shape != self.imgstack_left_layer1.shape) or \
                       (self.imgstack_left_layer1 is None and self.img_left_layer3.shape != self.img_left_layer1.shape):
                        QMessageBox.critical(self, "Warning", "This image file does not match the first image!")
                        self.img_left_layer3 = self.sceneLeft.imagetype_layer3 = self.imgstack_left_layer3 = None
                        self.layer3CHKbox_left = False
                        self.checkBox_layer3.setChecked(False)
                        return
                    else:
                        if load:
                            self.checkBox_layer3.blockSignals(True)
                            self.layer3CHKbox_left = True
                            self.checkBox_layer3.setChecked(True)
                            self.comboBox_channelColorLayer3.setEnabled(True)
                            self.radioButton_layer3.setEnabled(True)
                            self.checkBox_layer3.blockSignals(False)
                        self.img_left_displayed_layer3 = self.img_left_layer3
                        self.selectSlice()
            else:
                self.layer3CHKbox_right = self.checkBox_layer3.isChecked()
                if (self.img_right_layer3 is None and self.checkBox_layer3.isChecked()) or load:
                    path = str(QFileDialog.getOpenFileName(
                        None, "Select image file", self.workingdir, "Image Files (*.tif *.tiff);; All (*.*)"
                    ))
                    self.activateWindow()
                    if not path:
                        self.layer3CHKbox_right = False
                        self.checkBox_layer3.setChecked(False)
                        return
                    self.img_right_layer3, self.sceneRight.imagetype_layer3, self.imgstack_right_layer3 = self.imread(path)
                    self.img_adj_right_layer3 = np.copy(self.img_right_layer3)
                    if self.sceneRight.imagetype_layer3 != self.sceneRight.imagetype or \
                       (self.imgstack_right_layer1 is not None and self.imgstack_right_layer3.shape != self.imgstack_right_layer1.shape) or \
                       (self.imgstack_right_layer1 is None and self.img_right_layer3.shape != self.img_right_layer1.shape):
                        QMessageBox.critical(self, "Warning", "This image file does not match the first image!")
                        self.img_right_layer3 = self.sceneRight.imagetype_layer3 = self.imgstack_right_layer3 = None
                        self.layer3CHKbox_right = False
                        self.checkBox_layer3.setChecked(False)
                        return
                    else:
                        if load:
                            self.checkBox_layer3.blockSignals(True)
                            self.layer3CHKbox_right = True
                            self.checkBox_layer3.setChecked(True)
                            self.comboBox_channelColorLayer3.setEnabled(True)
                            self.radioButton_layer3.setEnabled(True)
                            self.checkBox_layer3.blockSignals(False)
                        self.img_right_displayed_layer3 = self.img_right_layer3
                        self.selectSlice()

        self.tableView_left.img1 = self.imgstack_left_layer1
        self.tableView_left.img2 = self.imgstack_left_layer2
        self.tableView_left.img3 = self.imgstack_left_layer3
        self.tableView_right.img1 = self.imgstack_right_layer1
        self.tableView_right.img2 = self.imgstack_right_layer2
        self.tableView_right.img3 = self.imgstack_right_layer3

        self.displayImage()

    def autosave(self):
        csv_file_out = os.path.splitext(self.leftImage)[0] + '_coordinates.txt'
        csvHandler.model2csv(self.modelLleft, csv_file_out, delimiter="\t")
        csv_file_out = os.path.splitext(self.rightImage)[0] + '_coordinates.txt'
        csvHandler.model2csv(self.modelRight, csv_file_out, delimiter="\t")

    def exportPoints(self):
        side = self.label_selectedTable.text()
        model = self.modelLleft if side == 'left' else self.modelRight
        csv_file_out, filterdialog = QFileDialog.getSaveFileNameAndFilter(
            self, 'Export file as',
            os.path.dirname(self.leftImage) if side == 'left' else os.path.dirname(self.rightImage),
            "Tabstop separated (*.csv *.txt);;Comma separated (*.csv *.txt)"
        )
        self.activateWindow()
        if str(filterdialog).startswith('Comma'):
            csvHandler.model2csv(model, csv_file_out, delimiter=",")
        elif str(filterdialog).startswith('Tabstop'):
            csvHandler.model2csv(model, csv_file_out, delimiter="\t")

    def importPoints(self):
        side = self.label_selectedTable.text()
        csv_file_in, filterdialog = QFileDialog.getOpenFileNameAndFilter(
            self, 'Import file as',
            os.path.dirname(self.leftImage) if side == 'left' else os.path.dirname(self.rightImage),
            "Tabstop separated (*.csv *.txt);;Comma separated (*.csv *.txt)"
        )
        self.activateWindow()
        if str(filterdialog).startswith('Comma'):
            itemlist = csvHandler.csv2list(csv_file_in, delimiter=",", parent=self, sniff=True)
        elif str(filterdialog).startswith('Tabstop'):
            itemlist = csvHandler.csv2list(csv_file_in, delimiter="\t", parent=self, sniff=True)
        if side == 'left':
            for item in itemlist:
                self.sceneLeft.addCircle(
                    float(item[0]),
                    float(item[1]),
                    float(item[2]) if len(item) > 2 else 0
                )
            self.sceneLeft.itemsToModel()
        elif side == 'right':
            for item in itemlist:
                self.sceneRight.addCircle(
                    float(item[0]),
                    float(item[1]),
                    float(item[2]) if len(item) > 2 else 0
                )
            self.sceneRight.itemsToModel()

    def model2np(self, model, rows):
        listarray = []
        for rowNumber in range(*rows):
            fields = [
                float(model.data(model.index(rowNumber, columnNumber), QtCore.Qt.DisplayRole))
                for columnNumber in range(model.columnCount())
            ]
            listarray.append(fields)
        return np.array(listarray).astype(np.float64)

    def correlate(self):
        # Correlation logic with 2D/3D decision and result display...
        # (Keep your correlation implementation as-is; only reformat print statements)
        if '{0:b}'.format(self.sceneLeft.imagetype)[-1] == '1' and '{0:b}'.format(self.sceneRight.imagetype)[-1] == '0':
            model2D = self.modelLleft
            model3D = self.modelRight
            img = np.copy(self.colorizeImage(self.img_adj_left_layer1,
                        color=self.colorCoder(self.layer1Color_left, 'left', 1)))
            imgSide = 'left'
            if img.shape[0] == 470:
                imgShape = [442, img.shape[1]]
            elif img.shape[0] == 941:
                imgShape = [884, img.shape[1]]
            elif img.shape[0] == 1883:
                imgShape = [1768, img.shape[1]]
            elif img.shape[0] == 3767:
                imgShape = [3536, img.shape[1]]
            else:
                imgShape = img.shape
            imageProps = [imgShape, self.sceneLeft.pixelSize, self.imgstack_right_layer1.shape]
            if img.ndim == 2:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        elif '{0:b}'.format(self.sceneLeft.imagetype)[-1] == '0' and '{0:b}'.format(self.sceneRight.imagetype)[-1] == '1':
            model2D = self.modelRight
            model3D = self.modelLleft
            img = np.copy(self.colorizeImage(self.img_adj_right_layer1,
                        color=self.colorCoder(self.layer1Color_right, 'right', 1)))
            imgSide = 'right'
            if img.shape[0] == 470:
                imgShape = [442, img.shape[1]]
            elif img.shape[0] == 941:
                imgShape = [884, img.shape[1]]
            elif img.shape[0] == 1883:
                imgShape = [1768, img.shape[1]]
            elif img.shape[0] == 3767:
                imgShape = [3536, img.shape[1]]
            else:
                imgShape = img.shape
            imageProps = [imgShape, self.sceneRight.pixelSize, self.imgstack_left_layer1.shape]
            if img.ndim == 2:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        else:
            def corrMsgBox(self, msg):
                print("message box")
                msgBox = QMessageBox()
                msgBox.setIcon(QMessageBox.Question)
                msgBox.setText(msg)
                l2rButton = msgBox.addButton("Left to Right", QMessageBox.ActionRole)
                r2lButton = msgBox.addButton("Right to Left", QMessageBox.ActionRole)
                abortButton = msgBox.addButton(QMessageBox.Cancel)
                msgBox.exec_()
                if msgBox.clickedButton() == l2rButton:
                    return "l2r"
                elif msgBox.clickedButton() == r2lButton:
                    return "r2l"
                elif msgBox.clickedButton() == abortButton:
                    return None

            if '{0:b}'.format(self.sceneLeft.imagetype)[-1] == '0' and '{0:b}'.format(self.sceneRight.imagetype)[-1] == '0':
                rowsLeft = self.modelLleft.rowCount()
                rowsRight = self.modelRight.rowCount()
                if rowsLeft > rowsRight:
                    corrMsgBoxRetVal = 'l2r'
                elif rowsLeft < rowsRight:
                    corrMsgBoxRetVal = 'r2l'
                else:
                    corrMsgBoxRetVal = corrMsgBox(self,
                        "It seems you want to do a 3D to 3D correlation. Since both datasets contain the same "
                        "amount of markers, please specify which side you want to correlate to:")
                if corrMsgBoxRetVal == 'l2r':
                    model2D = self.modelRight
                    model3D = self.modelLleft
                    img = np.copy(self.colorizeImage(self.img_adj_right_layer1,
                                color=self.colorCoder(self.layer1Color_right, 'right', 1)))
                    imgSide = 'right'
                elif corrMsgBoxRetVal == 'r2l':
                    model2D = self.modelLleft
                    model3D = self.modelRight
                    img = np.copy(self.colorizeImage(self.img_adj_left_layer1,
                                color=self.colorCoder(self.layer1Color_left, 'left', 1)))
                    imgSide = 'left'
                else:
                    return
                if img.ndim == 2:
                    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
                imageProps = None
            elif '{0:b}'.format(self.sceneLeft.imagetype)[-1] == '1' and '{0:b}'.format(self.sceneRight.imagetype)[-1] == '1':
                rowsLeft = self.modelLleft.rowCount()
                rowsRight = self.modelRight.rowCount()
                if rowsLeft > rowsRight:
                    corrMsgBoxRetVal = 'l2r'
                elif rowsLeft < rowsRight:
                    corrMsgBoxRetVal = 'r2l'
                else:
                    corrMsgBoxRetVal = corrMsgBox(self,
                        "It seems you want to do a 2D to 2D correlation. Since both datasets contain the same "
                        "amount of markers, please specify which side you want to correlate to:")
                if corrMsgBoxRetVal == 'l2r':
                    model2D = self.modelRight
                    model3D = self.modelLleft
                    img = np.copy(self.colorizeImage(self.img_adj_right_layer1,
                                color=self.colorCoder(self.layer1Color_right, 'right', 1)))
                    imgSide = 'right'
                elif corrMsgBoxRetVal == 'r2l':
                    model2D = self.modelLleft
                    model3D = self.modelRight
                    img = np.copy(self.colorizeImage(self.img_adj_left_layer1,
                                color=self.colorCoder(self.layer1Color_left, 'left', 1)))
                    imgSide = 'left'
                else:
                    return
                if img.ndim == 2:
                    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
                imageProps = None
            else:
                QMessageBox.critical(self, "Data Structure", 'Cannot determine if datasets are 2D or 3D')
                raise ValueError('Cannot determine if datasets are 2D or 3D')
        nrRowsModel2D = model2D.rowCount()
        nrRowsModel3D = model3D.rowCount()
        self.rotation_center = [
            self.doubleSpinBox_custom_rot_center_x.value(),
            self.doubleSpinBox_custom_rot_center_y.value(),
            self.doubleSpinBox_custom_rot_center_z.value()
        ]
        if imageProps is not None and None in imageProps:
            imageProps = None

        if nrRowsModel2D >= 3:
            if nrRowsModel2D <= nrRowsModel3D:
                timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
                self.correlation_results = correlation.main(
                    markers_3d=self.model2np(model3D, [0, nrRowsModel2D]),
                    markers_2d=self.model2np(model2D, [0, nrRowsModel2D]),
                    spots_3d=self.model2np(model3D, [nrRowsModel2D, nrRowsModel3D]),
                    rotation_center=self.rotation_center,
                    results_file=''.join([self.workingdir, '/', timestamp, '_correlation.txt']
                                          if self.checkBox_writeReport.isChecked() else ''),
                    imageProps=imageProps
                )
            else:
                QMessageBox.critical(self, "Data Structure", "The two datasets do not contain the same amount of markers!")
                return
        else:
            QMessageBox.critical(self, "Data Structure", 'At least THREE markers are needed to do the correlation')
            return

        transf_3d = self.correlation_results[1]
        alpha = self.doubleSpinBox_markerAlpha.value()
        radius = self.spinBox_markerRadius.value()
        poiAlpha = self.doubleSpinBox_poiAlpha.value()
        poiSize = self.spinBox_poiSize.value()
        poiForm = self.comboBox_poiForm.currentIndex()
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_orig = np.copy(img)
        img_overlay = np.zeros(img.shape, dtype=img.dtype)
        for i in range(transf_3d.shape[1]):
            cv2.circle(img, (int(round(transf_3d[0, i])), int(round(transf_3d[1, i]))), radius, self.markerColor, -1)
            cv2.circle(img_overlay, (int(round(transf_3d[0, i])), int(round(transf_3d[1, i]))), radius, self.markerColor, -1)
        img = cv2.addWeighted(img, alpha, img_orig, 1 - alpha, 0.0)
        img_overlay = cv2.addWeighted(img_overlay, alpha, np.zeros(img.shape, dtype=img.dtype), 1 - alpha, 0.0)
        img_orig = np.copy(img)
        img_overlay_orig = np.copy(img_overlay)
        if self.correlation_results[2] is not None:
            calc_spots_2d = self.correlation_results[2]
            for i in range(calc_spots_2d.shape[1]):
                if poiForm == 0:
                    poiCrossLength = poiSize if poiSize in [1, 2, 3] else poiSize
                    poiCrossThick = 1 if poiSize in [1, 2, 3] else int(round(poiSize * 0.33))
                    cv2.line(img,
                             (int(round(calc_spots_2d[0, i] - poiCrossLength)), int(round(calc_spots_2d[1, i]))),
                             (int(round(calc_spots_2d[0, i] + poiCrossLength)), int(round(calc_spots_2d[1, i]))),
                             self.poiColor, poiCrossThick)
                    cv2.line(img_overlay,
                             (int(round(calc_spots_2d[0, i] - poiCrossLength)), int(round(calc_spots_2d[1, i]))),
                             (int(round(calc_spots_2d[0, i] + poiCrossLength)), int(round(calc_spots_2d[1, i]))),
                             self.poiColor, poiCrossThick)
                    cv2.line(img,
                             (int(round(calc_spots_2d[0, i])), int(round(calc_spots_2d[1, i]) - poiCrossLength)),
                             (int(round(calc_spots_2d[0, i])), int(round(calc_spots_2d[1, i]) + poiCrossLength)),
                             self.poiColor, poiCrossThick)
                    cv2.line(img_overlay,
                             (int(round(calc_spots_2d[0, i])), int(round(calc_spots_2d[1, i]) - poiCrossLength)),
                             (int(round(calc_spots_2d[0, i])), int(round(calc_spots_2d[1, i]) + poiCrossLength)),
                             self.poiColor, poiCrossThick)
                elif poiForm == 1:
                    cv2.circle(img, (int(round(calc_spots_2d[0, i])), int(round(calc_spots_2d[1, i]))), poiSize, self.poiColor, -1)
                    cv2.circle(img_overlay, (int(round(calc_spots_2d[0, i])), int(round(calc_spots_2d[1, i]))), poiSize, self.poiColor, -1)
        img = cv2.addWeighted(img, poiAlpha, img_orig, 1 - poiAlpha, 0.0)
        img_overlay = cv2.addWeighted(img_overlay, poiAlpha, img_overlay_orig, 1 - poiAlpha, 0.0)
        if self.checkBox_writeReport.isChecked():
            cv2.imwrite(os.path.join(self.workingdir, timestamp + "_correlated.tif"), img)
        try:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_overlay = cv2.cvtColor(img_overlay, cv2.COLOR_BGR2RGB)
        except Exception:
            pass
        if imgSide == 'left':
            self.img_left_overlay = np.copy(img_overlay)
            self.displayImage(side='left', keepRGB=False)
        else:
            self.img_right_overlay = np.copy(img_overlay)
            self.displayImage(side='right', keepRGB=False)
        self.displayResults(
            frame=self.checkBox_scatterPlotFrame.isChecked(),
            framesize=self.doubleSpinBox_scatterPlotFrameSize.value()
        )
        model2D.tableview._scene.deleteArrows()
        for i in range(nrRowsModel2D):
            model2D.tableview._scene.addArrow(
                self.model2np(model2D, [0, nrRowsModel2D])[i, :2],
                self.correlation_results[1][:2, i],
                arrowangle=45,
                color=QtCore.Qt.red
            )

    def displayResults(self, frame=False, framesize=None):
        if hasattr(self, "correlation_results"):
            transf = self.correlation_results[0]
            delta2D = self.correlation_results[3]
            delta2D_mean = np.absolute(delta2D).mean(axis=1)
            translation = (transf.d[0], transf.d[1])
            translation_customRotation = self.correlation_results[5]
            eulers = transf.extract_euler(r=transf.q, mode='x', ret='one') * 180 / np.pi
            scale = transf.s_scalar

            self.label_phi.setText('{0:.3f}'.format(eulers[0]))
            self.label_phi.setStyleSheet(self.stylesheet_green)
            self.label_psi.setText('{0:.3f}'.format(eulers[2]))
            self.label_psi.setStyleSheet(self.stylesheet_green)
            self.label_theta.setText('{0:.3f}'.format(eulers[1]))
            self.label_theta.setStyleSheet(self.stylesheet_green)
            self.label_scale.setText('{0:.3f}'.format(scale))
            self.label_scale.setStyleSheet(self.stylesheet_green)
            self.label_translation.setText('x = {0:.3f} | y = {1:.3f}'.format(translation[0], translation[1]))
            self.label_translation.setStyleSheet(self.stylesheet_green)
            self.label_custom_rot_center.setText('[{0},{1},{2}]:'.format(
                int(self.doubleSpinBox_custom_rot_center_x.value()),
                int(self.doubleSpinBox_custom_rot_center_y.value()),
                int(self.doubleSpinBox_custom_rot_center_z.value())
            ))
            self.label_translation_custom_rot.setText('x = {0:.3f} | y = {1:.3f}'.format(
                translation_customRotation[0], translation_customRotation[1]
            ))
            self.label_translation_custom_rot.setStyleSheet(self.stylesheet_green)
            self.label_meandxdy.setText('{0:.5f} / {1:.5f}'.format(delta2D_mean[0], delta2D_mean[1]))
            if delta2D_mean[0] <= 1 and delta2D_mean[1] <= 1:
                self.label_meandxdy.setStyleSheet(self.stylesheet_green)
            elif delta2D_mean[0] < 2 or delta2D_mean[1] < 2:
                self.label_meandxdy.setStyleSheet(self.stylesheet_orange)
            else:
                self.label_meandxdy.setStyleSheet(self.stylesheet_red)
            self.label_rms.setText('{0:.5f}'.format(transf.rmsError))
            self.label_rms.setStyleSheet(self.stylesheet_green if transf.rmsError < 1 else self.stylesheet_orange)

            self.widget_matplotlib.setupScatterCanvas(width=4, height=4, dpi=52, toolbar=False)
            self.widget_matplotlib.scatterPlot(
                x=delta2D[0, :], y=delta2D[1, :],
                frame=frame, framesize=framesize,
                xlabel="px", ylabel="px"
            )

            self.modelResults.removeRows(0, self.modelResults.rowCount())
            if self.checkBox_resultsAbsolute.isChecked():
                delta2D = np.absolute(delta2D)
            for i in range(delta2D.shape[1]):
                item = [
                    QtGui.QStandardItem(str(i + 1)),
                    QtGui.QStandardItem('{0:.5f}'.format(delta2D[0, i])),
                    QtGui.QStandardItem('{0:.5f}'.format(delta2D[1, i]))
                ]
                self.modelResults.appendRow(item)
            self.modelResults.setHeaderData(0, QtCore.Qt.Horizontal, 'Nr.')
            self.modelResults.setHeaderData(1, QtCore.Qt.Horizontal, 'dx')
            self.modelResults.setHeaderData(2, QtCore.Qt.Horizontal, 'dy')
            self.tableView_results.setColumnWidth(1, 86)
            self.tableView_results.setColumnWidth(2, 86)
        else:
            pass

    def showSelectedResidual(self, doubleclick=False):
        indices = self.tableView_results.selectedIndexes()
        if '{0:b}'.format(self.sceneLeft.imagetype)[-1] == '1':
            tableView1 = self.tableView_left
            tableView2 = self.tableView_right
            graphicsView = self.graphicsView_left
        else:
            tableView1 = self.tableView_right
            tableView2 = self.tableView_left
            graphicsView = self.graphicsView_right
        if indices:
            rows = set(index.row() for index in indices)
            for row in rows:
                markerNr = int(self.modelResultsProxy.data(self.modelResultsProxy.index(row, 0)).toString()) - 1
                tableView1.selectRow(markerNr)
                tableView2.selectRow(markerNr)
        else:
            tableView1.clearSelection()
            tableView2.clearSelection()
        if doubleclick:
            if debug:
                print(clrmsg.DEBUG + "double click")
                print(clrmsg.DEBUG, graphicsView.transform().m11(), graphicsView.transform().m22())
            graphicsView.setTransform(QtGui.QTransform(
                20,
                graphicsView.transform().m12(),
                graphicsView.transform().m13(),
                graphicsView.transform().m21(),
                20,
                graphicsView.transform().m23(),
                graphicsView.transform().m31(),
                graphicsView.transform().m32(),
                graphicsView.transform().m33()
            ))
            if debug:
                print(clrmsg.DEBUG, graphicsView.transform().m11(), graphicsView.transform().m22())
            graphicsView.centerOn(
                float(tableView1._model.data(tableView1._model.index(markerNr, 0)).toString()),
                float(tableView1._model.data(tableView1._model.index(markerNr, 1)).toString())
            )

    def cmTableViewResults(self, pos):
        indices = self.tableView_results.selectedIndexes()
        if indices:
            cmApplyShift = QtGui.QAction('Apply shift to marker', self)
            cmApplyShift.triggered.connect(self.applyResidualShift)
            self.contextMenu = QtGui.QMenu(self)
            self.contextMenu.addAction(cmApplyShift)
            self.contextMenu.popup(QtGui.QCursor.pos())

    def applyResidualShift(self):
        indices = self.tableView_results.selectedIndexes()
        if '{0:b}'.format(self.sceneLeft.imagetype)[-1] == '1':
            tableView = self.tableView_left
            scene = self.sceneLeft
        else:
            tableView = self.tableView_right
            scene = self.sceneRight
        items = [item for item in scene.items() if isinstance(item, QtGui.QGraphicsEllipseItem)]
        if indices:
            rows = set(index.row() for index in indices)
            for row in rows:
                markerNr = int(self.modelResultsProxy.data(self.modelResultsProxy.index(row, 0)).toString()) - 1
                if debug:
                    print(clrmsg.DEBUG + "Marker number/background color (Qrgba)", markerNr,
                          self.modelResults.itemFromIndex(
                              self.modelResultsProxy.mapToSource(self.modelResultsProxy.index(row, 0))
                          ).background().color().rgba())
                if self.modelResults.itemFromIndex(
                        self.modelResultsProxy.mapToSource(self.modelResultsProxy.index(row, 0))
                ).background().color().rgba() == 4278190080:
                    BackColor = (50, 220, 175, 100)
                    ForeColor = (180, 180, 180, 255)
                    items[markerNr].setPos(
                        float(tableView._model.data(tableView._model.index(markerNr, 0)).toString()) + self.correlation_results[3][0, markerNr],
                        float(tableView._model.data(tableView._model.index(markerNr, 1)).toString()) + self.correlation_results[3][1, markerNr]
                    )
                    for col in range(3):
                        item_obj = self.modelResults.itemFromIndex(
                            self.modelResultsProxy.mapToSource(self.modelResultsProxy.index(row, col))
                        )
                        item_obj.setBackground(QtGui.QColor(*BackColor))
                        item_obj.setForeground(QtGui.QColor(*ForeColor))
        scene.itemsToModel()
        self.tableView_results.clearSelection()


class SplashScreen:
    def __init__(self):
        QtGui.QGuiApplication.processEvents()
        splash_pix = QtGui.QPixmap(os.path.join(execdir, 'icons', 'SplashScreen.png'))
        painter = QtGui.QPainter()
        painter.begin(splash_pix)
        painter.setPen(QtCore.Qt.white)
        painter.drawText(0, 0, splash_pix.size().width() - 3, splash_pix.size().height() - 1,
                         QtCore.Qt.AlignBottom | QtCore.Qt.AlignRight, __version__)
        painter.end()
        self.splash = QSplashScreen(splash_pix, QtCore.Qt.WindowStaysOnTopHint)
        self.splash.setMask(splash_pix.mask())
        self.splash.show()
        self.splash.showMessage("Initializing...", color=QtCore.Qt.white)
        QtGui.QGuiApplication.processEvents()
        time.sleep(1)
        self.splash.showMessage("Loading images...", color=QtCore.Qt.white)


class Main:
    def __init__(self, leftImage=None, rightImage=None, nosplash=False, workingdir=None):
        self.exitstatus = 1
        if leftImage is None or rightImage is None:
            sys.exit("Please pass 'leftImage=PATH' and 'rightImage=PATH' to this function")
        if not nosplash:
            global splashscreen
            splashscreen = SplashScreen()
        if workingdir is None:
            workingdir = execdir
        self.window = MainWidget(parent=self, leftImage=leftImage, rightImage=rightImage, workingdir=workingdir)
        self.window.show()
        self.window.raise_()
        if not nosplash:
            splashscreen.splash.finish(self.window)

    def cleanUp(self):
        try:
            del self.window
        except Exception as e:
            if debug:
                print(clrmsg.DEBUG + str(e))


if __name__ == "__main__":
    if debug:
        print(clrmsg.DEBUG + "Debug Test")
        print(clrmsg.OK + "OK Test")
        print(clrmsg.ERROR + "Error Test")
        print(clrmsg.INFO + "Info Test")
        print(clrmsg.WARNING + "Warning Test")
        print("=" * 20, "Initializing", "=" * 20)

    app = QtGui.QGuiApplication(sys.argv)
    left = str(QFileDialog.getOpenFileName(
        None, "Select first image file for correlation", execdir, "Image Files (*.tif *.tiff);; All (*.*)"
    ))
    if not left:
        sys.exit()
    right = str(QFileDialog.getOpenFileName(
        None, "Select second image file for correlation", execdir, "Image Files (*.tif *.tiff);; All (*.*)"
    ))
    if not right:
        sys.exit()

    main = Main(leftImage=left, rightImage=right)
    sys.exit(app.exec_())
