#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
3D Correlation Toolbox - 3DCT
...
(license and metadata here)
...
"""

import sys
import os
import tempfile
import time
import webbrowser
from subprocess import call

# Import core modules from PyQt5
from PyQt5 import QtCore, uic, QtGui, QtWidgets
from PyQt5.QtGui import QMovie, QPixmap, QPainter
from PyQt5.QtWidgets import (QApplication, QMainWindow, QMessageBox, QFileDialog,
                             QListWidgetItem, QSplashScreen)

# Import your other modules
from tdct import clrmsg, TDCT_debug, helpdoc, stackProcessing
import TDCT_correlation

# Add working directory temporarily to PYTHONPATH
if getattr(sys, 'frozen', False):
    execdir = sys._MEIPASS
else:
    execdir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(execdir)

debug = TDCT_debug.debug

if sys.platform == 'win32':
    if debug:
        print(clrmsg.INFO + 'PATH before:', os.environ.get('PATH', ''))
    os.environ['PATH'] = execdir + '\\;' + os.environ.get('PATH', '')
    if debug:
        print(clrmsg.INFO + 'PATH after: ', os.environ.get('PATH', ''))
__version__ = 'v2.3.0'

if debug:
    print(clrmsg.DEBUG + "Execdir =", execdir)

########## GUI layout file #######################################################
qtCreatorFile_main = os.path.join(execdir, "TDCT_main.ui")
Ui_MainWindow, QtBaseClass = uic.loadUiType(qtCreatorFile_main)


########## Main Application Class ################################################
class APP(QMainWindow, Ui_MainWindow):
    def __init__(self):
        super(APP, self).__init__()
        self.setupUi(self)

        # Hide debug menu if needed
        self.menuDebug.menuAction().setVisible(False)

        # Menu: set shortcuts and actions
        self.actionQuit.triggered.connect(self.close)
        self.actionQuit.setShortcuts(['Ctrl+Q', 'Esc'])
        self.actionQuit.setStatusTip('Exit application')
        self.actionHelp.triggered.connect(self.help)
        self.actionHelp.setStatusTip("Open User's Guide: http://3dct.semper.space/userguide.html")
        self.helpdoc = helpdoc.help(self)
        self.actionAbout.triggered.connect(self.about)

        # Open Buttons (update QFileDialog usage to QtWidgets)
        self.toolButton_WorkingDirOpen.clicked.connect(
            lambda: self.openDirectoy(self.lineEdit_WorkingDirPath.text()))
        self.toolButton_ImageStackOpen.clicked.connect(
            lambda: self.openDirectoy(self.lineEdit_ImageStackPath.text()))
        self.toolButton_ImageSequenceOpen.clicked.connect(
            lambda: self.openDirectoy(self.lineEdit_ImageSequencePath.text()))
        self.toolButton_NormalizeOpen.clicked.connect(
            lambda: self.openDirectoy(self.lineEdit_NormalizePath.text()))
        self.toolButton_MipOpen.clicked.connect(
            lambda: self.openDirectoy(self.lineEdit_MipPath.text()))

        ## Select Buttons
        self.toolButton_WorkingDirSelect.clicked.connect(
            lambda: self.selectPath(self.lineEdit_WorkingDirPath))
        self.toolButton_ImageStackSelect.clicked.connect(
            lambda: self.selectFile(self.lineEdit_ImageStackPath))
        self.toolButton_ImageSequenceSelect.clicked.connect(
            lambda: self.selectPath(self.lineEdit_ImageSequencePath))
        self.toolButton_NormalizeSelect.clicked.connect(
            lambda: self.selectFile(self.lineEdit_NormalizePath))
        self.toolButton_MipSelect.clicked.connect(
            lambda: self.selectFile(self.lineEdit_MipPath))
        self.toolButton_selectAsImage1.clicked.connect(self.selectImage1)
        self.toolButton_selectAsImage2.clicked.connect(self.selectImage2)
        self.toolButton_selectImage1.clicked.connect(
            lambda: self.selectFile(self.lineEdit_selectImage1))
        self.toolButton_selectImage2.clicked.connect(
            lambda: self.selectFile(self.lineEdit_selectImage2))

        ## Command Buttons
        self.commandLinkButton_correlate.clicked.connect(self.runCorrelationModule)
        self.commandLinkButton_Reslice.clicked.connect(self.imageStack)
        self.commandLinkButton_CreateStackFile.clicked.connect(self.imageSequence)
        self.commandLinkButton_Normalize.clicked.connect(self.normalize)
        self.commandLinkButton_Mip.clicked.connect(self.mip)

        ## Help Buttons
        self.toolButton_WorkingDirHelp.clicked.connect(self.helpdoc.WorkingDir)
        self.toolButton_ImageStackHelp.clicked.connect(self.helpdoc.ImageStack)
        self.toolButton_ImageSequenceHelp.clicked.connect(self.helpdoc.ImageSequence)
        self.toolButton_NormalizeHelp.clicked.connect(self.helpdoc.Normalize)
        self.toolButton_FileListHelp.clicked.connect(self.helpdoc.FileList)
        self.toolButton_MipHelp.clicked.connect(self.helpdoc.Mip)
        self.toolButton_CorrelationHelp.clicked.connect(self.helpdoc.Correlation)

        ## Misc buttons
        self.toolButton_FileListReload.clicked.connect(self.reloadFileList)
        self.toolButton_ImageStackGetPixelSize.clicked.connect(self.getPixelSize)
        self.toolButton_ImageSequenceGetPixelSize.clicked.connect(self.getPixelSize)

        ## QLineEdits validation
        self.lineEdit_WorkingDirPath.textChanged.connect(
            lambda: self.isValidPath(self.lineEdit_WorkingDirPath))
        self.lineEdit_ImageStackPath.textChanged.connect(
            lambda: self.isValidFile(self.lineEdit_ImageStackPath))
        self.lineEdit_ImageSequencePath.textChanged.connect(
            lambda: self.isValidPath(self.lineEdit_ImageSequencePath))
        self.lineEdit_NormalizePath.textChanged.connect(
            lambda: self.isValidFile(self.lineEdit_NormalizePath))
        self.lineEdit_MipPath.textChanged.connect(
            lambda: self.isValidFile(self.lineEdit_MipPath))
        self.lineEdit_selectImage1.textChanged.connect(
            lambda: self.isValidFile(self.lineEdit_selectImage1))
        self.lineEdit_selectImage2.textChanged.connect(
            lambda: self.isValidFile(self.lineEdit_selectImage2))

        ## Progressbars
        self.progressBar_ImageStack.setVisible(False)
        self.progressBar_ImageSequence.setVisible(False)
        self.progressBar_Normalize.setVisible(False)
        self.progressBar_Mip.setVisible(False)

        # Initialize Working directory
        self.workingdir = os.path.expanduser("~")
        self.lineEdit_WorkingDirPath.setText(self.workingdir)
        self.populate_filelist(self.workingdir)

    def isValidFile(self, lineEdit):
        if lineEdit.text() == "":
            lineEdit.setStyleSheet("QLineEdit { background-color: white; }")
            lineEdit.fileIsValid = False
            lineEdit.fileIsTiff = False
        elif os.path.isfile(lineEdit.text()):
            if os.path.splitext(str(lineEdit.text()))[1] in ['.tif', '.tiff']:
                lineEdit.setStyleSheet(
                    "QLineEdit { background-color: rgba(0,255,0,80); }")
                lineEdit.fileIsValid = True
                lineEdit.fileIsTiff = True
            else:
                lineEdit.setStyleSheet(
                    "QLineEdit { background-color: rgba(255,120,0,80); }")
                lineEdit.fileIsValid = True
                lineEdit.fileIsTiff = False
        else:
            lineEdit.setStyleSheet(
                "QLineEdit { background-color: rgba(255,0,0,80); }")
            lineEdit.fileIsValid = False
            lineEdit.fileIsTiff = False

    def isValidPath(self, lineEdit):
        if lineEdit.text() == "":
            lineEdit.setStyleSheet("QLineEdit { background-color: white; }")
            if lineEdit.objectName() == 'lineEdit_WorkingDirPath':
                self.listWidget_WorkingDir.clear()
        elif os.path.isdir(lineEdit.text()):
            if lineEdit.objectName() == 'lineEdit_WorkingDirPath':
                workingdir = self.checkDirectoryPrivileges(str(self.lineEdit_WorkingDirPath.text()))
                if workingdir:
                    lineEdit.setStyleSheet(
                        "QLineEdit { background-color: rgba(0,255,0,80); }")
                    self.workingdir = workingdir
                    self.populate_filelist(self.workingdir)
                    lineEdit.setText(self.workingdir)
                else:
                    lineEdit.setStyleSheet(
                        "QLineEdit { background-color: rgba(255,0,0,80); }")
            else:
                lineEdit.setStyleSheet(
                    "QLineEdit { background-color: rgba(0,255,0,80); }")
        else:
            lineEdit.setStyleSheet(
                "QLineEdit { background-color: rgba(255,0,0,80); }")

    def focusInEvent(self, event):
        if debug:
            print(clrmsg.DEBUG, 'Got focus')

    def about(self):
        gif = os.path.join(execdir, 'icons', 'SplashScreen.gif')
        if os.path.isfile(gif):
            movie = QMovie(gif)
            splash = MovieSplashScreen(movie)
            splash.show()
            while movie.state() == QMovie.Running:
                QApplication.processEvents()
                time.sleep(0.01)
        else:
            QMessageBox.about(
                self, "About 3DCT",
                "3D Correlation Toolbox {0}\n\n".format(__version__) +
                "Copyright (C) 2016  Jan Arnold\n\n" +
                "This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License...\n\n" +
                "Developed by: Jan Arnold\nCorrelation algorithm: Vladan Lucic"
            )

    def help(self):
        url = 'http://3dct.semper.space/userguide.html'
        webbrowser.open(url, new=2, autoraise=True)

    def checkDirectoryPrivileges(self, path, question="Do you want to select another directory?"):
        try:
            testfile = tempfile.TemporaryFile(dir=path)
            testfile.close()
            return path
        except Exception:
            reply = QMessageBox.critical(
                self, "Warning",
                "I cannot write to the folder: {0}\n\nDo you want to select another directory?".format(path),
                QMessageBox.Yes | QMessageBox.No
            )
            if reply == QMessageBox.Yes:
                while True:
                    newpath = str(QFileDialog.getExistingDirectory(self, "Select directory", path))
                    if newpath:
                        try:
                            testfile = tempfile.TemporaryFile(dir=newpath)
                            testfile.close()
                            return newpath
                        except:
                            reply = QMessageBox.critical(
                                self, "Warning",
                                "I cannot write to the folder: {0}\n\n{1}".format(newpath, question),
                                QMessageBox.Yes | QMessageBox.No
                            )
                            if reply == QMessageBox.No:
                                return None
                    else:
                        return None
            else:
                return None

    def openDirectoy(self, path):
        if debug:
            print(clrmsg.DEBUG, 'Passed path value:', path)
        directory, _ = os.path.split(str(path))
        if debug:
            print(clrmsg.DEBUG, 'os split (directory):', directory)
        if os.path.isdir(directory):
            if sys.platform == 'darwin':
                call(['open', '-R', directory])
            elif sys.platform.startswith('linux'):
                call(['xdg-open', directory])
            elif sys.platform == 'win32':
                call(['explorer', directory])

    def closeEvent(self, event):
        quit_msg = "Are you sure you want to exit the\n3D Correlation Toolbox?\n\nUnsaved data will be lost!"
        reply = QMessageBox.question(self, 'Message', quit_msg, QMessageBox.Yes | QMessageBox.No)
        if reply == QMessageBox.Yes:
            if hasattr(self, "correlationModul"):
                if hasattr(self.correlationModul, "window"):
                    self.correlationModul.window.close()
                    if self.correlationModul.exitstatus == 1:
                        event.ignore()
                    else:
                        event.accept()
                else:
                    event.accept()
            else:
                event.accept()
        else:
            event.ignore()

    def selectPath(self, pathLine):
        sender = self.sender()
        path = str(QFileDialog.getExistingDirectory(self, "Select Directory", self.workingdir))
        if path:
            if sender == self.toolButton_WorkingDirSelect:
                workingdir = self.checkDirectoryPrivileges(path)
                if workingdir:
                    self.workingdir = workingdir
                    self.populate_filelist(self.workingdir)
                    pathLine.setText(self.workingdir)
            else:
                pathLine.setText(path)

    def selectFile(self, pathLine):
        path, _ = QFileDialog.getOpenFileName(self, "Select tiff image file", self.workingdir,
                                              "Image Files (*.tif *.tiff);; All (*.*)")
        if path:
            pathLine.setText(path)

    def getPixelSize(self):
        sender = self.sender()
        if sender == self.toolButton_ImageStackGetPixelSize:
            try:
                pixelSizeXY = stackProcessing.pxSize(str(self.lineEdit_ImageStackPath.text()))
                pixelSizeZ = stackProcessing.pxSize(str(self.lineEdit_ImageStackPath.text()), z=True)
                if debug:
                    print(clrmsg.DEBUG + "Pixelsize xy/z", pixelSizeXY, pixelSizeZ)
                if pixelSizeXY:
                    self.doubleSpinBox_ImageStackFocusStepSizeReslized.setValue(pixelSizeXY * 1000)
                else:
                    raise Exception('No xy pixel size information found!')
                if pixelSizeZ:
                    self.doubleSpinBox_ImageStackFocusStepSizeOrig.setValue(pixelSizeZ * 1000)
                else:
                    raise Exception('No focus step size information found!')
            except Exception as e:
                QMessageBox.warning(self, "Warning", "Unable to extract pixel size.\n\n{0}".format(e))
        elif sender == self.toolButton_ImageSequenceGetPixelSize:
            try:
                tif_file = os.path.join(str(self.lineEdit_ImageSequencePath.text()), "Tile_001-001-000_0-000.tif")
                pixelSizeXY = stackProcessing.pxSize(tif_file)
                pixelSizeZ = stackProcessing.pxSize(tif_file, z=True)
                if debug:
                    print(clrmsg.DEBUG + "Pixelsize xy/z", pixelSizeXY, pixelSizeZ)
                if pixelSizeXY:
                    self.doubleSpinBox_ImageSequenceFocusStepSizeReslized.setValue(pixelSizeXY * 1000)
                else:
                    raise Exception('No xy pixel size information found!')
                if pixelSizeZ:
                    self.doubleSpinBox_ImageSequenceFocusStepSizeOrig.setValue(pixelSizeZ * 1000)
                else:
                    raise Exception('No focus step size information found!')
            except Exception as e:
                QMessageBox.warning(self, "Warning", "Unable to extract pixel size.\n\n{0}".format(e))

    def populate_filelist(self, path):
        """
        Populate List widget for listing files needed for coordinate extraction.
        """
        self.listWidget_WorkingDir.clear()
        self.listWidget_WorkingDir.itempath = path
        ## bugfix for linux: os.listdir returns unsorted file list
        for fname in sorted(os.listdir(path)):
            checkdir = os.path.join(path, fname)
            if (
                    os.path.isdir(checkdir) is False and
                    fname.startswith(".") is False and
                    fname.startswith("$") is False
            ):
                item = QtWidgets.QListWidgetItem()
                item.setText(fname)
                # item.setFlags(item.flags() | QtCore.Qt.ItemIsUserCheckable)
                # item.setCheckState(QtCore.Qt.Unchecked)
                self.listWidget_WorkingDir.addItem(item)

    def reloadFileList(self):
        """
        Refresh the working direction file list in GUI.
        """
        if hasattr(self, "workingdir"):
            self.populate_filelist(self.workingdir)
            if debug is True: print(clrmsg.DEBUG, 'Working directory file list reloaded')

    def selectImage1(self):
        self.lineEdit_selectImage1.setText(
            os.path.join(self.listWidget_WorkingDir.itempath, self.listWidget_WorkingDir.selectedItems()[0].text()))

    def selectImage2(self):
        self.lineEdit_selectImage2.setText(
            os.path.join(self.listWidget_WorkingDir.itempath, self.listWidget_WorkingDir.selectedItems()[0].text()))

    def runCorrelationModule(self):
        if hasattr(self, 'correlationModul'):
            if hasattr(self.correlationModul, 'window'):
                QMessageBox.warning(self, "Warning",
                                    "There is already a correlation instance running. Please close it or restart the application.")
                return
        if self.lineEdit_selectImage1.text() != "" and self.lineEdit_selectImage2.text() != "":
            if self.lineEdit_selectImage1.fileIsTiff and self.lineEdit_selectImage2.fileIsTiff:
                self.correlationModul = TDCT_correlation.Main(
                    leftImage=str(self.lineEdit_selectImage1.text()),
                    rightImage=str(self.lineEdit_selectImage2.text()),
                    nosplash=False,
                    workingdir=self.workingdir)
            else:
                if not self.lineEdit_selectImage1.fileIsValid or not self.lineEdit_selectImage2.fileIsValid:
                    QMessageBox.warning(self, "Warning",
                                        "Invalid file path detected. Please check the file paths.")
                elif not self.lineEdit_selectImage1.fileIsTiff or not self.lineEdit_selectImage2.fileIsTiff:
                    QMessageBox.warning(self, "Warning",
                                        "Only *.tif and *.tiff files are supported at the moment")
        else:
            if self.lineEdit_selectImage1.text() == "":
                self.lineEdit_selectImage1.setStyleSheet("QLineEdit { background-color: rgba(255,0,0,80); }")
            if self.lineEdit_selectImage2.text() == "":
                self.lineEdit_selectImage2.setStyleSheet("QLineEdit { background-color: rgba(255,0,0,80); }")

    def imageStack(self):
        """
        Image stack file reslicing. Takes the input and output focus step size set by the user and calls the stackProcessing
        function for the selected image stack.
        The pixel size information is extracted if possible and added to the image image stack meta data for future reference (see stackProcessing.py
        for more details inc code).
        By default the new resliced image stack is saved in the same direction as the original file. This directory is
        checked for write permission. If it is a read only directory, the user is asked to select a different directory or
        to aboard the process.
        """
        self.progressBar_ImageStack.setVisible(True)
        self.progressBar_ImageStack.setMaximum(0)
        QtGui.QGuiApplication.processEvents()  # Update GUI to display progressbar
        img_path = str(self.lineEdit_ImageStackPath.text())
        customSaveDir = self.checkDirectoryPrivileges(
            os.path.split(img_path)[0], question="Do you want me to save the data to another directory?")
        if img_path and self.lineEdit_ImageStackPath.fileIsTiff is True and customSaveDir:
            ss_in = self.doubleSpinBox_ImageStackFocusStepSizeOrig.value()
            ss_out = self.doubleSpinBox_ImageStackFocusStepSizeReslized.value()
            if debug is True: print
            clrmsg.DEBUG, img_path, ss_in, ss_out, customSaveDir
            self.progressBar_ImageStack.setMaximum(100)
            QtGui.QGuiApplication.processEvents()
            stackProcessing.main(
                img_path, ss_in, ss_out, qtprocessbar=self.progressBar_ImageStack,
                interpolationmethod='linear', saveorigstack=False, showgraph=False, customSaveDir=customSaveDir)
            self.progressBar_ImageStack.reset()
            self.progressBar_ImageStack.setVisible(False)
        else:
            self.progressBar_ImageStack.setMaximum(100)
            self.progressBar_ImageStack.reset()
            self.progressBar_ImageStack.setVisible(False)

    def imageSequence(self):
        """
        Image sequence merging with optional reslicing. Takes the input and output focus step size set by the user and calls the stackProcessing
        function for the selected folder containing image sequence files from the FEI CorrSight microscope. If reslicing is not selected (checkbox),
        then the image sequence is just merged to one single stack file.
        The pixel size information is extracted if possible and added to the image image stack meta data for future reference (see stackProcessing.py
        for more details inc code).
        By default the new merged and/or resliced image stack is saved in the same direction as the original file. This directory is
        checked for write permission. If it is a read only directory, the user is asked to select a different directory or
        to aboard the process.
        """
        self.progressBar_ImageSequence.setVisible(True)
        self.progressBar_ImageSequence.setMaximum(0)
        QtGui.QGuiApplication.processEvents()
        dirPath = str(self.lineEdit_ImageSequencePath.text())
        customSaveDir = self.checkDirectoryPrivileges(dirPath,
                                                      question="Do you want me to save the data to another directory?")
        if os.path.isdir(dirPath) and customSaveDir:
            if self.checkBox_ImageSequenceCube.isChecked():
                ss_in = self.doubleSpinBox_ImageSequenceFocusStepSizeOrig.value()
                ss_out = self.doubleSpinBox_ImageSequenceFocusStepSizeReslized.value()
                if debug is True: print
                clrmsg.DEBUG, dirPath, ss_in, ss_out, str(
                    self.checkBox_ImageSequenceSaveOrigStack.isChecked()), customSaveDir
                self.progressBar_ImageSequence.setMaximum(100)
                QtGui.QGuiApplication.processEvents()
                stackProcessing.main(
                    dirPath, ss_in, ss_out, qtprocessbar=self.progressBar_ImageSequence, interpolationmethod='linear',
                    saveorigstack=self.checkBox_ImageSequenceSaveOrigStack.isChecked(), showgraph=False,
                    customSaveDir=customSaveDir)
                self.progressBar_ImageSequence.reset()
            else:
                if debug is True: print
                clrmsg.DEBUG, 'no reslicing'
                self.progressBar_ImageSequence.setMaximum(100)
                QtGui.QGuiApplication.processEvents()
                stackProcessing.main(
                    dirPath, 0, 0,
                    qtprocessbar=self.progressBar_ImageSequence,
                    saveorigstack=True, interpolationmethod='none', customSaveDir=customSaveDir)
                self.progressBar_ImageSequence.reset()
                self.progressBar_ImageSequence.setVisible(False)
        else:
            self.progressBar_ImageSequence.setMaximum(100)
            self.progressBar_ImageSequence.reset()
            self.progressBar_ImageSequence.setVisible(False)

    def normalize(self):
        self.progressBar_Normalize.setVisible(True)
        self.progressBar_Normalize.setMaximum(0)
        QtGui.QGuiApplication.processEvents()
        img_path = str(self.lineEdit_NormalizePath.text())
        customSaveDir = self.checkDirectoryPrivileges(
            os.path.split(img_path)[0], question="Do you want me to save the data to another directory?")
        if img_path and self.lineEdit_NormalizePath.fileIsTiff is True and customSaveDir:
            if debug is True: print
            clrmsg.DEBUG, 'In/out:', img_path, customSaveDir
            self.progressBar_Normalize.setMaximum(100)
            QtGui.QGuiApplication.processEvents()
            stackProcessing.normalize(img_path, qtprocessbar=self.progressBar_Normalize, customSaveDir=customSaveDir)
            self.progressBar_Normalize.reset()
            self.progressBar_Normalize.setVisible(False)
        else:
            self.progressBar_Normalize.setMaximum(100)
            self.progressBar_Normalize.reset()
            self.progressBar_Normalize.setVisible(False)

    def mip(self):
        self.progressBar_Mip.setVisible(True)
        self.progressBar_Mip.setMaximum(0)
        QtGui.QGuiApplication.processEvents()
        img_path = str(self.lineEdit_MipPath.text())
        customSaveDir = self.checkDirectoryPrivileges(
            os.path.split(img_path)[0], question="Do you want me to save the data to another directory?")
        if img_path and self.lineEdit_MipPath.fileIsTiff is True and customSaveDir:
            if debug is True: print
            clrmsg.DEBUG, 'In/out/normalize:', img_path, customSaveDir, self.checkBox_MipNormalize.isChecked()
            self.progressBar_Mip.setMaximum(100)
            QtGui.QGuiApplication.processEvents()
            stackProcessing.mip(
                img_path, qtprocessbar=self.progressBar_Mip,
                customSaveDir=customSaveDir, normalize=self.checkBox_MipNormalize.isChecked())
            self.progressBar_Mip.reset()
            self.progressBar_Mip.setVisible(False)
        else:
            self.progressBar_Mip.setMaximum(100)
            self.progressBar_Mip.reset()
            self.progressBar_Mip.setVisible(False)

class MovieSplashScreen(QSplashScreen):
    def __init__(self, movie,  parent=None):
        movie.jumpToFrame(0)
        pixmap = QPixmap(movie.frameRect().size())
        super(MovieSplashScreen, self).__init__(pixmap, QtCore.Qt.WindowStaysOnTopHint)
        self.movie = movie
        self.movie.frameChanged.connect(self.repaint)
        self.aboutText = ("Max-Planck-Institute of Biochemistry\n\n"
                          "Developed by: Jan Arnold\n"
                          "Correlation algorithm: Vladan Lucic")
        self.versionLicText = ("Copyright (C) 2016  Jan Arnold\n\n"
                               "This program is free software: you can redistribute it and/or modify it\n"
                               "under the terms of the GNU General Public License as published by \n"
                               "the Free Software Foundation, either version 3 of the License, or \n"
                               "(at your option) any later version.\n\n"
                               "This program is distributed in the hope that it will be useful, \n"
                               "but WITHOUT ANY WARRANTY; without even the implied warranty of \n"
                               "MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. \n"
                               "See the GNU General Public License for more details.\n\n"
                               "You should have received a copy of the GNU General Public License \n"
                               "along with this program.  If not, see <http://www.gnu.org/licenses/>.\n\n" + str(
            __version__))

    def keyPressEvent(self, event):
        if event.key() == QtCore.Qt.Key_Escape:
            self.movie.stop()

    def showEvent(self, event):
        self.movie.start()

    def hideEvent(self, event):
        self.movie.stop()

    def paintEvent(self, event):
        painter = QPainter(self)
        pixmap = self.movie.currentPixmap()
        self.setMask(pixmap.mask())
        painter.drawPixmap(0, 0, pixmap)
        painter.setPen(QtCore.Qt.white)
        painter.drawText(0, 0, pixmap.size().width() - 3, pixmap.size().height() - 1,
                         QtCore.Qt.AlignBottom | QtCore.Qt.AlignRight, self.versionLicText)
        painter.drawText(5, 0, pixmap.size().width(), pixmap.size().height() - 5,
                         QtCore.Qt.AlignBottom | QtCore.Qt.AlignLeft, self.aboutText)

    def sizeHint(self):
        return self.movie.scaledSize()


class GenericThread(QtCore.QThread):
    def __init__(self, function, *args, **kwargs):
        super(GenericThread, self).__init__()
        self.function = function
        self.args = args
        self.kwargs = kwargs

    def __del__(self):
        self.wait()

    def run(self):
        self.function(*self.args, **self.kwargs)


if __name__ == "__main__":
    if debug:
        print(clrmsg.DEBUG + 'Debug active')
        print(clrmsg.OK + 'Main imports OK')
        print(clrmsg.INFO + 'This is 3D Correlation Toolbox', __version__)
        print(clrmsg.WARNING + 'Debug mode can/will slow down parts of the Toolbox (e.g. marker clicking)')
    app = QApplication(sys.argv)
    window = APP()
    window.show()
    window.raise_()  # Optionally, use window.raise() if needed
    sys.exit(app.exec_())
