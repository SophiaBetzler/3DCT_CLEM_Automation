import os
import sys

import numpy as np
from napari.qt.threading import thread_worker
from PyQt5.QtCore import pyqtSignal
from PyQt5.QtWidgets import QApplication, QDialog

from tdct.ui import tdct_fm_import_dialog
from tdct.util import INTERPOLATION_METHODS, multi_channel_interpolation
from tdct.io import load_and_parse_fm_image

# TODO: save interpolated image
# TODO: disable interaction while interpolating
# TODO: parse colour from image metadata

class FluorescenceImportDialog(QDialog, tdct_fm_import_dialog.Ui_Dialog):
    progress_update = pyqtSignal(dict)

    def __init__(self, path: str, parent=None):
        super().__init__()
        self.setupUi(self)
        self.parent = parent

        self.path = path

        self.image, self.md = load_and_parse_fm_image(path)

        self.setWindowTitle("Import Fluorescence Image")

        self.fm_image: np.ndarray = None
        self.image_interp: np.ndarray = None
        self.accepted_image: bool = False

        self.setup_connections()

    def setup_connections(self):
        self.pushButton_interpolate.clicked.connect(self.on_interpolate)
        self.label_description.setText(f"File: {os.path.basename(self.path)}")

        self.progress_update.connect(self.update_progress)
        self.pushButton_load_image.clicked.connect(self.on_load_image)
        self.pushButton_load_interpolated_image.clicked.connect(
            self.on_load_interpolated_image
        )
        self.pushButton_load_interpolated_image.setVisible(False)

        self.comboBox_interpolation_method.addItems(INTERPOLATION_METHODS)
        self.comboBox_interpolation_method.setCurrentText(INTERPOLATION_METHODS[0])

        # hide progress bar
        self.progressBar_interpolation.setVisible(False)

        # set pixel size
        pixel_size = self.md.get("pixel_size", None)
        zstep = self.md.get("zstep", None)

        if pixel_size is not None:
            self.doubleSpinBox_target_step_size.setValue(pixel_size * 1e9)
        if zstep is not None:
            self.doubleSpinBox_current_step_size.setValue(zstep * 1e9)

    def update_progress(self, ddict: dict):
        val = ddict["value"]
        max = ddict["max"]
        prog = int(val / max * 100)

        self.progressBar_interpolation.setVisible(True)
        self.progressBar_interpolation.setValue(prog)
        self.progressBar_interpolation.setFormat(
            f"Interpolating Channel: {val+1}/{max}"
        )

        print(f"Progress: {prog}%")

    def on_interpolate(self):
        self.progressBar_interpolation.setVisible(True)

        image = self.image
        zstep = self.doubleSpinBox_current_step_size.value()  # nm
        pixel_size = self.doubleSpinBox_target_step_size.value()  # nm
        interpolation_method = self.comboBox_interpolation_method.currentText()

        self.worker = self._interpolate_worker(
            image=image,
            current_pixelsize=zstep,
            target_pixelsize=pixel_size,
            method=interpolation_method,
        )
        self.worker.finished.connect(self._workflow_finished)
        self.worker.errored.connect(self._workflow_aborted)
        self.worker.start()

    def _workflow_finished(self):
        self.progressBar_interpolation.setVisible(False)
        self.worker = None

        if self.image_interp is not None:
            self.pushButton_load_interpolated_image.setVisible(True)

    def _workflow_aborted(self, exc):
        self.progressBar_interpolation.setVisible(False)
        self.worker = None

    @thread_worker
    def _interpolate_worker(
        self,
        image: np.ndarray,
        current_pixelsize: float,
        target_pixelsize: float,
        method: str = "spline",
    ):
        """Interpolation worker"""
        image_interp = multi_channel_interpolation(
            image=image,
            pixelsize_in=current_pixelsize,
            pixelsize_out=target_pixelsize,
            method=method,
            parent_ui=self,
        )

        self.image_interp = image_interp

    def on_load_image(self):
        self.fm_image = self.image
        self.accepted_image = True

        self.close()

    def on_load_interpolated_image(self):
        self.fm_image = self.image_interp
        self.accepted_image = True
        self.close()


PATH = "/home/patrick/github/3DCT/3D_correlation_test_dataset/test-image.ome.tiff"


def main():
    app = QApplication(sys.argv)
    dialog = FluorescenceImportDialog(path=PATH)
    _ = dialog.exec_()


if __name__ == "__main__":
    main()
