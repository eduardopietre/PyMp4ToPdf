import sys
import os
import cv2
import numpy as np
import PyQt5.QtWidgets as QtWidgets
import PyQt5.QtCore as QtCore
import PyQt5.QtGui as QtGui
from skimage.metrics import structural_similarity
from PIL import Image


CALLBACK_READING_FILE = 1
CALLBACK_DIFFERENCE = 2
CALLBACK_SSMI = 3
CALLBACK_DONE = 5


class ConverterThread(QtCore.QThread):

    def __init__(self, parent, args_getter):
        super(ConverterThread, self).__init__(parent)
        self.args_getter = args_getter

    def run(self):
        infile, out, n_frame, diff_threshold, ssim_threshold, callback = self.args_getter()
        converter = Mp4ToPdf(infile, out, n_frame, diff_threshold, ssim_threshold, callback)
        converter.convert()


class Mp4ToPdf:
    def __init__(self, infile, out, n_frame, diff_threshold, ssim_threshold, callback=None):
        self.infile = infile
        self.out = out
        self.n_frame = n_frame
        self.diff_threshold = diff_threshold
        self.ssim_threshold = ssim_threshold
        self.callback = callback

    def call_callback(self, code, progress=1, length=1):
        if self.callback:
            pct = round(progress / length) * 100
            self.callback(code, pct)

    def get_images(self):
        video = cv2.VideoCapture(self.infile)
        count = 0
        images = []

        length = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

        while video.isOpened():
            success, image = video.read()

            if success:
                images.append(image[:,:,::-1])  # cv2 reads as BRG, [:,:,::-1] converts it to RGB.
                count += self.n_frame
                video.set(1, count)

                self.call_callback(CALLBACK_READING_FILE, count + 1, length)
            else:
                video.release()
                break

        self.call_callback(CALLBACK_READING_FILE, length, length)
        return images


    def diff_filter(self, images):
        pairs = []

        for i in range(1, len(images)):
            img = images[i]
            prev_img = images[i - 1]

            diff = img - prev_img
            equal_pct = np.mean(np.abs(diff) < 0.01)  # 0.01 to prevent float pointing.

            if equal_pct < self.diff_threshold:
                pairs.append([img, prev_img])

            self.call_callback(CALLBACK_DIFFERENCE, i + 1, len(images))

        return pairs


    def structural_similarity_filter(self, pairs):
        fails = []

        for i, p in enumerate(pairs):
            ssim = structural_similarity(p[0], p[1], multichannel=True)
            if ssim < self.ssim_threshold:
                fails.append(p)
            self.call_callback(CALLBACK_SSMI, i + 1, len(pairs))

        return fails


    def save_as_pdf(self, images):
        as_images = [Image.fromarray(image) for image in images]
        as_images[0].save(self.out, "PDF", resolution=100.0, save_all=True, append_images=as_images[1:])

    def convert(self):
        images = self.get_images()
        diff_pairs = self.diff_filter(images)
        changes = self.structural_similarity_filter(diff_pairs)
        uniques = [e[0] for e in changes]
        self.save_as_pdf(uniques)
        self.call_callback(CALLBACK_DONE)


def create_button(text, action=None):
    btn = QtWidgets.QPushButton()
    btn.setText(text)
    if action:
        btn.clicked.connect(action)
    return btn

def create_label(text, fixed_width=None):
    lbl = QtWidgets.QLabel(text)
    if fixed_width:
        lbl.setFixedWidth(fixed_width)
    return lbl

def create_int_line_edit(placeholder, fixed_width=None, align=QtCore.Qt.AlignCenter):
    qline = QtWidgets.QLineEdit()
    qline.setText(placeholder)
    qline.setValidator(QtGui.QIntValidator())
    if fixed_width:
        qline.setFixedWidth(fixed_width)
    if align:
        qline.setAlignment(align)
    return qline

def create_horizontal_box(*objects):
    h_box = QtWidgets.QHBoxLayout()
    for objc in objects:
        h_box.addWidget(objc)
    return h_box

def create_loading_bar(fixed_width=None):
    bar = QtWidgets.QProgressBar()
    if fixed_width:
        bar.setFixedWidth(fixed_width)
    return bar

class MainWindow(object):

    def __init__(self, window):
        window.setObjectName("MainWindow")
        window.setFixedSize(400, 200)
        window.setWindowTitle("MP4 To PDF - GUI")

        self.converter = ConverterThread(None, self.get_converter_args)

        self.file_path = ""
        self.out_file = ""

        self.central_widget = QtWidgets.QWidget(window)
        self.layout = QtWidgets.QVBoxLayout(self.central_widget)

        self.layout.addWidget(create_button("Select File", self.select_mp4))

        self.lbl_selected_file = create_label("None", 200)
        self.layout.addLayout(create_horizontal_box(
            create_label("Selected MP4 File:"),
            self.lbl_selected_file
        ))

        self.cf_nframe = create_int_line_edit("100", 200)
        self.layout.addLayout(create_horizontal_box(
            create_label("Frame Skip:"),
            self.cf_nframe
        ))

        self.cf_diff = create_int_line_edit("90", 200)
        self.layout.addLayout(create_horizontal_box(
            create_label("Difference Threshold:"),
            self.cf_diff
        ))

        self.cf_smi = create_int_line_edit("90", 200)
        self.layout.addLayout(create_horizontal_box(
            create_label("SMI Threshold:"),
            self.cf_smi
        ))

        self.progress_bar1 = create_loading_bar(200)
        self.layout.addLayout(create_horizontal_box(
            create_label("Reading File:"),
            self.progress_bar1
        ))

        self.progress_bar2 = create_loading_bar(200)
        self.layout.addLayout(create_horizontal_box(
            create_label("Calculating differences:"),
            self.progress_bar2
        ))

        self.progress_bar3 = create_loading_bar(200)
        self.layout.addLayout(create_horizontal_box(
            create_label("Calculating structural similarity:"),
            self.progress_bar3
        ))

        self.btn_convert = create_button("Convert to PDF", self.convert)
        self.layout.addWidget(self.btn_convert)

        window.setCentralWidget(self.central_widget)

    def progress_callback(self, code, progress_pct):
        if code == CALLBACK_DONE:
            msg = QtWidgets.QMessageBox()
            msg.setIcon(QtWidgets.QMessageBox.Information)
            msg.setText("Done")
            msg.setInformativeText(f"Conversion done. Exported as ")
            msg.setWindowTitle("Done")
            msg.exec()
        else:
            lookup = {
                CALLBACK_READING_FILE : self.progress_bar1,
                CALLBACK_DIFFERENCE : self.progress_bar2,
                CALLBACK_SSMI : self.progress_bar3,
            }
            lookup[code].setValue(progress_pct)

    def get_converter_args(self):
        # order: infile, out, n_frame, diff_threshold, ssim_threshold, callback
        return (
            self.file_path,
            self.out_file,
            int(self.cf_nframe.text()),
            int(self.cf_diff.text()),
            int(self.cf_smi.text()),
            self.progress_callback
        )

    def convert(self):
        nframe = int(self.cf_nframe.text())
        diff = int(self.cf_diff.text())
        smi = int(self.cf_smi.text())

        self.out_file = f"{self.file_path.replace('.mp4', '')}.pdf"

        if nframe < 1 or nframe > 2000:
            msg = QtWidgets.QMessageBox()
            msg.setIcon(QtWidgets.QMessageBox.Critical)
            msg.setText("Error")
            msg.setInformativeText("'Frame Skip' must be between 1 and 2000.")
            msg.setWindowTitle("Error")
            msg.exec_()
            return

        if (diff < 1 or diff > 100) or (smi < 1 or smi > 100):
            msg = QtWidgets.QMessageBox()
            msg.setIcon(QtWidgets.QMessageBox.Critical)
            msg.setText("Error")
            msg.setInformativeText("'Difference Threshold' and 'SMI Threshold' must be between 1 and 100.")
            msg.setWindowTitle("Error")
            msg.exec_()
            return

        self.converter.start()
        self.btn_convert.setEnabled(False)

    def select_mp4(self):
        self.file_path, _ = MainWindow.dialog_file_path("Select the mp4 file", "mp4(*.mp4)")
        self.lbl_selected_file.setText(self.file_path)

    @classmethod
    def dialog_file_path(cls, title, file_filter):
        return QtWidgets.QFileDialog.getOpenFileName(None, title, os.getcwd(), file_filter)




if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    app.setStyle("Fusion") # ['Breeze', 'Oxygen', 'QtCurve', 'Windows', 'Fusion']
    window = QtWidgets.QMainWindow()

    ui = MainWindow(window)

    window.show()
    sys.exit(app.exec_())
