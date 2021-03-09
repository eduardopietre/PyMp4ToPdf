import sys
import os
import cv2
import numpy as np
import PyQt5.QtWidgets as QtWidgets
import PyQt5.QtCore as QtCore
import PyQt5.QtGui as QtGui
from skimage.metrics import structural_similarity
from PIL import Image


def to_pct(num, div):
    return round(num / div * 100)


class Mp4ToPdfWorker(QtCore.QThread):
    reading_file_update = QtCore.pyqtSignal(int)
    difference_update = QtCore.pyqtSignal(int)
    ssmi_update = QtCore.pyqtSignal(int)
    done = QtCore.pyqtSignal()

    def __init__(self, infile, out, n_frame, diff_threshold, ssim_threshold):
        super().__init__()
        self.infile = infile
        self.out = out
        self.n_frame = n_frame
        self.diff_threshold = diff_threshold
        self.ssim_threshold = ssim_threshold

    def run(self):
        self.convert()

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

                self.reading_file_update.emit(to_pct(count + 1, length))
            else:
                video.release()
                break

        self.reading_file_update.emit(100)
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

            self.difference_update.emit(to_pct(i + 1, len(images)))

        return pairs


    def structural_similarity_filter(self, pairs):
        fails = []

        for i, p in enumerate(pairs):
            ssim = structural_similarity(p[0], p[1], multichannel=True)
            if ssim < self.ssim_threshold:
                fails.append(p)
            self.ssmi_update.emit(to_pct(i + 1, len(pairs)))

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
        self.done.emit()


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

def create_message_box(alert_type, title, text, information):
    msg = QtWidgets.QMessageBox()
    msg.setIcon(alert_type)
    msg.setText(text)
    msg.setInformativeText(information)
    msg.setWindowTitle(title)
    return msg

class MainWindow(object):

    def __init__(self, window):
        window.setObjectName("MainWindow")
        window.setFixedSize(700, 300)
        window.setWindowTitle("MP4 To PDF - GUI")

        width = 450

        self.file_path = ""

        self.central_widget = QtWidgets.QWidget(window)
        self.layout = QtWidgets.QVBoxLayout(self.central_widget)

        self.layout.addWidget(create_button("Select File", self.select_mp4))

        self.lbl_selected_file = create_label("None", width)
        self.lbl_selected_file.setObjectName("select_file")
        self.layout.addLayout(create_horizontal_box(
            create_label("Selected MP4 File:"),
            self.lbl_selected_file
        ))

        self.cf_nframe = create_int_line_edit("100", width)
        self.layout.addLayout(create_horizontal_box(
            create_label("Frame Skip:"),
            self.cf_nframe
        ))

        self.cf_diff = create_int_line_edit("90", width)
        self.layout.addLayout(create_horizontal_box(
            create_label("Difference Threshold:"),
            self.cf_diff
        ))

        self.cf_smi = create_int_line_edit("90", width)
        self.layout.addLayout(create_horizontal_box(
            create_label("SMI Threshold:"),
            self.cf_smi
        ))

        self.progress_bar1 = create_loading_bar(width)
        self.layout.addLayout(create_horizontal_box(
            create_label("Reading File:"),
            self.progress_bar1
        ))

        self.progress_bar2 = create_loading_bar(width)
        self.layout.addLayout(create_horizontal_box(
            create_label("Calculating differences:"),
            self.progress_bar2
        ))

        self.progress_bar3 = create_loading_bar(width)
        self.layout.addLayout(create_horizontal_box(
            create_label("Calculating structural similarity:"),
            self.progress_bar3
        ))

        self.btn_convert = create_button("Convert to PDF", self.convert)
        self.layout.addWidget(self.btn_convert)

        window.setCentralWidget(self.central_widget)

        self.progress_bar1.setValue(0)
        self.progress_bar2.setValue(0)
        self.progress_bar3.setValue(0)

    def out_file(self):
        return f"{self.file_path.replace('.mp4', '')}.pdf"

    def convert(self):
        if not self.file_path:
            create_message_box(QtWidgets.QMessageBox.Critical, "Error", "Error", "No selected file.").exec_()
            return

        nframe = int(self.cf_nframe.text())
        diff = int(self.cf_diff.text())
        smi = int(self.cf_smi.text())

        if nframe < 1 or nframe > 2000:
            create_message_box(QtWidgets.QMessageBox.Critical, "Error", "Error", "'Frame Skip' must be between 1 and 2000.").exec_()
            return

        if (diff < 1 or diff > 100) or (smi < 1 or smi > 100):
            create_message_box(QtWidgets.QMessageBox.Critical, "Error", "Error", "'Difference Threshold' and 'SMI Threshold' must be between 1 and 100.").exec_()
            return

        self.btn_convert.setEnabled(False)

        self.thread = Mp4ToPdfWorker(self.file_path, self.out_file(), nframe, diff * 0.01, smi * 0.01) # important * 0.01

        self.thread.reading_file_update.connect(self.signal_reading_file_update)
        self.thread.difference_update.connect(self.signal_difference_update)
        self.thread.ssmi_update.connect(self.signal_ssmi_update)
        self.thread.done.connect(self.signal_done)

        self.thread.start()

    def signal_reading_file_update(self, pct):
        self.progress_bar1.setValue(pct)

    def signal_difference_update(self, pct):
        self.progress_bar2.setValue(pct)

    def signal_ssmi_update(self, pct):
        self.progress_bar3.setValue(pct)

    def signal_done(self):
        self.btn_convert.setEnabled(True)

        self.progress_bar1.setValue(0)
        self.progress_bar2.setValue(0)
        self.progress_bar3.setValue(0)

        create_message_box(QtWidgets.QMessageBox.Information, "Done", "Done", f"Conversion done. Exported as '{self.out_file()}'.").exec_()

    def select_mp4(self):
        self.file_path, _ = MainWindow.dialog_file_path("Select the mp4 file", "mp4(*.mp4)")
        self.lbl_selected_file.setText(self.file_path)

    @classmethod
    def dialog_file_path(cls, title, file_filter):
        return QtWidgets.QFileDialog.getOpenFileName(None, title, os.getcwd(), file_filter)


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    app.setStyle("Fusion") # ['Breeze', 'Oxygen', 'QtCurve', 'Windows', 'Fusion']
    app.setStyleSheet("""
        QLabel{font-size: 12pt;}
        QPushButton{font-size: 12pt;}
        QMessageBox{font-size: 12pt;}
        QLineEdit{font-size: 12pt;}
        QProgressBar{font-size: 12pt;background-color:#E0E0E0;}
    """)
    window = QtWidgets.QMainWindow()

    ui = MainWindow(window)

    window.show()
    sys.exit(app.exec_())
