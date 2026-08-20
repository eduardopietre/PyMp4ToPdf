import tkinter as tk
import tkinter.ttk as tkk
import queue
from tkinter import filedialog, font, messagebox
from threading import Thread

from mp4_to_pdf_core import (
    collect_sampled_frames,
    diff_pairs,
    iter_sampled_frames,
    save_bgr_frames_as_pdf,
    ssim_filter_pairs,
    unique_frames_from_iterable,
)


def to_per_mile(num, div):
    return round(num / div * 1000)


class Mp4ToPdfWorker(Thread):
    UPDATE_READING = 1
    UPDATE_DIFF = 2
    UPDATE_SMI = 3
    DONE = 4

    def __init__(self, _queue, infile, out, n_frame, diff_threshold, ssim_threshold):
        super().__init__()
        self.queue = _queue
        self.infile = infile
        self.out = out
        self.n_frame = n_frame
        self.diff_threshold = diff_threshold
        self.ssim_threshold = ssim_threshold

    def run(self):
        self.convert()

    def get_images(self):
        def on_progress(count, length):
            self.queue.put((self.UPDATE_READING, to_per_mile(count + 1, length)))

        images = collect_sampled_frames(
            self.infile,
            self.n_frame,
            on_progress=on_progress,
        )
        self.queue.put((self.UPDATE_READING, 1000))
        return images

    def diff_filter(self, images):
        pairs = diff_pairs(
            images,
            self.diff_threshold,
            on_progress=lambda current, total: self.queue.put(
                (self.UPDATE_DIFF, to_per_mile(current, total))
            ),
        )
        self.queue.put((self.UPDATE_DIFF, 1000))
        return pairs

    def structural_similarity_filter(self, pairs):
        fails = ssim_filter_pairs(
            pairs,
            self.ssim_threshold,
            on_progress=lambda current, total: self.queue.put(
                (self.UPDATE_SMI, to_per_mile(current, total))
            ),
        )
        self.queue.put((self.UPDATE_DIFF, 1000))
        return fails

    def save_as_pdf(self, images):
        save_bgr_frames_as_pdf(self.out, images)

    def convert(self):
        length_holder = {"length": 1}

        def on_open(length, fps):
            length_holder["length"] = max(length, 1)

        def on_progress(count, length):
            self.queue.put((self.UPDATE_READING, to_per_mile(count + 1, length)))

        def on_pair(index):
            estimated = max(1, (length_holder["length"] + self.n_frame - 1) // self.n_frame)
            self.queue.put((self.UPDATE_DIFF, to_per_mile(index, estimated)))

        def on_ssim(index):
            estimated = max(1, (length_holder["length"] + self.n_frame - 1) // self.n_frame)
            self.queue.put((self.UPDATE_SMI, to_per_mile(index, estimated)))

        uniques, _image_count, _pair_count = unique_frames_from_iterable(
            iter_sampled_frames(
                self.infile,
                self.n_frame,
                on_open=on_open,
                on_progress=on_progress,
            ),
            self.diff_threshold,
            self.ssim_threshold,
            on_pair=on_pair,
            on_ssim=on_ssim,
        )
        self.queue.put((self.UPDATE_READING, 1000))
        self.queue.put((self.UPDATE_DIFF, 1000))
        self.queue.put((self.UPDATE_DIFF, 1000))
        self.save_as_pdf(uniques)
        self.queue.put((self.DONE, 0))


class MainWindow:

    def __init__(self, root, width, height):
        self.root = root

        self.queue = queue.Queue()
        self.file_path = None

        self.setup_gui(width, height)
        self.refresh()


    def setup_gui(self, width, height):
        pady = 6

        # Global
        self.root.title("MP4 To PDF - GUI")
        self.root.geometry(f"{width}x{height}")
        self.root.resizable(0, 0)

        # Font
        self.default_font = font.nametofont("TkDefaultFont")
        self.default_font.configure(family="Arial", size=11)

        # Button Select File
        self.btn_select_file = tk.Button(self.root, text="Select File", command=self.select_mp4, width=30)
        self.btn_select_file.pack(pady=pady)

        # Selected Label
        self.lbl_selected = tk.Label(self.root, text="Selected: None", font="Arial 9", wraplength=400, height=4, justify="center")
        self.lbl_selected.pack(pady=pady)

        # Frame Skip label and spinbox
        frame1 = tk.Frame(self.root)
        tk.Label(frame1, text="Frame Skip:").pack(pady=pady, side=tk.LEFT)

        self.cf_nframe = tk.IntVar(value=150)

        self.sb_nframe = tk.Spinbox(frame1, from_=1, to=2000, textvariable=self.cf_nframe, width=5, font="Arial 10")
        self.sb_nframe.pack(padx=10, pady=pady, side=tk.RIGHT)
        frame1.pack()

        # Difference Threshold label and spinbox
        frame2 = tk.Frame(self.root)
        tk.Label(frame2, text="Difference Threshold (%):").pack(pady=pady, side=tk.LEFT)

        self.cf_diff = tk.IntVar(value=90)

        self.sb_diff = tk.Spinbox(frame2, from_=1, to=100, textvariable=self.cf_diff, width=5, font="Arial 10")
        self.sb_diff.pack(padx=10, pady=pady, side=tk.RIGHT)
        frame2.pack()

        # Structural Similarity Threshold label and spinbox
        frame3 = tk.Frame(self.root)
        tk.Label(frame3, text="Structural Similarity Threshold (%):").pack(pady=pady, side=tk.LEFT)

        self.cf_smi = tk.IntVar(value=90)

        self.sb_smi = tk.Spinbox(frame3, from_=1, to=100, textvariable=self.cf_smi, width=5, font="Arial 10")
        self.sb_smi.pack(padx=10, pady=pady, side=tk.RIGHT)
        frame3.pack()

        # Reading file loading bar
        frame4 = tk.Frame(self.root)
        tk.Label(frame4, text="Reading File:", width=20).pack(padx=10, pady=pady, side=tk.LEFT)

        self.bar1 = tkk.Progressbar(frame4, length=150, mode='determinate', maximum=1000)
        self.bar1.pack(pady=pady, side=tk.RIGHT)
        frame4.pack()

        # Calculating differences loading bar
        frame5 = tk.Frame(self.root)
        tk.Label(frame5, text="Calculating Differences:", width=20).pack(padx=10, pady=pady, side=tk.LEFT)

        self.bar2 = tkk.Progressbar(frame5, length=150, mode='determinate', maximum=1000)
        self.bar2.pack(pady=pady, side=tk.RIGHT)
        frame5.pack()

        # Calculating structural similarities loading bar
        frame6 = tk.Frame(self.root)
        tk.Label(frame6, text="Calculating Structural Similarities:", width=20, wraplength=200).pack(padx=10, pady=pady, side=tk.LEFT)

        self.bar3 = tkk.Progressbar(frame6, length=150, mode='determinate', maximum=1000)
        self.bar3.pack(pady=pady, side=tk.RIGHT)
        frame6.pack()

       # bar['maximum'] = 5
        #bar["value"] = 1

        # Button Convert
        self.btn_convert = tk.Button(self.root, text="Convert", command=self.convert, width=30)
        self.btn_convert.pack(pady=pady)


    def out_file(self):
        return f"{self.file_path.replace('.mp4', '')}.pdf"


    def convert(self):
        self.bar1["value"] = 0
        self.bar2["value"] = 0
        self.bar3["value"] = 0

        if not self.file_path:
            tk.messagebox.showwarning("Invalid File", "No selected file.")
            return

        nframe = self.cf_nframe.get()
        diff = self.cf_diff.get()
        smi = self.cf_smi.get()

        if nframe < 1 or nframe > 2000:
            tk.messagebox.showerror("Error", "'Frame Skip' must be between 1 and 2000.")
            return

        if (diff < 1 or diff > 100) or (smi < 1 or smi > 100):
            tk.messagebox.showerror("Error", "'Difference Threshold' and 'Structural Similarity Threshold' must be between 1 and 100.")
            return

        self.btn_convert["state"] = "disabled"

        self.thread = Mp4ToPdfWorker(self.queue, self.file_path, self.out_file(), nframe, diff * 0.01, smi * 0.01) # important * 0.01
        self.thread.start()


    def select_mp4(self):
        self.file_path = filedialog.askopenfilename(filetypes=[("MP4", ".mp4")])
        if self.file_path:
            self.lbl_selected["text"] = f"Selected: '{self.file_path}'"
        else:
            self.lbl_selected["text"] = f"Selected: None"


    def update_ui(self, code, value):
        if code == Mp4ToPdfWorker.UPDATE_READING:
            self.bar1["value"] = value
        elif code == Mp4ToPdfWorker.UPDATE_DIFF:
            self.bar2["value"] = value
        elif code == Mp4ToPdfWorker.UPDATE_SMI:
            self.bar3["value"] = value
        elif code == Mp4ToPdfWorker.DONE:
            self.btn_convert["state"] = "normal"
            tk.messagebox.showinfo("Done", f"Done. File exported as '{self.out_file()}'.")


    def refresh(self):
        while not self.queue.empty():
            data = self.queue.get()
            self.update_ui(data[0], data[1])

        self.root.after(200, self.refresh)


def center(obj, w, h):
    ws = obj.winfo_screenwidth()
    hs = obj.winfo_screenheight()
    x = (ws / 2) - (w / 2)
    y = (hs / 2) - (h / 2)
    obj.geometry(f"{int(w)}x{int(h)}+{int(x)}+{int(y)}")


if __name__ == "__main__":
    root = tk.Tk()

    w = 500
    h = 400

    ui = MainWindow(root, w, h)
    center(root, w, h)

    tk.mainloop()
