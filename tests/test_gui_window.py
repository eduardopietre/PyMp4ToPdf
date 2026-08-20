import tkinter as tk
from unittest.mock import MagicMock

import pytest

from mp4_to_pdf_gui import MainWindow, Mp4ToPdfWorker, center


@pytest.fixture
def main_window(monkeypatch):
    root = tk.Tk()
    root.withdraw()
    monkeypatch.setattr(root, "after", lambda *args, **kwargs: "after-id")
    window = MainWindow(root, 500, 400)
    yield window
    try:
        root.destroy()
    except tk.TclError:
        pass


@pytest.mark.gui
def test_window_title(main_window):
    assert main_window.root.title() == "MP4 To PDF - GUI"


@pytest.mark.gui
def test_window_is_not_resizable(main_window):
    assert main_window.root.resizable() == (False, False) or main_window.root.resizable() == (0, 0)


@pytest.mark.gui
def test_default_control_values(main_window):
    assert main_window.cf_nframe.get() == 150
    assert main_window.cf_diff.get() == 90
    assert main_window.cf_smi.get() == 90
    assert main_window.file_path is None
    assert main_window.lbl_selected["text"] == "Selected: None"
    assert str(main_window.btn_convert["state"]) == "normal"


@pytest.mark.gui
def test_progress_bars_start_at_zero(main_window):
    assert main_window.bar1["value"] == 0
    assert main_window.bar2["value"] == 0
    assert main_window.bar3["value"] == 0
    assert main_window.bar1["maximum"] == 1000
    assert main_window.bar2["maximum"] == 1000
    assert main_window.bar3["maximum"] == 1000


@pytest.mark.gui
def test_button_labels(main_window):
    assert main_window.btn_select_file["text"] == "Select File"
    assert main_window.btn_convert["text"] == "Convert"


@pytest.mark.gui
@pytest.mark.parametrize(
    "path, expected",
    [
        ("C:/videos/talk.mp4", "C:/videos/talk.pdf"),
        ("lecture.mp4", "lecture.pdf"),
        ("C:/a.mp4/b.mp4", "C:/a/b.pdf"),
        ("file.MP4", "file.MP4.pdf"),
        ("noext", "noext.pdf"),
    ],
)
def test_out_file_replacement(main_window, path, expected):
    main_window.file_path = path
    assert main_window.out_file() == expected


@pytest.mark.gui
def test_out_file_with_none_path_raises(main_window):
    main_window.file_path = None
    with pytest.raises(AttributeError):
        main_window.out_file()


@pytest.mark.gui
def test_convert_without_file_warns(main_window, monkeypatch):
    warn = MagicMock()
    monkeypatch.setattr("tkinter.messagebox.showwarning", warn)
    main_window.convert()
    warn.assert_called_once_with("Invalid File", "No selected file.")
    assert str(main_window.btn_convert["state"]) == "normal"


@pytest.mark.gui
@pytest.mark.parametrize("nframe", [0, -1, 2001, 5000])
def test_convert_rejects_invalid_frame_skip(main_window, monkeypatch, nframe):
    error = MagicMock()
    monkeypatch.setattr("tkinter.messagebox.showerror", error)
    main_window.file_path = "a.mp4"
    main_window.cf_nframe.set(nframe)
    main_window.convert()
    error.assert_called_once()
    assert "Frame Skip" in error.call_args[0][1]
    assert str(main_window.btn_convert["state"]) == "normal"


@pytest.mark.gui
@pytest.mark.parametrize("nframe", [1, 24, 150, 2000])
def test_convert_accepts_valid_frame_skip(main_window, monkeypatch, nframe):
    started = {}

    class FakeWorker:
        def __init__(self, *args, **kwargs):
            started["args"] = args
            started["kwargs"] = kwargs

        def start(self):
            started["started"] = True

    monkeypatch.setattr("mp4_to_pdf_gui.Mp4ToPdfWorker", FakeWorker)
    main_window.file_path = "a.mp4"
    main_window.cf_nframe.set(nframe)
    main_window.convert()
    assert started["started"] is True
    assert started["args"][3] == nframe


@pytest.mark.gui
@pytest.mark.parametrize("diff,smi", [(0, 90), (90, 0), (101, 90), (90, 101), (-1, 50), (50, -5)])
def test_convert_rejects_invalid_thresholds(main_window, monkeypatch, diff, smi):
    error = MagicMock()
    monkeypatch.setattr("tkinter.messagebox.showerror", error)
    main_window.file_path = "a.mp4"
    main_window.cf_diff.set(diff)
    main_window.cf_smi.set(smi)
    main_window.convert()
    error.assert_called_once()
    assert "Threshold" in error.call_args[0][1]


@pytest.mark.gui
@pytest.mark.parametrize("diff,smi", [(1, 1), (1, 100), (100, 1), (90, 90), (100, 100)])
def test_convert_scales_percent_thresholds(main_window, monkeypatch, diff, smi):
    started = {}

    class FakeWorker:
        def __init__(self, q, infile, out, nframe, diff_t, ssim_t):
            started["diff"] = diff_t
            started["ssim"] = ssim_t

        def start(self):
            started["started"] = True

    monkeypatch.setattr("mp4_to_pdf_gui.Mp4ToPdfWorker", FakeWorker)
    main_window.file_path = "talk.mp4"
    main_window.cf_diff.set(diff)
    main_window.cf_smi.set(smi)
    main_window.convert()
    assert started["diff"] == pytest.approx(diff * 0.01)
    assert started["ssim"] == pytest.approx(smi * 0.01)


@pytest.mark.gui
def test_convert_disables_button_and_resets_bars(main_window, monkeypatch):
    class FakeWorker:
        def __init__(self, *args, **kwargs):
            pass

        def start(self):
            pass

    monkeypatch.setattr("mp4_to_pdf_gui.Mp4ToPdfWorker", FakeWorker)
    main_window.file_path = "a.mp4"
    main_window.bar1["value"] = 10
    main_window.bar2["value"] = 20
    main_window.bar3["value"] = 30
    main_window.convert()
    assert main_window.bar1["value"] == 0
    assert main_window.bar2["value"] == 0
    assert main_window.bar3["value"] == 0
    assert str(main_window.btn_convert["state"]) == "disabled"


@pytest.mark.gui
def test_convert_passes_outfile_and_queue(main_window, monkeypatch):
    started = {}

    class FakeWorker:
        def __init__(self, q, infile, out, nframe, diff_t, ssim_t):
            started["queue"] = q
            started["infile"] = infile
            started["out"] = out

        def start(self):
            pass

    monkeypatch.setattr("mp4_to_pdf_gui.Mp4ToPdfWorker", FakeWorker)
    main_window.file_path = "C:/x.mp4"
    main_window.convert()
    assert started["infile"] == "C:/x.mp4"
    assert started["out"] == "C:/x.pdf"
    assert started["queue"] is main_window.queue


@pytest.mark.gui
def test_select_mp4_with_path(main_window, monkeypatch):
    monkeypatch.setattr("mp4_to_pdf_gui.filedialog.askopenfilename", lambda **kwargs: "C:/demo.mp4")
    main_window.select_mp4()
    assert main_window.file_path == "C:/demo.mp4"
    assert main_window.lbl_selected["text"] == "Selected: 'C:/demo.mp4'"


@pytest.mark.gui
def test_select_mp4_cancelled(main_window, monkeypatch):
    main_window.file_path = "old.mp4"
    monkeypatch.setattr("mp4_to_pdf_gui.filedialog.askopenfilename", lambda **kwargs: "")
    main_window.select_mp4()
    assert main_window.file_path == ""
    assert main_window.lbl_selected["text"] == "Selected: None"


@pytest.mark.gui
def test_select_mp4_filters_mp4(main_window, monkeypatch):
    recorded = {}

    def fake_dialog(**kwargs):
        recorded.update(kwargs)
        return "a.mp4"

    monkeypatch.setattr("mp4_to_pdf_gui.filedialog.askopenfilename", fake_dialog)
    main_window.select_mp4()
    assert recorded["filetypes"] == [("MP4", ".mp4")]


@pytest.mark.gui
def test_update_ui_progress_codes(main_window):
    main_window.update_ui(Mp4ToPdfWorker.UPDATE_READING, 111)
    main_window.update_ui(Mp4ToPdfWorker.UPDATE_DIFF, 222)
    main_window.update_ui(Mp4ToPdfWorker.UPDATE_SMI, 333)
    assert main_window.bar1["value"] == 111
    assert main_window.bar2["value"] == 222
    assert main_window.bar3["value"] == 333


@pytest.mark.gui
def test_update_ui_done_reenables_button(main_window, monkeypatch):
    info = MagicMock()
    monkeypatch.setattr("tkinter.messagebox.showinfo", info)
    main_window.file_path = "out.mp4"
    main_window.btn_convert["state"] = "disabled"
    main_window.update_ui(Mp4ToPdfWorker.DONE, 0)
    assert str(main_window.btn_convert["state"]) == "normal"
    info.assert_called_once()
    assert "out.pdf" in info.call_args[0][1]


@pytest.mark.gui
def test_update_ui_unknown_code_is_ignored(main_window):
    main_window.update_ui(99, 5)
    assert main_window.bar1["value"] == 0
    assert main_window.bar2["value"] == 0
    assert main_window.bar3["value"] == 0


@pytest.mark.gui
def test_refresh_drains_queue(main_window):
    main_window.queue.put((Mp4ToPdfWorker.UPDATE_READING, 50))
    main_window.queue.put((Mp4ToPdfWorker.UPDATE_DIFF, 70))
    main_window.refresh()
    assert main_window.bar1["value"] == 50
    assert main_window.bar2["value"] == 70
    assert main_window.queue.empty()


@pytest.mark.gui
def test_refresh_reschedules_itself(main_window, monkeypatch):
    scheduled = {}

    def fake_after(ms, callback):
        scheduled["ms"] = ms
        scheduled["callback"] = callback
        return "id"

    monkeypatch.setattr(main_window.root, "after", fake_after)
    main_window.refresh()
    assert scheduled["ms"] == 200
    assert scheduled["callback"] == main_window.refresh


def test_center_geometry():
    class FakeWin:
        def winfo_screenwidth(self):
            return 1920

        def winfo_screenheight(self):
            return 1080

        def geometry(self, value):
            self.value = value

    win = FakeWin()
    center(win, 500, 400)
    assert win.value == "500x400+710+340"


@pytest.mark.parametrize("sw,sh,w,h,expected", [
    (800, 600, 200, 100, "200x100+300+250"),
    (1000, 1000, 1000, 1000, "1000x1000+0+0"),
    (100, 100, 200, 200, "200x200+-50+-50"),
])
def test_center_various_screens(sw, sh, w, h, expected):
    class FakeWin:
        def winfo_screenwidth(self):
            return sw

        def winfo_screenheight(self):
            return sh

        def geometry(self, value):
            self.value = value

    win = FakeWin()
    center(win, w, h)
    assert win.value == expected
