import datetime

from mp4_to_pdf_core import (
    collect_sampled_frames,
    diff_pairs,
    iter_sampled_frames,
    save_bgr_frames_as_pdf,
    ssim_filter_pairs,
    unique_frames_from_iterable,
)


class Mp4ToPdf:

    def __init__(self, infile, out, n_frame, lim, diff_threshold, ssim_threshold, verbose=False):
        self.infile = infile
        self.out = out
        self.n_frame = n_frame
        self.lim = lim
        self.diff_threshold = diff_threshold
        self.ssim_threshold = ssim_threshold
        self.verbose = verbose

    def log(self, text):
        if self.verbose:
            print(text)

    def progress_bar(self, iteration, total, prefix='Progress:', suffix='Complete', decimals=1, length=50, fill='█', print_end="\r"):
        if self.verbose:
            percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
            filled_length = int(length * iteration // total)
            bar = fill * filled_length + '-' * (length - filled_length)
            print(f'\r{prefix} |{bar}| {percent}% {suffix}', end=print_end)
            if iteration == total:
                print()

    def log_video_info(self, length, fps):
        self.log(f"File {self.infile}:")
        self.log(f"\tFPS: {fps}.")
        self.log(f"\tLenght: {length} frames.")
        self.log(f"\tDuration: {datetime.timedelta(seconds=length / fps)}.")

    def get_images(self):
        length_holder = {"length": 0}

        def on_open(length, fps):
            length_holder["length"] = length
            self.log_video_info(length, fps)
            self.progress_bar(0, length)

        def on_progress(count, length):
            self.progress_bar(count + 1, length)

        images = collect_sampled_frames(
            self.infile,
            self.n_frame,
            lim=self.lim,
            on_open=on_open,
            on_progress=on_progress,
        )
        self.progress_bar(length_holder["length"], length_holder["length"])
        return images

    def diff_filter(self, images):
        self.progress_bar(0, len(images))
        return diff_pairs(
            images,
            self.diff_threshold,
            on_progress=lambda current, total: self.progress_bar(current, total),
        )

    def structural_similarity_filter(self, pairs):
        self.progress_bar(0, len(pairs))
        return ssim_filter_pairs(
            pairs,
            self.ssim_threshold,
            on_progress=lambda current, total: self.progress_bar(current, total),
        )

    def save_as_pdf(self, images):
        save_bgr_frames_as_pdf(self.out, images)

    def convert(self):
        self.log(f"Reading file {self.infile}...")
        length_holder = {"length": 0}

        def on_open(length, fps):
            length_holder["length"] = length
            self.log_video_info(length, fps)
            self.progress_bar(0, length)

        def on_progress(count, length):
            self.progress_bar(count + 1, length)

        uniques, image_count, pair_count = unique_frames_from_iterable(
            iter_sampled_frames(
                self.infile,
                self.n_frame,
                lim=self.lim,
                on_open=on_open,
                on_progress=on_progress,
            ),
            self.diff_threshold,
            self.ssim_threshold,
        )
        self.progress_bar(length_holder["length"], length_holder["length"])
        self.log(f"Read {image_count} images.")
        self.log("Calculating differences....")
        self.log(f"Found {pair_count} pairs with differences.")
        self.log("Applying structural similarity...")
        self.log(f"Found {len(uniques)} uniques with SSIM.")
        self.log("Exporting...")
        self.save_as_pdf(uniques)
        self.log("Done.")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("infile", help="The .mp4 file path.")
    parser.add_argument("--out", help="The .pdf output file path. Defaults to the .mp4 file name plus .pdf.")
    parser.add_argument("--nframe", type=int, help="Read every N'th frame. Defaults to 24.", default=24)
    parser.add_argument("--lim", type=int, help="Read only N frames.", default=None)
    parser.add_argument("--diff", type=float, help="Min diff needed for checking. Defaults to 0.90 (0=Nothing like, 1=Identical).", default=0.90)
    parser.add_argument("--ssim", type=float, help="Structural similarity threshold. Defaults to 0.90 (0=Nothing like, 1=Identical).", default=0.90)
    parser.add_argument("-v", help="Verbose mode.", action="store_true")

    args = parser.parse_args()
    out_file = args.out if args.out else f"{args.infile.replace('.mp4', '')}.pdf"

    if not args.infile.endswith(".mp4"):
        raise Exception("Infile must be a mp4.")

    converter = Mp4ToPdf(args.infile, out_file, args.nframe, args.lim, args.diff, args.ssim, verbose=args.v)
    converter.convert()
