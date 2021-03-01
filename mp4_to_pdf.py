import cv2
import numpy as np
import datetime
from skimage.metrics import structural_similarity
from PIL import Image


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


    # Print iterations progress
    # from https://stackoverflow.com/questions/3173320/text-progress-bar-in-the-console
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """
    def progress_bar(self, iteration, total, prefix='Progress:', suffix='Complete', decimals=1, length=50, fill='█', print_end="\r"):
        if self.verbose:
            percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
            filled_length = int(length * iteration // total)
            bar = fill * filled_length + '-' * (length - filled_length)
            print(f'\r{prefix} |{bar}| {percent}% {suffix}', end=print_end)
            if iteration == total: # Print New Line on Complete
                print()


    def log_video_info(self, length, fps):
        self.log(f"File {self.infile}:")
        self.log(f"\tFPS: {fps}.")
        self.log(f"\tLenght: {length} frames.")
        self.log(f"\tDuration: {datetime.timedelta(seconds=length / fps)}.")


    def get_images(self):
        video = cv2.VideoCapture(self.infile)
        count = 0
        images = []

        length = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = video.get(cv2.CAP_PROP_FPS)

        self.log_video_info(length, fps)
        self.progress_bar(0, length)

        while video.isOpened():
            success, image = video.read()

            if success:
                images.append(image)
                count += self.n_frame
                video.set(1, count)

                self.progress_bar(count + 1, length)
            else:
                video.release()
                break

            if self.lim and count > self.lim:
                break

        self.progress_bar(length, length)
        return images


    def diff_filter(self, images):
        pairs = []

        self.progress_bar(0, len(images))
        for i in range(1, len(images)):
            img = images[i]
            prev_img = images[i - 1]

            diff = img - prev_img
            equal_pct = np.mean(np.abs(diff) < 0.01)  # 0.01 to prevent float pointing.

            if equal_pct < self.diff_threshold:
                pairs.append([img, prev_img])

            self.progress_bar(i + 1, len(images))

        return pairs


    def structural_similarity_filter(self, pairs):
        fails = []

        self.progress_bar(0, len(pairs))
        for i, p in enumerate(pairs):
            ssim = structural_similarity(p[0], p[1], multichannel=True)
            if ssim < self.ssim_threshold:
                fails.append(p)
            self.progress_bar(i + 1, len(pairs))

        return fails


    def save_as_pdf(self, images):
        as_images = [Image.fromarray(image) for image in images]
        as_images[0].save(self.out, "PDF", resolution=100.0, save_all=True, append_images=as_images[1:])


    def convert(self):
        self.log(f"Reading file {args.infile}...")
        images = self.get_images()
        self.log(f"Read {len(images)} images.")

        self.log("Calculating differences....")
        diff_pairs = self.diff_filter(images)
        self.log(f"Found {len(diff_pairs)} pairs with differences.")

        self.log("Applying structural similarity...")
        changes = self.structural_similarity_filter(diff_pairs)
        self.log(f"Found {len(changes)} uniques with SSIM.")

        uniques = [e[0] for e in changes]

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
