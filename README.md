# PyMp4ToPdf
**A simple script to convert .mp4 to .pdf, util for extracting slides from a video file.**  
  
## Requisites ##
* Python 3.6 or latter
    * Package OpenCV (tested with [opencv-contrib-python](https://pypi.org/project/opencv-contrib-python/)).
    * Package [Numpy](https://pypi.org/project/numpy/).
    * Package [Scikit-image](https://scikit-image.org/).
    * Package [Pillow](https://pillow.readthedocs.io/en/stable/index.html).
  
## Usage ##
In the project folder run **`python3 mp4_to_pdf.py -h`** to see the following help message:  
```
usage: mp4_to_pdf.py [-h] [--out OUT] [--nframe NFRAME] [--lim LIM]
                     [--diff DIFF] [--ssim SSIM] [-v]
                     infile

positional arguments:
  infile           The .mp4 file path.

optional arguments:
  -h, --help       show this help message and exit
  --out OUT        The .pdf output file path. Defaults to the .mp4 file name
                   plus .pdf.
  --nframe NFRAME  Read every N'th frame. Defaults to 24.
  --lim LIM        Read only N frames.
  --diff DIFF      Min diff needed for checking. Defaults to 0.90 (0=Nothing
                   like, 1=Identical).
  --ssim SSIM      Structural similarity threshold. Defaults to 0.90
                   (0=Nothing like, 1=Identical).
  -v               Verbose mode.
```

### Known Issues ###
- [ ] Image colors are sometimes deviated to bluish.
