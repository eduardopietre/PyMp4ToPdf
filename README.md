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
Example:  
```
python3 mp4_to_pdf.py video.mp4 --out output.pdf --nframe 100 -v
```
Will convert the file named 'video.mp4' (in the same directory), read only every 100th frame (speeds up things) and save as 'output.pdf' (in the same directory).  
  
Verbose mode `-v` is highly recommended as it displays progress. 
  
Including `--lim 5000`, for example, would cause it to only read the first 5000 frames, instead of the whole file.   
  
If you would like to finetune and improve the result, you may change `--diff` and `--ssim` default arguments.  
Changing `--ssim` usually results in better results than changing `--diff`, but since `--diff` is faster and is used first, changing it may lead to better speed-ups. Be careful as `--diff` is highly sensitive.  
