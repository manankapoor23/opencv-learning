# CV-Lab

Hands-on notebooks for the four Image Processing assignments in the parent folder, plus a
YOLO starter. Every task is a stub you fill in, followed by a list of properties you write
your own tests for. No helper module, no hidden graders: every line that runs is one you can read.

---

## 1. This machine

| | |
|---|---|
| Model | MacBook Air (Mac14,2), Apple **M2** |
| CPU / GPU | 8 cores (4P + 4E) / **8-core GPU**, Metal 4 |
| Memory | **8 GB unified** (shared between CPU and GPU — this is the real limit) |
| Disk free | ~29 GB |
| Accelerator | **MPS**, not CUDA. `torch.backends.mps.is_available() == True` |

### Measured, not guessed

| workload | result |
|---|---|
| resnet18 forward, 640×640 (≈ one YOLOv8n frame of work) | **19.7 ms** on MPS · 143.9 ms on CPU → **7.3× speedup, ~50 fps** |
| resnet18 training step, batch 8 @ 224 | 80 ms → **~64 images/sec** |
| 1080p Gaussian blur 15×15 (OpenCV, CPU) | 13.0 ms |
| 1080p Canny edges | **2.9 ms** |
| 1080p SIFT detect + describe | 264 ms |

### The honest capability threshold

**Comfortable — do all of this here:**
- Every classical OpenCV topic, at full resolution, interactively. Assignments 1–4 are
  microseconds of work; the entire Gonzalez book is within reach. Filtering, morphology,
  edges, Fourier, histogram processing, contours, template matching, camera calibration,
  stereo, optical flow, panorama stitching, SIFT/ORB matching.
- **Real-time inference** on small models: YOLOv8n/s, MobileNet, ResNet-18/50 classification,
  segmentation with yolov8n-seg, pose, face/landmark detection — all at video frame rates
  from the webcam.
- **Fine-tuning** small detectors: `yolov8n` at `imgsz=416`, `batch=4–8`, a few hundred to a
  few thousand images. Minutes-to-an-hour per run, not days.
- Fine-tuning a small classifier (ResNet-18 / MobileNet) on a few-thousand-image dataset.

**Works, but slowly — fine for learning, painful for iterating:**
- YOLOv8m/l inference (single images fine; not real time).
- Fine-tuning at `imgsz=640` with batch 8–16. Memory pressure, not errors, is what bites.
- ViT / DETR / SAM inference — a few seconds per image.
- Anything where you want to try 20 hyperparameter settings.

**Don't — use Colab's free T4 or a rented GPU instead:**
- Training any detector or backbone **from scratch** (ImageNet/COCO-scale, days of compute).
- Diffusion model *training*; Stable Diffusion inference runs but at ~30–60 s/image.
- Large video models, NeRF/Gaussian-splatting training, batch sizes above ~16 at 640px.
- Multi-GPU anything.

**The two real constraints:**
1. **8 GB is shared.** The GPU has no separate VRAM. A model, its activations, your dataset
   cache, Chrome, and the OS all compete. macOS swaps rather than erroring, so an over-large
   batch shows up as everything getting *slower*, not as a crash. Watch Activity Monitor →
   Memory Pressure. Close the browser before a training run and you get real speedup.
2. **MPS is not CUDA.** A few PyTorch ops are unimplemented and will raise. Set
   `PYTORCH_ENABLE_MPS_FALLBACK=1` to turn those into a silent CPU fallback (slow, but it runs).
   No `bitsandbytes`, no flash-attention, no `torch.compile` gains worth counting.

**Verdict:** more than enough for this course and for learning YOLO properly. You will be
limited by what you understand, not by the hardware, until you try to train something from
scratch — and you should not be doing that while learning anyway.

---

## 2. Setup

There is **one** env for this course and one kernel. Pick `Python 3.12 (cv)` — nothing else.

| | |
|---|---|
| env location | `~/cv` (a plain venv) |
| interpreter | `~/cv/bin/python` — Python 3.12.5 |
| kernel name | **`Python 3.12 (cv)`** |
| every notebook | already pinned to it |

```bash
~/cv/bin/python -m jupyter lab          # or just open the notebook in VS Code
```

Cell 1 of `00_setup_and_device_check.ipynb` must print `3.12.5` and `/Users/manankapoor/cv/bin/python`.
If it prints anything else, the wrong kernel is selected.

**What's inside:**

| package | version | why |
|---|---|---|
| `opencv-contrib-python` | **4.11.0.86** (pinned) | includes SIFT, aruco, ximgproc, dnn_superres |
| `numpy` | 2.5.2 | |
| `torch` / `torchvision` | 2.13.0 / 0.28.0 | MPS works |
| `ultralytics` | latest | YOLO — already installed, no `%pip` needed |
| `scikit-image`, `PyWavelets` | | reference implementations for LBP / GLCM / DWT |
| `scikit-learn`, `scipy`, `matplotlib`, `jupyterlab` | | |

**Three traps this env exists to avoid** — all three actually bit during setup:

1. The kernel named plain **"Python 3"** on this machine is `/opt/anaconda3/bin/python`, which has
   **no cv2 and no torch**. That is why notebooks failed to run before.
2. `pip install opencv-contrib-python` now resolves to **5.0.0**, a major version with breaking
   changes. It is pinned to 4.11 here because that is what every notebook is verified against.
3. Installing `ultralytics` silently pulled in **`opencv-python` 5.0** *alongside* the pinned
   `opencv-contrib-python` — two packages both providing `cv2`, last one wins. Resolved by
   uninstalling both and reinstalling only `opencv-contrib-python==4.11.0.86`. If `cv2.__version__`
   ever reports 5.x, this is what happened; run:
   ```bash
   ~/cv/bin/python -m pip uninstall -y opencv-python opencv-contrib-python
   ~/cv/bin/python -m pip install "opencv-contrib-python==4.11.0.86"
   ```

> **Why `~/cv` and not inside this folder:** venv scripts hardcode their own path in a `#!` shebang
> line, and this project folder is named `"Image-Processing "` **with a trailing space**. A space in
> a shebang breaks `pip` and the kernel launcher in confusing ways. Keeping the env outside the
> space-containing path sidesteps it entirely.

To rebuild the whole env from scratch:
```bash
rm -rf ~/cv && /Library/Frameworks/Python.framework/Versions/3.12/bin/python3 -m venv ~/cv
~/cv/bin/python -m pip install numpy matplotlib ipykernel jupyterlab scipy scikit-learn \
    scikit-image PyWavelets pillow tqdm torch torchvision ultralytics
~/cv/bin/python -m pip uninstall -y opencv-python opencv-contrib-python
~/cv/bin/python -m pip install "opencv-contrib-python==4.11.0.86"
~/cv/bin/python -m ipykernel install --user --name cv --display-name "Python 3.12 (cv)"
```

---

## 3. Contents

```
CV-Lab/
├── images/       10 standard OpenCV sample images
├── 00_setup_and_device_check.ipynb
├── 01_binary_thresholding.ipynb     <- Assignment 1
├── 02_rgb_to_grayscale.ipynb        <- Assignment 2
├── 03_border_and_complement.ipynb   <- Assignment 3
├── 04_intensity_transforms.ipynb    <- Assignment 4
├── 05_yolo_first_detection.ipynb    <- bonus: detection, IoU, NMS
├── 06_image_fundamentals.ipynb
├── 07_color_models.ipynb
├── 08_spatial_filtering.ipynb
├── 09_histogram_processing.ipynb
└── webcam_yolo.py                   <- real-time YOLO from the webcam
```

| notebook | assignment text | core idea |
|---|---|---|
| 01 | grayscale → binary, mean threshold and user threshold | boolean masks; `uint8` overflow |
| 02 | RGB → grayscale, mean and user weights | the channel axis; input validation |
| 03 | border/padding with user width+colour; complement | slice assignment; why padding exists |
| 04 | log, gamma, intensity level slicing | transfer curves; lookup tables |
| 05 | — | boxes, IoU, non-maximum suppression |

---

## 4. How to use these

Each assignment notebook goes: concept → a **worked** cell that already runs → your stub →
**your tests** → **the real OpenCV call** → visualisation → questions → stretch goals.

### Naming convention: strip `my_` and you have the OpenCV name

Every stub you implement is named after the OpenCV function that does the same job:

| you write | the real call you must be able to write by hand |
|---|---|
| `my_threshold`, `my_threshold_mean` | `cv2.threshold` |
| `my_cvtColor_mean`, `my_cvtColor_weighted`, `my_cvtColor_HSI`, `my_cvtColor_YCrCb` | `cv2.cvtColor`, `cv2.transform` |
| `my_copyMakeBorder` | `cv2.copyMakeBorder` |
| `my_bitwise_not` | `cv2.bitwise_not` |
| `my_log_transform`, `my_gamma_LUT`, `my_contrast_stretch`, `my_quantize` | `cv2.LUT`, `cv2.normalize` |
| `my_resize_nearest` | `cv2.resize` |
| `my_connectedComponents` | `cv2.connectedComponents(WithStats)` |
| `my_distance` | `cv2.distanceTransform` |
| `my_neighbours` | `cv2.getStructuringElement` |
| `my_applyColorMap` | `cv2.applyColorMap` |
| `my_filter2D` | `cv2.filter2D` |
| `my_getGaussianKernel` | `cv2.getGaussianKernel` |
| `my_medianBlur` | `cv2.medianBlur` |
| `my_unsharp_mask` | `cv2.addWeighted` + `cv2.GaussianBlur` |
| `my_calcHist` | `cv2.calcHist` |
| `my_equalizeHist` | `cv2.equalizeHist` |
| `my_NMSBoxes`, `my_iou` | `cv2.dnn.NMSBoxes`, `torchvision.ops.box_iou` |

Right after your tests there is an **"the real OpenCV call"** cell that runs the genuine
function and `assert`s it agrees with yours — so you see the exact signature, the argument
order, and the traps (`cv2.threshold` returns a *tuple*; `dsize` is `(width, height)`;
`filter2D` does correlation not convolution; HSV hue is 0–179). Every notebook ends with an
**OpenCV API card**: the signatures worth memorising for a written exam.

1. **Run the given cells.** They are working reference code for the plumbing (loading,
   plotting, dtype demos). Read the output, predict it before you run where the text asks you to.
2. **Fill in the `# TODO` stub.** Each one raises `NotImplementedError` until you do. The
   docstring is the complete spec — shape, dtype, rounding, edge cases.
3. **Write the tests.** Under each stub is a list of properties the correct function must have.
   Turn each one into an `assert` with a message that says what broke. Build tiny inputs by hand
   where you know the exact answer, cover the edge cases, then compare against OpenCV on a real
   photo. When a test fails, working out *why* is the lesson.
4. **Only then open the "Nudge" toggle.** They describe the approach, not the code.
5. **Do the "Think about it" questions.** These are the viva questions. Write the answers in the
   notebook — they're what turns "I got it to pass" into "I understand it".

The properties to test are things like shapes, dtypes, endpoints, monotonicity, invariants
such as `complement(complement(x)) == x`, and agreement with the equivalent OpenCV call. If a
test fails, the bug can be in your function *or* in your test. Telling which is a real skill.

---

## 5. After assignment 4

The next topics in Gonzalez, in the order they build:

1. **Histogram equalisation** — the transfer curve *is* the cumulative histogram. Natural
   sequel to notebook 04; ~4 lines with `np.cumsum`.
2. **Spatial filtering / convolution** — mean, Gaussian, median. This is where the padding
   from notebook 03 stops being trivia.
3. **Edges** — Sobel gradients → Canny. Then you can read the `cv2.Canny` you already used.
4. **Morphology** — erode, dilate, open, close, on the binary images from notebook 01.
5. **Fourier domain** — `np.fft.fft2`, low/high-pass filters, and why convolution is
   multiplication over there.
6. **Features** → SIFT/ORB, matching, homography, stitching. The bridge into real CV.

Then notebook 05's "what to try next" table for the deep-learning side.

---

### One housekeeping note

The parent folder is named `"Image-Processing "` — **with a trailing space**. That will bite you
in shell commands and any script that builds paths by hand. Quote it always, or rename it:

```bash
cd /Users/manankapoor/Desktop/SEM5 && mv "Image-Processing " Image-Processing
```
