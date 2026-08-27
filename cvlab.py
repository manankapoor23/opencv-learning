"""Shared helpers + self-grading checks for the CV lab notebooks.

Usage inside a notebook:   from cvlab import *

The `check_*` functions test YOUR function against properties the correct
answer must have. They never contain the answer, so reading this file will
not solve the assignment for you -- but it will tell you exactly which
property you broke.
"""

from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np

IMAGES = Path(__file__).parent / "images"


# ---------------------------------------------------------------- loading / display

def load(name, gray=False):
    """GRADER INFRASTRUCTURE -- do not use this in your own cells.

    It hides the two things you must be able to write yourself:
        cv2.imread(path, flags)                    and
        cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    The notebooks call those directly on purpose. This exists only so the check_*
    functions can fetch a test image in one line. See notebook 01 for the real thing.

    Returns RGB (H,W,3) uint8, or (H,W) if gray=True.
    """
    path = IMAGES / name
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE if gray else cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"{path} not found. Available: {sorted(p.name for p in IMAGES.iterdir())}")
    return img if gray else cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def show(*imgs, titles=None, size=4, cols=None):
    """show(a, b, c, titles=["a","b","c"]) -- plots 2D as grayscale 0-255, 3D as RGB."""
    if len(imgs) == 1 and isinstance(imgs[0], (list, tuple)):
        imgs = tuple(imgs[0])
    cols = cols or len(imgs)
    rows = -(-len(imgs) // cols)
    fig, axes = plt.subplots(rows, cols, figsize=(size * cols, size * rows), squeeze=False)
    for ax, img, i in zip(axes.ravel(), imgs, range(len(imgs))):
        if img.ndim == 2:
            ax.imshow(img, cmap="gray", vmin=0, vmax=255)
        else:
            ax.imshow(img)
        ax.set_title(titles[i] if titles else "", fontsize=10)
        ax.axis("off")
    for ax in axes.ravel()[len(imgs):]:
        ax.axis("off")
    plt.tight_layout()
    plt.show()


def hist(img, title="histogram"):
    """Plot the intensity histogram of a grayscale image."""
    plt.figure(figsize=(6, 3))
    plt.hist(img.ravel(), bins=256, range=(0, 256), color="steelblue")
    plt.title(title)
    plt.xlabel("intensity")
    plt.ylabel("pixel count")
    plt.xlim(0, 255)
    plt.tight_layout()
    plt.show()


def info(img, name="img"):
    """Print the four things you should always know about an array."""
    print(f"{name}: shape={img.shape}  dtype={img.dtype}  min={img.min()}  max={img.max()}")


# ---------------------------------------------------------------- test fixtures

def ramp(h=8):
    """(h,256) uint8 image whose every row is 0,1,2,...,255. Mean intensity = 127.5."""
    return np.tile(np.arange(256, dtype=np.uint8), (h, 1))


def patches(colors, s=4):
    """Stack solid colour squares into one (s, s*n, 3) uint8 image."""
    return np.hstack([np.full((s, s, 3), c, dtype=np.uint8) for c in colors])


# ---------------------------------------------------------------- checker

class _Check:
    def __init__(self, name):
        self.name = name
        self.fails = []

    def _log(self, ok, label, detail=""):
        print(f"  {'PASS' if ok else 'FAIL'}  {label}{'' if ok else '   <- ' + detail}")
        if not ok:
            self.fails.append(label)

    def true(self, label, cond, hint=""):
        self._log(bool(cond), label, hint)
        return self

    def eq(self, label, got, want, tol=0):
        got_a, want_a = np.asarray(got), np.asarray(want)
        # float64 throughout so the same method grades images, scalars and float arrays
        ok = got_a.shape == want_a.shape and np.all(
            np.abs(got_a.astype(np.float64) - want_a.astype(np.float64)) <= tol
        )
        if got_a.size <= 8:
            detail = f"got {got_a.tolist()}, expected {want_a.tolist()}"
        else:
            bad = int(np.sum(np.abs(got_a.astype(np.float64) - want_a.astype(np.float64)) > tol)) \
                if got_a.shape == want_a.shape else -1
            detail = (f"shape {got_a.shape} vs expected {want_a.shape}" if bad < 0
                      else f"{bad}/{got_a.size} pixels differ by more than {tol}")
        self._log(ok, label, detail)
        return self

    def sets(self, label, got, want):
        """Compare as unordered collections -- ordering of returned coordinates is free."""
        g, w = sorted(map(tuple, got)), sorted(map(tuple, want))
        self._log(g == w, label, f"got {g}, expected {w}")
        return self

    def raises(self, label, fn, *a, **kw):
        try:
            fn(*a, **kw)
        except Exception:
            self._log(True, label)
        else:
            self._log(False, label, "no exception raised -- invalid input was accepted")
        return self

    def done(self):
        if self.fails:
            raise AssertionError(f"{self.name}: {len(self.fails)} check(s) failed -> {self.fails}")
        print(f"\nAll checks passed for {self.name}. Nice.")


def _basic(c, out, ref, label="output"):
    """Every image-returning task must satisfy these."""
    c.true(f"{label} is uint8", out.dtype == np.uint8, f"got dtype {out.dtype} -- cast with .astype(np.uint8)")
    c.true(f"{label} shape == input shape", out.shape == ref.shape, f"got {out.shape}, expected {ref.shape}")


# ---------------------------------------------------------------- A1: binary

def check_binary_fixed(fn):
    """fn(gray, t) -> binary image, white where intensity > t."""
    c = _Check("binary_fixed")
    r = ramp()
    out = np.asarray(fn(r, 100))
    _basic(c, out, r)
    c.true("only 0 and 255 appear", set(np.unique(out)) <= {0, 255},
           f"found values {np.unique(out)[:10]} -- output must be strictly two-valued")
    c.true("pixels above t are 255", np.all(out[r > 100] == 255), "some bright pixels came out black")
    c.true("pixels at or below t are 0", np.all(out[r <= 100] == 0), "some dark pixels came out white")
    c.eq("t=0 keeps only intensity 0 black", int(np.sum(np.asarray(fn(r, 0)) == 0)), r.shape[0])
    c.true("t=255 gives an all-black image", np.all(np.asarray(fn(r, 255)) == 0))
    g = load("lena.jpg", gray=True)
    c.eq("matches cv2.threshold on lena at t=90",
         fn(g, 90), cv2.threshold(g, 90, 255, cv2.THRESH_BINARY)[1])
    return c.done()


def check_binary_mean(fn):
    """fn(gray) -> binary image thresholded at the image's own mean intensity."""
    c = _Check("binary_mean")
    r = ramp()
    out = np.asarray(fn(r))
    _basic(c, out, r)
    c.true("only 0 and 255 appear", set(np.unique(out)) <= {0, 255})
    # ramp mean is 127.5, so 127 must be black and 128 white
    c.eq("intensity 127 -> black (ramp mean is 127.5)", out[:, 127], np.zeros(r.shape[0], np.uint8))
    c.eq("intensity 128 -> white", out[:, 128], np.full(r.shape[0], 255, np.uint8))
    dark = np.full((4, 4), 10, np.uint8)
    dark[0, 0] = 200  # mean = 21.875
    o2 = np.asarray(fn(dark))
    c.true("threshold follows the image, not a hardcoded 127", o2[0, 0] == 255 and o2[1, 1] == 0,
           "a dark image with one bright pixel must still split -- did you hardcode the threshold?")
    for name in ("lena.jpg", "baboon.jpg", "sudoku.png"):
        g = load(name, gray=True)
        c.eq(f"matches cv2.threshold at mean on {name}",
             fn(g), cv2.threshold(g, g.mean(), 255, cv2.THRESH_BINARY)[1])
    return c.done()


# ---------------------------------------------------------------- A2: grayscale

def check_gray_mean(fn):
    """fn(rgb) -> grayscale, plain mean of the three planes."""
    c = _Check("gray_mean")
    p = patches([(0, 0, 0), (255, 255, 255), (255, 0, 0), (30, 60, 90), (10, 10, 11)])
    out = np.asarray(fn(p))
    c.true("output is 2D (H,W)", out.ndim == 2, f"got shape {out.shape} -- the channel axis must be gone")
    c.true("output is uint8", out.dtype == np.uint8, f"got {out.dtype}")
    c.true("output shape matches H,W", out.shape == p.shape[:2], f"got {out.shape}, expected {p.shape[:2]}")
    got = [int(out[2, 4 * i + 2]) for i in range(5)]
    c.eq("black->0, white->255, pure red->85, (30,60,90)->60, (10,10,11)->10",
         got, [0, 255, 85, 60, 10], tol=1)
    rgb = load("fruits.jpg")
    o = np.asarray(fn(rgb))
    c.true("no overflow on a real photo (mean is never above any channel max)",
           o.max() <= int(rgb.max()),
           "output brighter than the brightest channel -> uint8 overflowed, average in float")
    c.eq("independent of channel order (mean is symmetric)", fn(rgb), fn(rgb[..., ::-1]), tol=1)
    return c.done()


def check_gray_weighted(fn):
    """fn(rgb, wr, wg, wb) -> weighted grayscale; must reject invalid weights."""
    c = _Check("gray_weighted")
    rgb = load("fruits.jpg")
    c.eq("weights (1,0,0) return the R plane exactly", fn(rgb, 1.0, 0.0, 0.0), rgb[..., 0], tol=1)
    c.eq("weights (0,1,0) return the G plane exactly", fn(rgb, 0.0, 1.0, 0.0), rgb[..., 1], tol=1)
    c.eq("weights (0,0,1) return the B plane exactly", fn(rgb, 0.0, 0.0, 1.0), rgb[..., 2], tol=1)
    third = 1.0 / 3
    c.eq("equal weights == the plain mean", fn(rgb, third, third, third),
         rgb.mean(axis=2).round().astype(np.uint8), tol=1)
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    c.eq("luma weights (.299,.587,.114) match cv2.cvtColor",
         fn(rgb, 0.299, 0.587, 0.114), cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY), tol=2)
    c.raises("rejects weights that do not sum to 1", fn, rgb, 0.5, 0.5, 0.5)
    c.raises("rejects a negative weight", fn, rgb, 1.5, -0.5, 0.0)
    c.raises("rejects a weight above 1", fn, rgb, 1.2, 0.0, -0.2)
    return c.done()


# ---------------------------------------------------------------- A3: border + complement

def check_border(fn):
    """fn(img, width, color) -> img with a solid border of `width` px on all four sides."""
    c = _Check("add_border")
    g = np.full((10, 12), 50, np.uint8)
    out = np.asarray(fn(g, 3, 200))
    c.true("grayscale: uint8 preserved", out.dtype == np.uint8)
    c.eq("grayscale: shape grows by 2*width on both axes", out.shape, (16, 18))
    c.eq("grayscale: original pixels sit untouched in the middle", out[3:13, 3:15], g)
    c.true("grayscale: all four edges are the border colour",
           np.all(out[0] == 200) and np.all(out[-1] == 200)
           and np.all(out[:, 0] == 200) and np.all(out[:, -1] == 200),
           "at least one side is missing or wrong")
    c.eq("width=0 returns the image unchanged", fn(g, 0, 200), g)

    rgb = load("lena.jpg")
    o = np.asarray(fn(rgb, 5, (255, 0, 0)))
    c.eq("colour: shape grows on H and W only, 3 channels kept",
         o.shape, (rgb.shape[0] + 10, rgb.shape[1] + 10, 3))
    c.eq("colour: border pixel is the requested colour", o[0, 0], (255, 0, 0))
    c.eq("colour: image content preserved", o[5:-5, 5:-5], rgb)
    c.eq("matches cv2.copyMakeBorder", fn(rgb, 7, (0, 255, 0)),
         cv2.copyMakeBorder(rgb, 7, 7, 7, 7, cv2.BORDER_CONSTANT, value=(0, 255, 0)))

    b = (load("sudoku.png", gray=True) > 128).astype(np.uint8) * 255
    ob = np.asarray(fn(b, 4, 255))
    c.true("binary image stays two-valued after bordering", set(np.unique(ob)) <= {0, 255})
    return c.done()


def check_complement(fn):
    """fn(img) -> photographic negative."""
    c = _Check("complement")
    r = ramp()
    out = np.asarray(fn(r))
    _basic(c, out, r)
    c.eq("0 -> 255", out[:, 0], np.full(r.shape[0], 255, np.uint8))
    c.eq("255 -> 0", out[:, 255], np.zeros(r.shape[0], np.uint8))
    c.eq("127 -> 128", out[:, 127], np.full(r.shape[0], 128, np.uint8))
    c.eq("applying it twice returns the original", fn(out), r)
    c.true("output is decreasing where input increases", np.all(np.diff(out[0].astype(int)) <= 0),
           "the negative of a ramp must fall, not rise")
    rgb = load("lena.jpg")
    c.eq("works on colour images channel-wise", fn(rgb), 255 - rgb)
    b = np.array([[0, 255], [255, 0]], np.uint8)
    c.eq("binary: black and white swap", fn(b), np.array([[255, 0], [0, 255]], np.uint8))
    return c.done()


# ---------------------------------------------------------------- A4: intensity transforms

def check_log(fn):
    """fn(gray) -> log transform s = c*log(1+r), scaled so the output fills 0..255."""
    c = _Check("log_transform")
    r = ramp()
    out = np.asarray(fn(r))
    _basic(c, out, r)
    row = out[0].astype(int)
    c.eq("intensity 0 stays 0", row[0], 0)
    c.eq("intensity 255 maps to 255 (output is scaled to fill the range)", row[255], 255, tol=1)
    c.true("mapping never decreases", np.all(np.diff(row) >= 0),
           "log is monotonic -- a brighter input can never map to a darker output")
    c.true("dark tones are lifted (that is the point of a log transform)", row[64] > 64 + 20,
           f"input 64 mapped to {row[64]}; expected it noticeably brighter")
    c.true("curve is concave (gain shrinks as intensity rises)",
           np.all(np.diff(row, 2) <= 1),  # +1 slack: uint8 rounding jitters the 2nd difference
           "the curve bends the wrong way -- that is not a log shape")
    dark = load("board.jpg", gray=True)
    o = np.asarray(fn(dark))
    c.true("a real image gets brighter on average", o.mean() > dark.mean(),
           f"mean went {dark.mean():.1f} -> {o.mean():.1f}")
    return c.done()


def check_gamma(fn):
    """fn(gray, gamma) -> power-law transform s = c*r**gamma, c=1, output filling 0..255."""
    c = _Check("gamma_transform")
    r = ramp()
    out = np.asarray(fn(r, 1.0))
    _basic(c, out, r)
    c.eq("gamma=1 is the identity", out, r, tol=1)
    for gm in (0.4, 0.7, 1.0, 1.8, 2.5):
        row = np.asarray(fn(r, gm))[0].astype(int)
        c.true(f"gamma={gm}: endpoints pinned at 0 and 255",
               row[0] == 0 and abs(row[255] - 255) <= 1, f"got {row[0]} and {row[255]}")
        c.true(f"gamma={gm}: mapping never decreases", np.all(np.diff(row) >= 0))
    c.true("gamma<1 brightens the midtones", np.asarray(fn(r, 0.4))[0, 128] > 128 + 20)
    c.true("gamma>1 darkens the midtones", np.asarray(fn(r, 2.5))[0, 128] < 128 - 20)
    c.true("larger gamma is always darker at intensity 128",
           np.asarray(fn(r, 0.5))[0, 128] > np.asarray(fn(r, 1.5))[0, 128])
    g = load("lena.jpg", gray=True)
    c.eq("matches an LUT-based gamma on lena", fn(g, 2.2),
         cv2.LUT(g, ((np.arange(256) / 255.0) ** 2.2 * 255).round().astype(np.uint8)), tol=2)
    return c.done()


def check_stretch(fn):
    """fn(gray, in_lo, in_hi, out_lo, out_hi) -> 3-segment piecewise-linear stretch."""
    c = _Check("contrast_stretch")
    r = ramp()
    out = np.asarray(fn(r, 80, 120, 50, 150))
    _basic(c, out, r)
    row = out[0].astype(int)
    c.eq("in_lo (80) maps to out_lo (50)", row[80], 50, tol=1)
    c.eq("in_hi (120) maps to out_hi (150)", row[120], 150, tol=1)
    c.eq("the segment midpoint 100 maps to 100", row[100], 100, tol=2)
    c.eq("0 stays 0", row[0], 0)
    c.eq("255 stays 255", row[255], 255, tol=1)
    c.true("mapping never decreases", np.all(np.diff(row) >= 0))
    c.true("the chosen band is stretched, i.e. steeper than 45 degrees",
           (row[120] - row[80]) / 40 > 1.0,
           "150-50 over 120-80 is a slope of 2.5 -- yours came out flat")
    c.true("below in_lo the slope is gentler than in the band",
           (row[79] - row[0]) / 79 < (row[120] - row[80]) / 40)
    ident = np.asarray(fn(r, 0, 255, 0, 255))[0].astype(int)
    c.eq("full range in -> full range out is the identity", ident, np.arange(256), tol=1)
    inv = np.asarray(fn(r, 50, 200, 200, 50))[0].astype(int)
    c.true("an inverted output band actually inverts that band", inv[60] > inv[190],
           "out_lo > out_hi should flip the band -- do not clamp or sort the arguments")
    return c.done()


# ---------------------------------------------------------------- A5: detection maths

def check_iou(fn):
    """fn(box_a, box_b) -> IoU of two boxes given as (x1, y1, x2, y2)."""
    c = _Check("iou")
    a = (0, 0, 10, 10)
    c.eq("a box with itself is 1.0", round(float(fn(a, a)), 6), 1.0, tol=1e-6)
    c.eq("far-apart boxes are 0.0", round(float(fn(a, (50, 50, 60, 60))), 6), 0.0, tol=1e-6)
    c.eq("edge-touching boxes are 0.0 (zero-area overlap)",
         round(float(fn(a, (10, 0, 20, 10))), 6), 0.0, tol=1e-6)
    c.eq("half-overlap corner case (0,0,10,10) vs (5,5,15,15) = 25/175",
         round(float(fn(a, (5, 5, 15, 15))), 4), 0.1429, tol=1e-3)
    c.eq("fully contained box = 25/100", round(float(fn(a, (0, 0, 5, 5))), 4), 0.25, tol=1e-3)
    c.eq("order does not matter", round(float(fn(a, (3, 3, 12, 12))), 6),
         round(float(fn((3, 3, 12, 12), a)), 6), tol=1e-6)
    c.true("result always in [0,1]",
           all(0.0 <= float(fn(a, b)) <= 1.0
               for b in [(0, 0, 1, 1), (-5, -5, 5, 5), (2, 2, 3, 3), (0, 0, 100, 100)]))
    c.eq("no negative overlap when boxes miss on one axis only",
         round(float(fn((0, 0, 10, 10), (0, 20, 10, 30))), 6), 0.0, tol=1e-6)
    return c.done()


def check_nms(fn):
    """fn(boxes, scores, iou_thr) -> list of kept indices, highest score first."""
    c = _Check("nms")
    boxes = [(0, 0, 10, 10), (1, 1, 11, 11), (50, 50, 60, 60)]
    scores = [0.9, 0.8, 0.7]
    c.eq("suppresses the overlapping duplicate, keeps the far box",
         sorted(fn(boxes, scores, 0.5)), [0, 2])
    c.eq("a high threshold keeps everything", sorted(fn(boxes, scores, 0.99)), [0, 1, 2])
    c.eq("highest-scoring box comes out first", list(fn(boxes, scores, 0.5))[0], 0)
    c.eq("winner is picked by score, not by input order",
         list(fn(boxes, [0.1, 0.95, 0.7], 0.5))[0], 1)
    c.eq("empty input gives an empty result", len(list(fn([], [], 0.5))), 0)
    c.eq("a single box survives", list(fn([(0, 0, 5, 5)], [0.4], 0.5)), [0])
    chain = [(0, 0, 10, 10), (8, 0, 18, 10), (16, 0, 26, 10)]
    c.eq("chained mild overlaps are all kept at thr=0.5", sorted(fn(chain, [0.9, 0.8, 0.7], 0.5)), [0, 1, 2])
    c.true("does not mutate the caller's lists",
           (lambda b, s: (fn(b, s, 0.5), b == boxes and s == scores)[1])(list(boxes), list(scores)))
    return c.done()


# ================================================================ 06: image fundamentals

def check_quantize(fn):
    """fn(gray, levels) -> uniformly requantised image spanning the full 0..255 range."""
    c = _Check("quantize")
    r = ramp()
    c.eq("levels=256 is the identity", fn(r, 256), r)
    o2 = np.asarray(fn(r, 2))
    _basic(c, o2, r)
    c.true("levels=2 uses exactly the values {0, 255}", set(np.unique(o2)) == {0, 255})
    c.eq("levels=2 splits at 128", o2[0, [0, 127, 128, 255]], [0, 0, 255, 255])
    o4 = np.asarray(fn(r, 4))
    c.true("levels=4 uses exactly {0, 85, 170, 255}", set(np.unique(o4)) == {0, 85, 170, 255})
    c.eq("levels=4 bin edges land at 64/128/192", o4[0, [63, 64, 127, 128, 191, 192]],
         [0, 85, 85, 170, 170, 255])
    o8 = np.asarray(fn(r, 8))
    c.true("levels=8 produces 8 distinct values", len(np.unique(o8)) == 8)
    c.true("mapping never decreases", np.all(np.diff(o8[0].astype(int)) >= 0))
    g = load("lena.jpg", gray=True)
    for L in (2, 4, 16, 64):
        c.true(f"lena at {L} levels has at most {L} distinct values",
               len(np.unique(np.asarray(fn(g, L)))) <= L)
    c.true("fewer levels means larger error vs the original",
           np.abs(np.asarray(fn(g, 4)).astype(int) - g.astype(int)).mean()
           > np.abs(np.asarray(fn(g, 64)).astype(int) - g.astype(int)).mean())
    return c.done()


def check_downsample(fn):
    """fn(gray, factor) -> naive subsampling, no pre-filtering."""
    c = _Check("downsample")
    g = np.arange(48, dtype=np.uint8).reshape(6, 8)
    c.eq("factor=1 returns the image unchanged", fn(g, 1), g)
    o = np.asarray(fn(g, 2))
    c.eq("factor=2 halves both axes", o.shape, (3, 4))
    c.eq("factor=2 keeps the pixels at even row/col indices", o, g[::2, ::2])
    c.eq("factor=3 rounds the shape up (6x8 -> 2x3)", np.asarray(fn(g, 3)).shape, (2, 3))
    big = load("lena.jpg", gray=True)
    c.true("output dtype is preserved", np.asarray(fn(big, 4)).dtype == np.uint8,
           f"got {np.asarray(fn(big, 4)).dtype}")
    c.true("it really is subsampling, not averaging",
           np.all(np.isin(np.unique(np.asarray(fn(big, 8))), np.unique(big))),
           "an averaged result invents values that were not in the original")
    return c.done()


def check_neighbours(fn):
    """fn(shape, p, kind) -> in-bounds neighbour coordinates; kind is 'N4', 'N8' or 'ND'."""
    c = _Check("neighbours")
    sh, p = (5, 5), (2, 2)
    c.sets("N4 of an interior pixel", fn(sh, p, "N4"), [(1, 2), (3, 2), (2, 1), (2, 3)])
    c.sets("ND (the diagonals only)", fn(sh, p, "ND"), [(1, 1), (1, 3), (3, 1), (3, 3)])
    c.sets("N8 is N4 + ND", fn(sh, p, "N8"),
           [(1, 1), (1, 2), (1, 3), (2, 1), (2, 3), (3, 1), (3, 2), (3, 3)])
    c.true("a pixel is never its own neighbour", p not in [tuple(x) for x in fn(sh, p, "N8")])
    c.sets("corner (0,0): N4 clipped to the image", fn(sh, (0, 0), "N4"), [(0, 1), (1, 0)])
    c.sets("corner (0,0): N8 clipped to the image", fn(sh, (0, 0), "N8"),
           [(0, 1), (1, 0), (1, 1)])
    c.sets("bottom-right corner clipped", fn(sh, (4, 4), "N8"), [(3, 3), (3, 4), (4, 3)])
    c.eq("edge pixel on a non-square image has 5 N8 neighbours",
         len(list(fn((3, 7), (0, 3), "N8"))), 5)
    c.eq("1x1 image: no neighbours at all", len(list(fn((1, 1), (0, 0), "N8"))), 0)
    return c.done()


def check_distance(fn):
    """fn(p, q, metric) -> distance; metric is 'D4' (city-block), 'D8' (chessboard) or 'euclidean'."""
    c = _Check("distance")
    p, q = (0, 0), (3, 4)
    c.eq("D4 (city-block) of (0,0)-(3,4) is 7", fn(p, q, "D4"), 7)
    c.eq("D8 (chessboard) of (0,0)-(3,4) is 4", fn(p, q, "D8"), 4)
    c.eq("euclidean of (0,0)-(3,4) is 5", fn(p, q, "euclidean"), 5, tol=1e-9)
    for m in ("D4", "D8", "euclidean"):
        c.eq(f"{m}: distance to itself is 0", fn(p, p, m), 0, tol=1e-9)
        c.eq(f"{m}: symmetric", fn(p, q, m), fn(q, p, m), tol=1e-9)
        c.true(f"{m}: never negative", fn((5, 2), (1, 9), m) >= 0)
    c.eq("D4 of a pure diagonal step is 2", fn((0, 0), (1, 1), "D4"), 2)
    c.eq("D8 of a pure diagonal step is 1", fn((0, 0), (1, 1), "D8"), 1)
    c.true("D8 <= euclidean <= D4 always (the standard sandwich)",
           all(fn((0, 0), z, "D8") <= fn((0, 0), z, "euclidean") + 1e-9
               <= fn((0, 0), z, "D4") + 1e-9
               for z in [(1, 1), (3, 4), (7, 2), (0, 5), (6, 6)]))
    c.eq("negative coordinates still work", fn((-2, -3), (1, 1), "D4"), 7)
    return c.done()


def check_components(fn):
    """fn(binary, connectivity) -> number of connected white regions; connectivity is 4 or 8."""
    c = _Check("count_components")
    z = np.zeros((5, 5), np.uint8)
    c.eq("an all-black image has 0 components", fn(z, 4), 0)
    c.eq("an all-white image has 1 component", fn(np.full((5, 5), 255, np.uint8), 8), 1)
    one = z.copy(); one[2, 2] = 255
    c.eq("a single pixel is 1 component", fn(one, 4), 1)
    # the classic diagonal: two 4-components but one 8-component
    diag = z.copy(); diag[1, 1] = 255; diag[2, 2] = 255
    c.eq("diagonal pair: 2 components under 4-connectivity", fn(diag, 4), 2)
    c.eq("diagonal pair: 1 component under 8-connectivity", fn(diag, 8), 1)
    checker = (np.indices((6, 6)).sum(axis=0) % 2 * 255).astype(np.uint8)
    c.eq("checkerboard: 18 components under 4-connectivity", fn(checker, 4), 18)
    c.eq("checkerboard: 1 component under 8-connectivity", fn(checker, 8), 1)
    ring = np.full((7, 7), 255, np.uint8); ring[2:5, 2:5] = 0
    c.eq("a ring is 1 component (the hole is not counted)", fn(ring, 4), 1)
    for name, conn in [("sudoku.png", 4), ("sudoku.png", 8), ("board.jpg", 8)]:
        b = ((load(name, gray=True) > 128) * 255).astype(np.uint8)
        want = cv2.connectedComponents(b, connectivity=conn)[0] - 1  # minus the background label
        c.eq(f"matches cv2.connectedComponents on {name} ({conn}-conn)", fn(b, conn), want)
    return c.done()


# ================================================================ 07: colour

def check_rgb_to_hsi(fn):
    """fn(rgb) -> (H, S, I) float arrays; H in [0,360), S in [0,1], I in [0,1]."""
    c = _Check("rgb_to_hsi")
    p = patches([(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 255),
                 (0, 0, 0), (128, 128, 128), (255, 255, 0)])
    H, S, I = (np.asarray(x) for x in fn(p))
    c.eq("all three outputs have the image's H,W shape", (H.shape, S.shape, I.shape),
         (p.shape[:2],) * 3)
    at = lambda a, i: float(a[2, 4 * i + 2])
    c.eq("hue of pure red is 0", at(H, 0), 0, tol=0.5)
    c.eq("hue of pure green is 120", at(H, 1), 120, tol=0.5)
    c.eq("hue of pure blue is 240", at(H, 2), 240, tol=0.5)
    c.eq("hue of yellow is 60", at(H, 6), 60, tol=0.5)
    c.eq("saturation of pure red is 1", at(S, 0), 1.0, tol=1e-3)
    c.eq("saturation of white is 0", at(S, 3), 0.0, tol=1e-3)
    c.eq("saturation of mid grey is 0", at(S, 5), 0.0, tol=1e-3)
    c.eq("intensity of white is 1", at(I, 3), 1.0, tol=1e-3)
    c.eq("intensity of black is 0", at(I, 4), 0.0, tol=1e-3)
    c.eq("intensity of pure red is 1/3", at(I, 0), 1 / 3, tol=1e-3)
    c.eq("intensity of mid grey is 128/255", at(I, 5), 128 / 255, tol=1e-3)
    rgb = load("fruits.jpg")
    H2, S2, I2 = (np.asarray(x) for x in fn(rgb))
    c.true("H stays inside [0,360)", H2.min() >= -1e-6 and H2.max() < 360 + 1e-6)
    c.true("S stays inside [0,1]", S2.min() >= -1e-6 and S2.max() <= 1 + 1e-6)
    c.eq("I is exactly the mean of the three planes, scaled to 0..1",
         I2, rgb.mean(axis=2) / 255.0, tol=1e-6)
    c.true("no NaN anywhere (watch the S=0 and black-pixel cases)",
           not (np.isnan(H2).any() or np.isnan(S2).any() or np.isnan(I2).any()),
           "a zero denominator leaked through -- guard it")
    return c.done()


def check_rgb_to_ycbcr(fn):
    """fn(rgb) -> (H,W,3) uint8 array of Y, Cb, Cr (JPEG full-range BT.601)."""
    c = _Check("rgb_to_ycbcr")
    p = patches([(0, 0, 0), (255, 255, 255), (255, 0, 0), (0, 255, 0), (0, 0, 255)])
    out = np.asarray(fn(p))
    c.true("output is (H,W,3) uint8", out.shape == p.shape and out.dtype == np.uint8,
           f"got {out.shape} {out.dtype}")
    at = lambda i: out[2, 4 * i + 2].astype(int).tolist()
    c.eq("black -> Y=0, Cb=128, Cr=128", at(0), [0, 128, 128], tol=1)
    c.eq("white -> Y=255, Cb=128, Cr=128", at(1), [255, 128, 128], tol=1)
    c.eq("pure red -> Y=76, Cb=85, Cr=255", at(2), [76, 85, 255], tol=2)
    c.eq("pure blue -> Y=29, Cb=255, Cr=107", at(4), [29, 255, 107], tol=2)
    rgb = load("fruits.jpg")
    o = np.asarray(fn(rgb))
    ref = cv2.cvtColor(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), cv2.COLOR_BGR2YCrCb)
    c.eq("Y channel matches cv2", o[..., 0], ref[..., 0], tol=2)
    c.eq("Cb channel matches cv2 (note cv2 orders it Y,Cr,Cb)", o[..., 1], ref[..., 2], tol=2)
    c.eq("Cr channel matches cv2", o[..., 2], ref[..., 1], tol=2)
    grey = np.dstack([np.full((4, 4), 90, np.uint8)] * 3)
    c.eq("a grey patch has zero chroma (both at 128)", np.asarray(fn(grey))[2, 2, 1:],
         [128, 128], tol=1)
    return c.done()


def check_pseudocolor(fn):
    """fn(gray, bounds, colors) -> RGB image; bounds ascending, len(colors) == len(bounds)+1."""
    c = _Check("pseudocolor_slice")
    r = ramp()
    cols = [(0, 0, 255), (0, 255, 0), (255, 0, 0)]
    out = np.asarray(fn(r, [85, 170], cols))
    c.true("output is (H,W,3) uint8", out.shape == r.shape + (3,) and out.dtype == np.uint8,
           f"got {out.shape} {out.dtype}")
    c.eq("intensity 0 gets colour 0", out[0, 0], cols[0])
    c.eq("intensity 85 still gets colour 0 (band is v <= bound)", out[0, 85], cols[0])
    c.eq("intensity 86 crosses into colour 1", out[0, 86], cols[1])
    c.eq("intensity 170 gets colour 1", out[0, 170], cols[1])
    c.eq("intensity 171 crosses into colour 2", out[0, 171], cols[2])
    c.eq("intensity 255 gets the last colour", out[0, 255], cols[2])
    c.true("exactly the given colours appear, nothing blended",
           {tuple(x) for x in out.reshape(-1, 3)} <= {tuple(x) for x in cols})
    single = np.asarray(fn(r, [], [(7, 8, 9)]))
    c.true("no bounds -> one flat colour everywhere", np.all(single == np.array([7, 8, 9])))
    g = load("board.jpg", gray=True)
    o = np.asarray(fn(g, [50, 100, 150, 200], [(0,0,0), (0,0,255), (0,255,0), (255,255,0), (255,0,0)]))
    c.eq("works on a real image at the right shape", o.shape, g.shape + (3,))
    return c.done()


# ================================================================ 08: spatial filtering

def check_convolve(fn):
    """fn(gray, kernel) -> float64 same-size TRUE convolution, zero padding."""
    c = _Check("convolve2d")
    g = load("lena.jpg", gray=True)[:120, :140]
    gf = g.astype(np.float64)

    def ref(src, k):
        return cv2.filter2D(src.astype(np.float64), -1, np.flip(np.asarray(k, float)),
                            borderType=cv2.BORDER_CONSTANT)

    ident = np.zeros((3, 3)); ident[1, 1] = 1
    out = np.asarray(fn(g, ident))
    c.eq("output keeps the input shape", out.shape, g.shape)
    c.true("output is floating point, not clipped to uint8", np.issubdtype(out.dtype, np.floating),
           f"got {out.dtype} -- return float so the caller decides on clipping")
    c.eq("a delta kernel is the identity", out, gf, tol=1e-6)
    box = np.ones((3, 3)) / 9
    c.eq("3x3 box blur matches the reference", fn(g, box), ref(g, box), tol=1e-6)
    c.eq("5x5 box blur matches the reference", fn(g, np.ones((5, 5)) / 25),
         ref(g, np.ones((5, 5)) / 25), tol=1e-6)

    # the flip is the whole difference between convolution and correlation
    asym = np.array([[0, 0, 0], [1, 0, 0], [0, 0, 0]], float)   # picks the pixel to the LEFT
    oc = np.asarray(fn(g, asym))
    c.eq("an asymmetric kernel is CONVOLVED, not correlated", oc, ref(g, asym), tol=1e-6)
    c.true("...and that is genuinely different from correlation",
           not np.allclose(oc, cv2.filter2D(gf, -1, asym, borderType=cv2.BORDER_CONSTANT)),
           "your result equals correlation -- you forgot to flip the kernel")

    sob = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], float)
    c.eq("a Sobel kernel matches the reference", fn(g, sob), ref(g, sob), tol=1e-6)
    c.true("negative outputs are preserved", np.asarray(fn(g, sob)).min() < 0,
           "a gradient kernel must produce negative values -- do not clip inside convolve")
    flat = np.full((20, 20), 100, np.uint8)
    c.eq("interior of a flat region survives a normalised box blur",
         np.asarray(fn(flat, box))[5:15, 5:15], np.full((10, 10), 100.0), tol=1e-6)
    c.true("zero padding darkens the border of a flat region",
           np.asarray(fn(flat, box))[0, 0] < 100 - 1,
           "with zero padding the corner should average in the zeros outside")
    c.eq("non-square kernels work", np.asarray(fn(g, np.ones((1, 5)) / 5)).shape, g.shape)
    return c.done()


def check_gaussian_kernel(fn):
    """fn(k, sigma) -> (k,k) float kernel summing to 1."""
    c = _Check("gaussian_kernel")
    for k, s in [(3, 1.0), (5, 1.0), (7, 2.5), (9, 0.8)]:
        K = np.asarray(fn(k, s))
        c.eq(f"k={k}: shape is ({k},{k})", K.shape, (k, k))
        c.eq(f"k={k}, sigma={s}: sums to 1", K.sum(), 1.0, tol=1e-9)
        c.true(f"k={k}: peak is at the centre", K[k // 2, k // 2] == K.max())
        c.true(f"k={k}: all weights positive", K.min() > 0)
        c.eq(f"k={k}: symmetric left-right", K, np.fliplr(K), tol=1e-12)
        c.eq(f"k={k}: symmetric up-down", K, np.flipud(K), tol=1e-12)
        ref = cv2.getGaussianKernel(k, s) @ cv2.getGaussianKernel(k, s).T
        c.eq(f"k={k}, sigma={s}: matches cv2.getGaussianKernel outer product", K, ref, tol=1e-6)
    c.true("larger sigma spreads the weight outwards",
           np.asarray(fn(9, 3.0))[4, 4] < np.asarray(fn(9, 1.0))[4, 4])
    c.true("it is separable: the matrix is rank 1",
           np.linalg.matrix_rank(np.asarray(fn(7, 1.5)), tol=1e-10) == 1,
           "a 2-D Gaussian is an outer product, so its rank must be exactly 1")
    return c.done()


def check_median(fn):
    """fn(gray, k) -> median filter with replicate padding (so it matches cv2.medianBlur)."""
    c = _Check("median_filter")
    g = load("lena.jpg", gray=True)[:90, :110]
    out = np.asarray(fn(g, 3))
    _basic(c, out, g)
    c.eq("k=1 returns the image unchanged", fn(g, 1), g)
    for k in (3, 5, 7):
        c.eq(f"k={k} matches cv2.medianBlur", fn(g, k), cv2.medianBlur(g, k))
    spike = np.full((7, 7), 50, np.uint8); spike[3, 3] = 255
    c.eq("a lone salt spike is removed completely", np.asarray(fn(spike, 3))[3, 3], 50)
    edge = np.zeros((9, 9), np.uint8); edge[:, 5:] = 200
    c.eq("a straight step edge is preserved exactly", fn(edge, 3), edge,
         )
    noisy = g.copy()
    rng = np.random.default_rng(0)
    m = rng.random(g.shape) < 0.1
    noisy[m] = 255
    c.true("beats a box blur on salt noise (that is the point of the median)",
           np.abs(np.asarray(fn(noisy, 3)).astype(int) - g.astype(int)).mean()
           < np.abs(cv2.blur(noisy, (3, 3)).astype(int) - g.astype(int)).mean())
    return c.done()


def check_unsharp(fn):
    """fn(gray, sigma, amount) -> gray + amount*(gray - GaussianBlur(gray, (0,0), sigma))."""
    c = _Check("unsharp_mask")
    g = load("lena.jpg", gray=True)[:100, :120]
    out = np.asarray(fn(g, 1.0, 1.0))
    _basic(c, out, g)
    c.eq("amount=0 returns the image unchanged", fn(g, 1.0, 0.0), g, tol=1)
    ref = lambda s, a: np.clip(g.astype(np.float64)
                               + a * (g.astype(np.float64) - cv2.GaussianBlur(g, (0, 0), s)),
                               0, 255).round().astype(np.uint8)
    for s, a in [(1.0, 1.0), (2.0, 1.5), (3.0, 0.5)]:
        c.eq(f"sigma={s}, amount={a} matches the specified formula", fn(g, s, a), ref(s, a), tol=1)
    flat = np.full((30, 30), 120, np.uint8)
    c.eq("a flat region is untouched (nothing to sharpen)",
         np.asarray(fn(flat, 1.0, 2.0))[10:20, 10:20], np.full((10, 10), 120, np.uint8), tol=1)
    c.true("output stays inside 0..255",
           np.asarray(fn(g, 1.0, 5.0)).min() >= 0 and np.asarray(fn(g, 1.0, 5.0)).max() <= 255,
           "clip before casting -- a strong amount overshoots both ends")
    c.true("sharpening raises local contrast",
           np.asarray(fn(g, 1.0, 1.5)).std() > g.std())
    return c.done()


# ================================================================ 09: histogram processing

def check_histogram(fn):
    """fn(gray) -> length-256 array of pixel counts."""
    c = _Check("histogram")
    r = ramp()
    h = np.asarray(fn(r))
    c.eq("returns 256 bins", h.shape, (256,))
    c.eq("a ramp of 8 rows puts 8 pixels in every bin", h, np.full(256, 8))
    c.eq("counts sum to the pixel count", h.sum(), r.size)
    z = np.zeros((4, 5), np.uint8)
    hz = np.asarray(fn(z))
    c.eq("an all-black image puts everything in bin 0", hz[0], 20)
    c.eq("...and nothing anywhere else", hz[1:].sum(), 0)
    for name in ("lena.jpg", "baboon.jpg", "board.jpg"):
        g = load(name, gray=True)
        c.eq(f"matches np.bincount on {name}", fn(g), np.bincount(g.ravel(), minlength=256))
    return c.done()


def check_equalize(fn):
    """fn(gray) -> histogram-equalised image, matching cv2.equalizeHist exactly."""
    c = _Check("equalize")
    g = load("board.jpg", gray=True)
    out = np.asarray(fn(g))
    _basic(c, out, g)
    for name in ("board.jpg", "lena.jpg", "baboon.jpg", "sudoku.png"):
        x = load(name, gray=True)
        c.eq(f"matches cv2.equalizeHist on {name}", fn(x), cv2.equalizeHist(x))
    c.eq("the darkest pixel is pushed to 0", out.min(), 0)
    c.eq("the brightest pixel is pushed to 255", out.max(), 255)
    c.true("contrast increases on a low-contrast image", out.std() > g.std())
    c.true("the mapping is monotonic in the input", np.all(
        np.diff([np.asarray(fn(g))[g == v].min() if (g == v).any() else -1
                 for v in range(256) if (g == v).any()]) >= 0),
        "equalisation is a LUT, so a brighter input can never become darker")
    flat = np.full((16, 16), 77, np.uint8)
    o = np.asarray(fn(flat))
    c.true("a single-valued image stays single-valued (no divide-by-zero)",
           len(np.unique(o)) == 1, "guard the case where every pixel has the same value")
    c.true("equalising twice is (almost) the same as once",
           np.abs(np.asarray(fn(out)).astype(int) - out.astype(int)).mean() < 2.0,
           "equalisation is close to idempotent -- a big change means the CDF is wrong")
    return c.done()


def check_match_histogram(fn):
    """fn(src, ref) -> src remapped so its histogram approximates ref's."""
    c = _Check("match_histogram")
    src = load("board.jpg", gray=True)
    ref = load("lena.jpg", gray=True)
    out = np.asarray(fn(src, ref))
    _basic(c, out, src)
    cdf = lambda x: np.bincount(x.ravel(), minlength=256).cumsum() / x.size
    err = np.abs(cdf(out) - cdf(ref)).max()
    c.true(f"output CDF tracks the reference CDF (max gap {err:.3f}, needs < 0.06)", err < 0.06,
           "the mapping is not landing on the reference distribution")
    c.true("closer to the reference than the source was",
           err < np.abs(cdf(src) - cdf(ref)).max())
    c.eq("matching an image to itself is (almost) the identity", fn(src, src), src, tol=2)
    c.true("the mapping never decreases", np.all(
        np.diff([out[src == v].min() for v in range(256) if (src == v).any()]) >= 0),
        "it is a LUT built from two monotone CDFs, so it must be monotone")
    unif = ramp(64)
    c.true("matching to a flat histogram behaves like equalisation",
           np.abs(np.asarray(fn(src, unif)).astype(int)
                  - cv2.equalizeHist(src).astype(int)).mean() < 6.0)
    c.true("works when the reference is darker than the source",
           np.asarray(fn(ref, src)).mean() < ref.mean())
    return c.done()
