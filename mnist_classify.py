import gzip
import struct
from pathlib import Path

import numpy as np


def load_images(path: Path) -> np.ndarray:
    with gzip.open(path, "rb") as f:
        magic, num, rows, cols = struct.unpack(">IIII", f.read(16))
        if magic != 2051:
            raise ValueError(f"Unexpected magic for images: {magic}")
        data = f.read()
    return np.frombuffer(data, dtype=np.uint8).reshape(num, rows, cols)


def load_labels(path: Path) -> np.ndarray:
    with gzip.open(path, "rb") as f:
        magic, num = struct.unpack(">II", f.read(8))
        if magic != 2049:
            raise ValueError(f"Unexpected magic for labels: {magic}")
        data = f.read()
    labels = np.frombuffer(data, dtype=np.uint8)
    if labels.shape[0] != num:
        raise ValueError("Label count mismatch")
    return labels


# ---------------------------------------------------------------------------
# Image preprocessing
# ---------------------------------------------------------------------------

def binarize(image: np.ndarray, threshold: int = 48) -> np.ndarray:
    return image >= threshold


def crop_to_foreground(mask: np.ndarray) -> np.ndarray:
    ys, xs = np.where(mask)
    if ys.size == 0:
        return mask
    y0, y1 = ys.min(), ys.max()
    x0, x1 = xs.min(), xs.max()
    pad = 1
    y0 = max(0, y0 - pad)
    x0 = max(0, x0 - pad)
    y1 = min(mask.shape[0] - 1, y1 + pad)
    x1 = min(mask.shape[1] - 1, x1 + pad)
    return mask[y0 : y1 + 1, x0 : x1 + 1]


# ---------------------------------------------------------------------------
# Topology: connected components and holes
# ---------------------------------------------------------------------------

def connected_components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    components: list[list[tuple[int, int]]] = []
    for y in range(h):
        for x in range(w):
            if not mask[y, x] or seen[y, x]:
                continue
            stack = [(y, x)]
            seen[y, x] = True
            component: list[tuple[int, int]] = []
            while stack:
                cy, cx = stack.pop()
                component.append((cy, cx))
                for dy in (-1, 0, 1):
                    for dx in (-1, 0, 1):
                        if dy == 0 and dx == 0:
                            continue
                        ny, nx = cy + dy, cx + dx
                        if 0 <= ny < h and 0 <= nx < w and mask[ny, nx] and not seen[ny, nx]:
                            seen[ny, nx] = True
                            stack.append((ny, nx))
            components.append(component)
    return components


def close_small_gaps(mask: np.ndarray) -> np.ndarray:
    closed = mask.copy()
    h, w = mask.shape
    for y in range(1, h - 1):
        for x in range(1, w - 1):
            if mask[y, x]:
                continue
            if (mask[y, x - 1] and mask[y, x + 1]) or \
               (mask[y - 1, x] and mask[y + 1, x]) or \
               (mask[y - 1, x - 1] and mask[y + 1, x + 1]) or \
               (mask[y - 1, x + 1] and mask[y + 1, x - 1]):
                closed[y, x] = True
    return closed


def find_holes(mask: np.ndarray) -> list[tuple[float, float, float]]:
    """Returns list of (relative_area, center_y, center_x) for each hole."""
    padded = np.pad(mask, 1, constant_values=False)
    background = ~padded
    components = connected_components(background)
    holes: list[tuple[float, float, float]] = []
    area = float(mask.shape[0] * mask.shape[1])
    for component in components:
        touches_border = False
        total_y, total_x = 0, 0
        for y, x in component:
            if y == 0 or x == 0 or y == padded.shape[0] - 1 or x == padded.shape[1] - 1:
                touches_border = True
                break
            total_y += y - 1
            total_x += x - 1
        if touches_border:
            continue
        size = len(component)
        holes.append((size / area, total_y / size / mask.shape[0], total_x / size / mask.shape[1]))
    return holes


# ---------------------------------------------------------------------------
# Feature: projection profiles
# ---------------------------------------------------------------------------

def vertical_profile(mask: np.ndarray) -> np.ndarray:
    """Sum of foreground pixels per column, normalized to [0,1]."""
    profile = mask.sum(axis=0).astype(float)
    mx = profile.max()
    if mx > 0:
        profile /= mx
    return profile


def sample_profile(profile: np.ndarray, n: int = 10) -> list[float]:
    """Resample a profile to n evenly spaced values."""
    length = len(profile)
    if length == 0:
        return [0.0] * n
    indices = [int(round(i * (length - 1) / (n - 1))) for i in range(n)]
    return [float(profile[idx]) for idx in indices]


# ---------------------------------------------------------------------------
# Feature: stroke crossings
# ---------------------------------------------------------------------------

def count_crossings(line: np.ndarray) -> int:
    """Count foreground-to-background transitions (number of runs)."""
    runs = 0
    in_fg = False
    for v in line:
        if v and not in_fg:
            runs += 1
            in_fg = True
        elif not v:
            in_fg = False
    return runs


def horizontal_crossings(mask: np.ndarray, fractions: list[float]) -> list[int]:
    """Count stroke crossings at given row fractions."""
    h = mask.shape[0]
    result = []
    for f in fractions:
        y = min(h - 1, int(round(f * (h - 1))))
        result.append(count_crossings(mask[y, :]))
    return result


def vertical_crossings(mask: np.ndarray, fractions: list[float]) -> list[int]:
    """Count stroke crossings at given column fractions."""
    w = mask.shape[1]
    result = []
    for f in fractions:
        x = min(w - 1, int(round(f * (w - 1))))
        result.append(count_crossings(mask[:, x]))
    return result


# ---------------------------------------------------------------------------
# Feature: grid densities (3x3)
# ---------------------------------------------------------------------------

def region_density(mask: np.ndarray, y0: float, y1: float, x0: float, x1: float) -> float:
    h, w = mask.shape
    ya = min(h, max(0, int(round(y0 * h))))
    yb = min(h, max(ya + 1, int(round(y1 * h))))
    xa = min(w, max(0, int(round(x0 * w))))
    xb = min(w, max(xa + 1, int(round(x1 * w))))
    return float(mask[ya:yb, xa:xb].mean())


# ---------------------------------------------------------------------------
# Feature: symmetry
# ---------------------------------------------------------------------------

def vertical_symmetry(mask: np.ndarray) -> float:
    return 1.0 - float(np.logical_xor(mask, np.fliplr(mask)).mean())


def horizontal_symmetry(mask: np.ndarray) -> float:
    return 1.0 - float(np.logical_xor(mask, np.flipud(mask)).mean())


# ---------------------------------------------------------------------------
# Feature: stroke width and position
# ---------------------------------------------------------------------------

def row_span(line: np.ndarray) -> tuple[float, float, float]:
    """Return (left_start, right_end, width) normalized to [0,1]."""
    positions = np.flatnonzero(line)
    if positions.size == 0:
        return 0.5, 0.5, 0.0
    n = float(line.shape[0])
    return float(positions[0] / n), float(positions[-1] / n), float(positions.size / n)


def col_span(line: np.ndarray) -> tuple[float, float, float]:
    """Return (top_start, bottom_end, height) normalized to [0,1]."""
    return row_span(line)


# ---------------------------------------------------------------------------
# Feature: diagonal analysis
# ---------------------------------------------------------------------------

def diagonal_crossings(mask: np.ndarray, reverse: bool = False) -> int:
    h, w = mask.shape
    line = []
    for x in range(w):
        y = min(h - 1, int(round(x * (h - 1) / max(1, w - 1))))
        sx = w - 1 - x if reverse else x
        line.append(bool(mask[y, sx]))
    return count_crossings(np.array(line, dtype=bool))


# ---------------------------------------------------------------------------
# All features bundled
# ---------------------------------------------------------------------------

def extract_features(image: np.ndarray) -> dict:
    mask = crop_to_foreground(binarize(image))
    h, w = mask.shape

    if not mask.any():
        return {"empty": True}

    # Basic geometry
    ys, xs = np.where(mask)
    aspect = w / h
    fill = float(mask.mean())
    center_y = float(ys.mean() / h)
    center_x = float(xs.mean() / w)

    # Holes
    holes = find_holes(mask)
    repaired_mask = close_small_gaps(mask)
    repaired_holes = find_holes(repaired_mask)
    n_holes = len(holes)
    n_repaired_holes = len(repaired_holes)
    largest_hole = max((ho[0] for ho in holes), default=0.0)
    repaired_largest = max((ho[0] for ho in repaired_holes), default=0.0)
    hole_cy = sum(ho[1] for ho in holes) / n_holes if n_holes else 0.5
    hole_cx = sum(ho[2] for ho in holes) / n_holes if n_holes else 0.5

    # Region densities (original thirds)
    top = region_density(mask, 0.0, 0.33, 0.0, 1.0)
    middle = region_density(mask, 0.33, 0.66, 0.0, 1.0)
    bottom = region_density(mask, 0.66, 1.0, 0.0, 1.0)
    left = region_density(mask, 0.0, 1.0, 0.0, 0.33)
    right = region_density(mask, 0.0, 1.0, 0.66, 1.0)
    center = region_density(mask, 0.0, 1.0, 0.33, 0.66)
    ul = region_density(mask, 0.0, 0.5, 0.0, 0.5)
    ur = region_density(mask, 0.0, 0.5, 0.5, 1.0)
    ll = region_density(mask, 0.5, 1.0, 0.0, 0.5)
    lr = region_density(mask, 0.5, 1.0, 0.5, 1.0)

    # Stroke crossings at multiple scan lines
    hc = horizontal_crossings(mask, [0.1, 0.2, 0.35, 0.5, 0.65, 0.8, 0.9])
    vc = vertical_crossings(mask, [0.2, 0.35, 0.5, 0.65, 0.8])

    # Row/col spans at key positions
    rs10 = row_span(mask[min(h - 1, int(round(0.1 * (h - 1)))), :])
    rs20 = row_span(mask[min(h - 1, int(round(0.2 * (h - 1)))), :])
    rs50 = row_span(mask[min(h - 1, int(round(0.5 * (h - 1)))), :])
    rs80 = row_span(mask[min(h - 1, int(round(0.8 * (h - 1)))), :])
    rs90 = row_span(mask[min(h - 1, int(round(0.9 * (h - 1)))), :])
    cs50 = col_span(mask[:, min(w - 1, int(round(0.5 * (w - 1))))])
    cs80 = col_span(mask[:, min(w - 1, int(round(0.8 * (w - 1))))])

    # Symmetry
    vsym = vertical_symmetry(mask)
    hsym = horizontal_symmetry(mask)

    # Diagonals
    main_diag = diagonal_crossings(mask, reverse=False)
    anti_diag = diagonal_crossings(mask, reverse=True)

    # Projection profiles (sampled to 10 points)
    vprof = sample_profile(vertical_profile(mask), 10)

    return {
        "aspect": aspect,
        "fill": fill,
        "cy": center_y,
        "cx": center_x,
        "n_holes": n_holes,
        "n_rh": n_repaired_holes,
        "largest_hole": largest_hole,
        "rep_largest": repaired_largest,
        "hole_cy": hole_cy,
        "hole_cx": hole_cx,
        # region densities
        "top": top, "middle": middle, "bottom": bottom,
        "left": left, "right": right, "center": center,
        "ul": ul, "ur": ur, "ll": ll, "lr": lr,
        # horizontal crossings
        "hc10": hc[0], "hc20": hc[1], "hc35": hc[2], "hc50": hc[3], "hc65": hc[4], "hc80": hc[5], "hc90": hc[6],
        # vertical crossings
        "vc20": vc[0], "vc35": vc[1], "vc50": vc[2], "vc65": vc[3], "vc80": vc[4],
        # row spans
        "rs10_left": rs10[0], "rs10_w": rs10[2],
        "rs20_left": rs20[0], "rs20_w": rs20[2],
        "rs50_left": rs50[0], "rs50_w": rs50[2],
        "rs80_left": rs80[0], "rs80_w": rs80[2],
        "rs90_left": rs90[0], "rs90_w": rs90[2],
        # col spans
        "cs50_top": cs50[0], "cs50_h": cs50[2],
        "cs80_top": cs80[0], "cs80_h": cs80[2],
        # symmetry
        "vsym": vsym,
        "hsym": hsym,
        # diagonals
        "main_diag": main_diag,
        "anti_diag": anti_diag,
        # profile
        "vp": vprof,
    }


# ---------------------------------------------------------------------------
# Per-digit detectors
# ---------------------------------------------------------------------------

def detect_one(f: dict) -> float:
    """Digit 1: narrow, single stroke, no holes."""
    score = 0.0
    score += 8.0 * (f["n_holes"] == 0)
    score += 4.0 * (f["aspect"] < 0.58)
    score += 3.0 * (f["hc20"] <= 1 and f["hc50"] <= 1 and f["hc80"] <= 1)
    score += 2.5 * (f["vc50"] <= 1)
    score += 1.0 * (f["vc35"] < 2)
    score += 2.0 * (f["center"] > max(f["left"], f["right"]) * 1.7)
    score += 1.0 * (f["aspect"] < 0.52)
    score += 1.0 * (f["center"] > max(f["left"], f["right"]) * 2.0)
    # vertical profile: 1 has a narrow spike — most columns are empty
    vp = f["vp"]
    low_cols = sum(1 for v in vp if v < 0.3)
    score += 1.5 * (low_cols >= 5)
    score -= 2.0 * f["fill"]
    score -= 1.0 * (f["vc50"] > 1)
    return score


def detect_zero(f: dict) -> float:
    """Digit 0: one hole, vertically symmetric, oval shape."""
    score = 0.0
    repaired_loop = f["n_rh"] == 1 and f["rep_largest"] > 0.08
    broken_loop = f["n_holes"] == 0 and repaired_loop
    balanced = abs(f["left"] - f["right"]) < 0.1 and abs(f["top"] - f["bottom"]) < 0.1 and f["rs80_w"] > 0.42

    score += 8.0 * (f["n_holes"] == 1)
    score += 4.0 * broken_loop
    score += 1.0 * balanced
    score += 3.5 * f["vsym"]
    score += 2.5 * (f["hc50"] == 2)
    score += 2.0 * (f["vc50"] == 2)
    score += 1.5 * (f["largest_hole"] > 0.1)
    score += 2.0 * (0.42 <= f["hole_cy"] <= 0.6)
    score += 1.5 * (abs(f["top"] - f["bottom"]) < 0.12)
    score += 1.0 * (0.28 <= f["rs50_w"] <= 0.5)
    score -= 2.5 * (f["hole_cy"] > 0.62)
    score -= 2.5 * (f["hole_cy"] < 0.4)
    score -= 2.0 * (f["n_holes"] >= 2)
    return score


def detect_two(f: dict) -> float:
    """Digit 2: no holes, top-right curve sweeps to bottom-left base."""
    score = 0.0
    sweeping = f["rs50_left"] > 0.42 and f["rs50_w"] < 0.32 and f["rs80_w"] > 0.5

    score += 6.0 * (f["n_holes"] == 0 or f["largest_hole"] < 0.03)
    score += 2.5 * (f["top"] > f["middle"] * 1.02)
    score += 3.0 * (f["bottom"] > f["middle"] * 1.15)
    score += 2.5 * (f["ll"] > f["lr"] * 1.05)
    score += 2.0 * (f["hc50"] <= 1)
    score += 1.5 * (f["vc50"] >= 2)
    score += 1.5 * (f["rs50_left"] > 0.4)
    score += 1.5 * (f["rs80_w"] > 0.5)
    score += 1.5 * (f["rs20_left"] < 0.38)
    score += 2.0 * sweeping
    score += 1.5 * (f["main_diag"] >= f["anti_diag"] + 0.2)
    score -= 1.5 * (f["rs50_left"] < 0.3)
    score -= 2.0 * (f["n_holes"] == 0)
    score -= 1.5 * (f["rs50_w"] > 0.38)
    return score


def detect_three(f: dict) -> float:
    """Digit 3: no holes, right-heavy, two bumps at top and bottom."""
    score = 0.0
    score += 6.0 * (f["n_holes"] == 0)
    score += 3.0 * (f["right"] > f["left"] * 1.35)
    score += 2.0 * (f["top"] > 0.22)
    score += 2.0 * (f["bottom"] > 0.22)
    score += 2.0 * (f["hc20"] <= 1 and f["hc80"] <= 1)
    score += 1.5 * (f["vc50"] >= 3)
    score += 2.0 * (f["vc50"] >= 3)
    score += 2.0 * (f["rs50_w"] > 0.35)
    score += 1.5 * (f["rs80_left"] > 0.16)
    score -= 2.0 * (f["rs50_left"] < 0.24)
    score -= 2.0 * (f["left"] > f["right"] * 0.95)
    score -= 1.5 * (f["ll"] > f["lr"] * 1.1)
    return score


def detect_four(f: dict) -> float:
    """Digit 4: open top, dense middle crossbar, narrow bottom."""
    score = 0.0
    score += 6.0 * (f["n_holes"] <= 1)
    score += 3.0 * (f["middle"] > f["top"] * 1.45)
    score += 3.0 * (f["middle"] > f["bottom"] * 2.0)
    score += 2.5 * (f["hc20"] >= 2)
    score += 2.5 * (f["hc35"] >= 2)
    score += 2.0 * (f["hc80"] <= 1)
    score += 1.5 * (f["right"] >= f["left"] * 0.9)
    score += 1.0 * (f["vc50"] <= 1)
    score += 1.0 * (f["middle"] > f["bottom"] * 2.2)
    score -= 2.0 * _broken_loop_nine(f)
    score -= 0.5 * (f["cs50_top"] < 0.14)
    return score


def detect_five(f: dict) -> float:
    """Digit 5: left-heavy, top bar, lower-right bowl."""
    score = 0.0
    broken_loop_zero = f["n_holes"] == 0 and f["n_rh"] == 1 and f["rep_largest"] > 0.08

    score += 6.0 * (f["n_holes"] == 0)
    score += 3.0 * (f["left"] > f["right"] * 1.15)
    score += 2.0 * (f["top"] > f["middle"] * 1.02)
    score += 2.0 * (f["bottom"] > f["middle"] * 0.95)
    score += 2.0 * (f["lr"] > f["ll"] * 0.75)
    score += 1.5 * (f["vc50"] >= 3)
    score += 2.0 * (f["ul"] > f["ur"])
    score += 2.0 * (f["rs50_left"] < 0.3)
    score += 1.5 * (f["rs80_w"] < 0.5)
    score += 0.5 * (f["cs80_top"] < 0.12)
    score += 1.0 * (f["left"] > f["right"] * 1.2)
    score += 1.5 * (f["anti_diag"] >= f["main_diag"] + 0.5)
    score -= 2.0 * broken_loop_zero
    score -= 1.0 * (f["bottom"] > f["top"] * 1.2)
    score -= 1.0 * (f["main_diag"] >= f["anti_diag"] + 0.4)
    score -= 1.0 * (f["rs50_left"] > 0.32)
    score -= 1.5 * (f["main_diag"] > f["anti_diag"] + 0.2)
    return score


def detect_six(f: dict) -> float:
    """Digit 6: bottom-heavy, one hole in lower half."""
    score = 0.0
    diag_balance = f["main_diag"] - f["anti_diag"]
    broken_loop = (
        f["n_holes"] == 0 and f["n_rh"] == 1
        and f["rep_largest"] > 0.02
        and f["bottom"] > f["top"] * 1.25
        and f["hc80"] >= 2
    )

    score += 8.0 * (f["n_holes"] == 1)
    score += 3.0 * broken_loop
    score += 3.5 * (
        f["n_holes"] == 0
        and f["top"] < 0.25
        and f["bottom"] > f["middle"] * 1.1
        and f["cs50_top"] > 0.14
        and diag_balance > 0.45
    )
    score += 3.5 * (f["hole_cy"] > 0.56)
    score += 2.5 * (f["bottom"] > f["top"] * 1.35)
    score += 2.0 * (f["left"] >= f["right"] * 1.05)
    score += 2.0 * (f["hc65"] >= 2)
    score += 1.5 * (f["hc20"] <= 1)
    score += 1.5 * (f["largest_hole"] > 0.015)
    score += 1.0 * (f["cs50_top"] > 0.12)
    score += 1.0 * (f["middle"] > f["top"] * 1.35)
    score += 1.0 * (diag_balance > 0.6)
    score += 2.5 * (f["main_diag"] > f["anti_diag"] + 0.35)
    score += 1.0 * (f["rs50_left"] < 0.18)
    score += 0.75 * (f["bottom"] > f["top"] * 1.6)
    score -= 2.5 * (f["top"] > 0.26)
    score -= 2.0 * (f["rs50_left"] > 0.28)
    score -= 2.5 * (f["anti_diag"] > f["main_diag"] + 0.2)
    score -= 1.5 * (diag_balance < 0.2)
    return score


def detect_seven(f: dict) -> float:
    """Digit 7: top bar, diagonal stroke downward, no holes."""
    score = 0.0
    score += 6.0 * (f["n_holes"] == 0)
    score += 3.0 * (f["top"] > f["middle"] * 1.3)
    score += 3.0 * (f["top"] > f["bottom"] * 1.7)
    score += 2.5 * (f["hc80"] <= 1)
    score += 2.0 * (f["hc65"] <= 1)
    score += 1.5 * (f["ur"] > f["ul"])
    score += 1.0 * (f["cs50_top"] < 0.091)
    score += 1.0 * (f["rs80_w"] < 0.24)
    score += 1.0 * (f["bottom"] < f["top"] * 0.75)
    score -= 2.0 * (f["rs50_w"] > 0.42)
    score -= 1.5 * (f["vc50"] > 2)
    score -= 1.5 * (f["n_rh"] > 0)
    score -= 1.0 * (f["cs50_top"] > 0.12)
    return score


def detect_eight(f: dict) -> float:
    """Digit 8: two holes, symmetric, pinched middle."""
    score = 0.0
    score += 9.0 * (f["n_holes"] >= 2)
    score += 5.0 * (f["n_holes"] == 1)
    score += 3.0 * (f["n_rh"] >= 2)
    score += 1.5 * (f["n_holes"] == 1 and f["left"] >= f["right"] * 1.05 and f["hc50"] <= 1)
    score += 1.5 * (f["n_holes"] == 1 and f["n_rh"] >= 2)
    score += 2.0 * f["vsym"]
    score += 2.0 * f["hsym"]
    score += 2.0 * (f["hc20"] >= 2)
    score += 2.0 * (f["hc80"] >= 2)
    score += 1.5 * (f["vc50"] >= 3)
    score += 1.5 * (f["largest_hole"] > 0.025)
    score += 1.5 * (0.35 <= f["hole_cy"] <= 0.65)
    score += 1.5 * (f["rs80_w"] < 0.52)
    score += 1.5 * (f["ll"] > f["lr"] * 1.08)
    score -= 1.5 * (f["n_holes"] == 1 and f["largest_hole"] > 0.1)
    return score


def detect_nine(f: dict) -> float:
    """Digit 9: one hole in upper half, top-heavy."""
    score = 0.0
    score += 8.0 * (f["n_holes"] == 1)
    score += 3.5 * _broken_loop_nine(f)
    score += 3.0 * _slashed_nine(f)
    score += 3.5 * (f["hole_cy"] < 0.42)
    score += 2.5 * (f["top"] > f["bottom"] * 1.35)
    score += 2.0 * (f["right"] >= f["left"] * 1.05)
    score += 2.0 * (f["hc20"] >= 2)
    score += 1.5 * (f["hc80"] <= 1)
    score += 1.5 * (f["rs80_left"] > 0.35)
    score += 1.5 * (f["rs80_w"] < 0.32)
    score += 1.0 * (f["largest_hole"] > 0.03)
    score += 1.5 * (f["rs50_w"] > 0.4)
    score += 1.5 * (f["n_rh"] > 0)
    score += 0.5 * (f["cs50_top"] < 0.11)
    score -= 1.0 * (f["vc50"] <= 1)
    score -= 1.0 * (f["cs50_top"] > 0.18)
    return score


# Helper conditions used across detectors
def _broken_loop_nine(f: dict) -> bool:
    return (
        f["n_holes"] == 0
        and f["n_rh"] == 1
        and f["rep_largest"] > 0.03
        and f["top"] > f["bottom"] * 1.3
        and f["rs80_left"] > 0.35
        and f["rs80_w"] < 0.28
    )


def _slashed_nine(f: dict) -> bool:
    return (
        f["n_holes"] == 0
        and f["top"] > f["bottom"] * 1.45
        and f["rs50_left"] > 0.45
        and f["rs50_w"] < 0.24
        and f["vc50"] >= 2
    )


# ---------------------------------------------------------------------------
# Classifier
# ---------------------------------------------------------------------------

DETECTORS = [
    detect_zero,
    detect_one,
    detect_two,
    detect_three,
    detect_four,
    detect_five,
    detect_six,
    detect_seven,
    detect_eight,
    detect_nine,
]


class DigitClassifier:
    def classify(self, image: np.ndarray) -> int:
        features = extract_features(image)
        if features.get("empty"):
            return 0

        scores = [det(features) for det in DETECTORS]

        # Sort by score descending
        order = sorted(range(10), key=lambda d: scores[d], reverse=True)
        best = order[0]
        second = order[1]
        third = order[2]

        result = self.break_tie(features, scores, best, second)

        # Second pass: if result is close to third candidate, check again
        # Check third candidate
        if abs(scores[result] - scores[third]) <= 3.5:
            alt = self.break_tie(features, scores, result, third)
            if alt != result:
                result = alt

        return result

    def break_tie(self, f: dict, scores: list[float], best: int, second: int) -> int:
        pair = {best, second}
        gap = abs(scores[best] - scores[second])

        if pair == {3, 5} and gap <= 2.0:
            if f["ul"] > f["ur"] * 1.08 and f["rs50_left"] < 0.24:
                return 5
            if f["left"] > f["right"] * 1.08 and f["anti_diag"] > f["main_diag"] + 0.25:
                return 5
            if f["right"] > f["left"] * 1.1 and f["rs50_left"] > 0.22:
                return 3
            if f["anti_diag"] > f["main_diag"] + 0.2 and f["rs50_left"] < 0.26:
                return 5
            if f["rs50_left"] < 0.14:
                return 5
            return 3

        if pair == {2, 8} and gap <= 2.5:
            if f["rs50_left"] > 0.45 and (f["top"] < 0.34 or f["hole_cy"] > 0.62):
                return 2
            return 8

        if pair == {2, 3} and gap <= 3.0:
            if f["ll"] >= 0.415:
                return 2
            return 3

        if pair == {2, 6} and gap <= 3.0:
            if f["rs50_left"] >= 0.357:
                return 2
            return 6

        if pair == {4, 9} and gap <= 3.5:
            if f["cs50_top"] >= 0.182:
                return 4
            return 9

        if pair == {5, 6} and gap <= 2.5:
            if f["n_rh"] == 1 and f["rep_largest"] > 0.02:
                return 6
            if f["bottom"] > f["top"] * 1.25 and f["hc80"] >= 2 and f["main_diag"] >= f["anti_diag"]:
                return 6
            if f["rs20_left"] < 0.25:
                return 5
            return 6

        if pair == {6, 8} and gap <= 3.0:
            if f["hc35"] <= 1:
                return 6
            return 8

        if pair == {7, 9} and gap <= 3.0:
            if _slashed_nine(f):
                return 9
            return 7

        if pair == {8, 9} and gap <= 3.0:
            if f["n_holes"] >= 2 or f["n_rh"] >= 2:
                return 8
            if f["n_holes"] == 1 and f["left"] >= f["right"] * 1.05:
                return 8
            if (
                f["n_holes"] == 1
                and f["hc80"] >= 2
                and f["rs80_left"] < 0.24
                and f["rep_largest"] > 0.035
                and f["ll"] > f["lr"] * 1.35
            ):
                return 8
            return 9

        return best


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    images_path = Path("train-images-idx3-ubyte.gz")
    labels_path = Path("train-labels-idx1-ubyte.gz")

    images = load_images(images_path)
    labels = load_labels(labels_path)

    classifier = DigitClassifier()

    correct = 0
    for image, label in zip(images, labels):
        if classifier.classify(image) == int(label):
            correct += 1

    total = labels.shape[0]
    accuracy = correct / total
    print(f"Accuracy: {accuracy:.4f} ({correct}/{total})")


if __name__ == "__main__":
    main()
