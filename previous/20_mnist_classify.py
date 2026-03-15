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


def binarize(image: np.ndarray) -> np.ndarray:
    return image >= 48


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


def region_density(mask: np.ndarray, y0: float, y1: float, x0: float, x1: float) -> float:
    h, w = mask.shape
    ya = min(h, max(0, int(round(y0 * h))))
    yb = min(h, max(ya + 1, int(round(y1 * h))))
    xa = min(w, max(0, int(round(x0 * w))))
    xb = min(w, max(xa + 1, int(round(x1 * w))))
    return float(mask[ya:yb, xa:xb].mean())


def count_foreground_runs(line: np.ndarray) -> float:
    runs = 0
    in_run = False
    for value in line:
        if value and not in_run:
            runs += 1
            in_run = True
        elif not value:
            in_run = False
    return float(runs)


def line_edges(line: np.ndarray) -> tuple[float, float, float]:
    positions = np.flatnonzero(line)
    if positions.size == 0:
        return 0.5, 0.5, 0.0
    length = float(line.shape[0])
    return float(positions[0] / length), float(positions[-1] / length), float(positions.size / length)


def diagonal_runs(mask: np.ndarray, reverse: bool = False) -> float:
    h, w = mask.shape
    line = []
    for x in range(w):
        y = min(h - 1, int(round(x * (h - 1) / max(1, w - 1))))
        sample_x = w - 1 - x if reverse else x
        line.append(bool(mask[y, sample_x]))
    return count_foreground_runs(np.array(line, dtype=bool))


def close_small_gaps(mask: np.ndarray) -> np.ndarray:
    closed = mask.copy()
    h, w = mask.shape

    for y in range(1, h - 1):
        for x in range(1, w - 1):
            if mask[y, x]:
                continue
            horizontal_gap = mask[y, x - 1] and mask[y, x + 1]
            vertical_gap = mask[y - 1, x] and mask[y + 1, x]
            main_diagonal_gap = mask[y - 1, x - 1] and mask[y + 1, x + 1]
            anti_diagonal_gap = mask[y - 1, x + 1] and mask[y + 1, x - 1]
            if horizontal_gap or vertical_gap or main_diagonal_gap or anti_diagonal_gap:
                closed[y, x] = True

    return closed


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
                        ny = cy + dy
                        nx = cx + dx
                        if 0 <= ny < h and 0 <= nx < w and mask[ny, nx] and not seen[ny, nx]:
                            seen[ny, nx] = True
                            stack.append((ny, nx))
            components.append(component)
    return components


def find_holes(mask: np.ndarray) -> list[tuple[float, float, float]]:
    padded = np.pad(mask, 1, constant_values=False)
    background = ~padded
    components = connected_components(background)
    holes: list[tuple[float, float, float]] = []
    area = float(mask.shape[0] * mask.shape[1])

    for component in components:
        touches_border = False
        total_y = 0
        total_x = 0
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


def extract_features(image: np.ndarray) -> dict[str, float]:
    mask = crop_to_foreground(binarize(image))
    h, w = mask.shape
    row_levels = (20, 35, 50, 65, 80)
    col_levels = (20, 35, 50, 65, 80)
    span_levels = (20, 50, 80)
    if not mask.any():
        return {
            "aspect_ratio": 1.0,
            "fill": 0.0,
            "center_y": 0.5,
            "center_x": 0.5,
            "holes": 0.0,
            "repaired_holes": 0.0,
            "largest_hole": 0.0,
            "repaired_largest_hole": 0.0,
            "hole_y": 0.5,
            "hole_x": 0.5,
            "top": 0.0,
            "middle": 0.0,
            "bottom": 0.0,
            "left": 0.0,
            "center": 0.0,
            "right": 0.0,
            "upper_left": 0.0,
            "upper_right": 0.0,
            "lower_left": 0.0,
            "lower_right": 0.0,
            "vertical_symmetry": 0.0,
            "horizontal_symmetry": 0.0,
            "hr20": 0.0,
            "hr35": 0.0,
            "hr50": 0.0,
            "hr65": 0.0,
            "hr80": 0.0,
            "vc20": 0.0,
            "vc35": 0.0,
            "vc50": 0.0,
            "vc65": 0.0,
            "vc80": 0.0,
            "row_left_20": 0.5,
            "row_left_50": 0.5,
            "row_left_80": 0.5,
            "row_width_20": 0.0,
            "row_width_50": 0.0,
            "row_width_80": 0.0,
            "col_top_20": 0.5,
            "col_top_50": 0.5,
            "col_top_80": 0.5,
            "col_height_20": 0.0,
            "col_height_50": 0.0,
            "col_height_80": 0.0,
            "main_diag_runs": 0.0,
            "anti_diag_runs": 0.0,
        }

    ys, xs = np.where(mask)
    holes = find_holes(mask)
    repaired_holes = find_holes(close_small_gaps(mask))
    features = {
        "aspect_ratio": w / h,
        "fill": float(mask.mean()),
        "center_y": float(ys.mean() / h),
        "center_x": float(xs.mean() / w),
        "holes": float(len(holes)),
        "repaired_holes": float(len(repaired_holes)),
        "largest_hole": float(max((hole[0] for hole in holes), default=0.0)),
        "repaired_largest_hole": float(max((hole[0] for hole in repaired_holes), default=0.0)),
        "hole_y": float(sum(hole[1] for hole in holes) / len(holes)) if holes else 0.5,
        "hole_x": float(sum(hole[2] for hole in holes) / len(holes)) if holes else 0.5,
        "top": region_density(mask, 0.0, 0.33, 0.0, 1.0),
        "middle": region_density(mask, 0.33, 0.66, 0.0, 1.0),
        "bottom": region_density(mask, 0.66, 1.0, 0.0, 1.0),
        "left": region_density(mask, 0.0, 1.0, 0.0, 0.33),
        "center": region_density(mask, 0.0, 1.0, 0.33, 0.66),
        "right": region_density(mask, 0.0, 1.0, 0.66, 1.0),
        "upper_left": region_density(mask, 0.0, 0.5, 0.0, 0.5),
        "upper_right": region_density(mask, 0.0, 0.5, 0.5, 1.0),
        "lower_left": region_density(mask, 0.5, 1.0, 0.0, 0.5),
        "lower_right": region_density(mask, 0.5, 1.0, 0.5, 1.0),
        "vertical_symmetry": 1.0 - float(np.logical_xor(mask, np.fliplr(mask)).mean()),
        "horizontal_symmetry": 1.0 - float(np.logical_xor(mask, np.flipud(mask)).mean()),
    }
    for level in row_levels:
        y = int(round((level / 100.0) * (h - 1)))
        features[f"hr{level}"] = count_foreground_runs(mask[y, :])
    for level in col_levels:
        x = int(round((level / 100.0) * (w - 1)))
        features[f"vc{level}"] = count_foreground_runs(mask[:, x])
    for level in span_levels:
        y = int(round((level / 100.0) * (h - 1)))
        row_left, _, row_width = line_edges(mask[y, :])
        features[f"row_left_{level}"] = row_left
        features[f"row_width_{level}"] = row_width
    for level in span_levels:
        x = int(round((level / 100.0) * (w - 1)))
        col_top, _, col_height = line_edges(mask[:, x])
        features[f"col_top_{level}"] = col_top
        features[f"col_height_{level}"] = col_height
    features["main_diag_runs"] = diagonal_runs(mask)
    features["anti_diag_runs"] = diagonal_runs(mask, reverse=True)
    return features


class DigitClassifier:
    def score_digit(self, features: dict[str, float], digit: int) -> float:
        holes = features["holes"]
        repaired_holes = features["repaired_holes"]
        largest_hole = features["largest_hole"]
        repaired_largest_hole = features["repaired_largest_hole"]
        top = features["top"]
        middle = features["middle"]
        bottom = features["bottom"]
        left = features["left"]
        right = features["right"]
        aspect_ratio = features["aspect_ratio"]
        hole_y = features["hole_y"]
        fill = features["fill"]
        hr20 = features["hr20"]
        hr35 = features["hr35"]
        hr50 = features["hr50"]
        hr65 = features["hr65"]
        hr80 = features["hr80"]
        vc50 = features["vc50"]
        row_left_20 = features["row_left_20"]
        row_left_50 = features["row_left_50"]
        row_left_80 = features["row_left_80"]
        row_width_50 = features["row_width_50"]
        row_width_80 = features["row_width_80"]
        col_top_50 = features["col_top_50"]
        col_top_80 = features["col_top_80"]
        main_diag_runs = features["main_diag_runs"]
        anti_diag_runs = features["anti_diag_runs"]
        diag_balance = main_diag_runs - anti_diag_runs
        sweeping_two = row_left_50 > 0.42 and row_width_50 < 0.32 and row_width_80 > 0.5
        repaired_loop = repaired_holes == 1.0 and repaired_largest_hole > 0.08
        broken_loop_zero = holes == 0.0 and repaired_loop
        balanced_zero = abs(left - right) < 0.1 and abs(top - bottom) < 0.1 and row_width_80 > 0.42
        broken_loop_nine = (
            holes == 0.0
            and repaired_holes == 1.0
            and repaired_largest_hole > 0.03
            and top > bottom * 1.3
            and row_left_80 > 0.35
            and row_width_80 < 0.28
        )
        broken_loop_six = (
            holes == 0.0
            and repaired_holes == 1.0
            and repaired_largest_hole > 0.02
            and bottom > top * 1.25
            and hr80 >= 1.8
        )
        slashed_nine = (
            holes == 0.0
            and top > bottom * 1.45
            and row_left_50 > 0.45
            and row_width_50 < 0.24
            and vc50 >= 2.0
        )
        if digit == 0:
            return (
                8.0 * (holes == 1.0)
                + 4.0 * broken_loop_zero
                + 1.0 * balanced_zero
                + 3.5 * features["vertical_symmetry"]
                + 2.5 * (1.6 <= hr50 <= 2.4)
                + 2.0 * (1.6 <= vc50 <= 2.4)
                + 1.5 * (largest_hole > 0.1)
                + 2.0 * (0.42 <= hole_y <= 0.6)
                + 1.5 * (abs(top - bottom) < 0.12)
                + 1.0 * (0.28 <= row_width_50 <= 0.5)
                - 2.5 * (hole_y > 0.62)
                - 2.5 * (hole_y < 0.4)
                - 2.0 * (holes >= 2.0)
            )
        if digit == 1:
            return (
                8.0 * (holes == 0.0)
                + 4.0 * (aspect_ratio < 0.58)
                + 3.0 * (hr20 <= 1.1 and hr50 <= 1.1 and hr80 <= 1.1)
                + 2.5 * (vc50 <= 1.2)
                + 1.0 * (features["vc35"] < 2.0)
                + 2.0 * (features["center"] > max(left, right) * 1.7)
                + 1.0 * (aspect_ratio < 0.52)
                + 1.0 * (features["center"] > max(left, right) * 2.0)
                - 2.0 * fill
                - 1.0 * (vc50 > 1.5)
            )
        if digit == 2:
            return (
                6.0 * (holes == 0.0 or largest_hole < 0.03)
                + 2.5 * (top > middle * 1.02)
                + 3.0 * (bottom > middle * 1.15)
                + 2.5 * (features["lower_left"] > features["lower_right"] * 1.05)
                + 2.0 * (hr50 <= 1.2)
                + 1.5 * (vc50 >= 1.8)
                + 1.5 * (row_left_50 > 0.4)
                + 1.5 * (row_width_80 > 0.5)
                + 1.5 * (row_left_20 < 0.38)
                + 2.0 * sweeping_two
                + 1.5 * (main_diag_runs >= anti_diag_runs + 0.2)
                - 1.5 * (row_left_50 < 0.3)
                - 2.0 * (holes == 0.0)
                - 1.5 * (row_width_50 > 0.38)
            )
        if digit == 3:
            return (
                6.0 * (holes == 0.0)
                + 3.0 * (right > left * 1.35)
                + 2.0 * (top > 0.22)
                + 2.0 * (bottom > 0.22)
                + 2.0 * (hr20 <= 1.6 and hr80 <= 1.6)
                + 1.5 * (vc50 >= 2.3)
                + 2.0 * (vc50 >= 2.5)
                + 2.0 * (row_width_50 > 0.35)
                + 1.5 * (row_left_80 > 0.16)
                - 2.0 * (row_left_50 < 0.24)
                - 2.0 * (left > right * 0.95)
                - 1.5 * (features["lower_left"] > features["lower_right"] * 1.1)
            )
        if digit == 4:
            return (
                6.0 * (holes <= 1.0)
                + 3.0 * (middle > top * 1.45)
                + 3.0 * (middle > bottom * 2.0)
                + 2.5 * (hr20 >= 1.6)
                + 2.5 * (hr35 >= 1.7)
                + 2.0 * (hr80 <= 1.2)
                + 1.5 * (right >= left * 0.9)
                + 1.0 * (vc50 <= 1.4)
                + 1.0 * (middle > bottom * 2.2)
                - 2.0 * broken_loop_nine
                - 0.5 * (col_top_50 < 0.14)
            )
        if digit == 5:
            return (
                6.0 * (holes == 0.0)
                + 3.0 * (left > right * 1.15)
                + 2.0 * (top > middle * 1.02)
                + 2.0 * (bottom > middle * 0.95)
                + 2.0 * (features["lower_right"] > features["lower_left"] * 0.75)
                + 1.5 * (vc50 >= 2.2)
                + 2.0 * (features["upper_left"] > features["upper_right"])
                + 2.0 * (row_left_50 < 0.3)
                + 1.5 * (row_width_80 < 0.5)
                + 0.5 * (col_top_80 < 0.12)
                + 1.0 * (left > right * 1.2)
                + 1.5 * (anti_diag_runs >= main_diag_runs + 0.5)
                - 2.0 * broken_loop_zero
                - 1.0 * (bottom > top * 1.2)
                - 1.0 * (main_diag_runs >= anti_diag_runs + 0.4)
                - 1.0 * (row_left_50 > 0.32)
                - 1.5 * (main_diag_runs > anti_diag_runs + 0.2)
            )
        if digit == 6:
            return (
                8.0 * (holes == 1.0)
                + 3.0 * broken_loop_six
                + 3.5
                * (
                    holes == 0.0
                    and top < 0.25
                    and bottom > middle * 1.1
                    and col_top_50 > 0.14
                    and diag_balance > 0.45
                )
                + 3.5 * (hole_y > 0.56)
                + 2.5 * (bottom > top * 1.35)
                + 2.0 * (left >= right * 1.05)
                + 2.0 * (hr65 >= 1.6)
                + 1.5 * (hr20 <= 1.2)
                + 1.5 * (largest_hole > 0.015)
                + 1.0 * (col_top_50 > 0.12)
                + 1.0 * (middle > top * 1.35)
                + 1.0 * (diag_balance > 0.6)
                + 2.5 * (main_diag_runs > anti_diag_runs + 0.35)
                + 1.0 * (row_left_50 < 0.18)
                + 0.75 * (bottom > top * 1.6)
                - 2.5 * (top > 0.26)
                - 2.0 * (row_left_50 > 0.28)
                - 2.5 * (anti_diag_runs > main_diag_runs + 0.2)
                - 1.5 * (diag_balance < 0.2)
            )
        if digit == 7:
            return (
                6.0 * (holes == 0.0)
                + 3.0 * (top > middle * 1.3)
                + 3.0 * (top > bottom * 1.7)
                + 2.5 * (hr80 <= 1.1)
                + 2.0 * (hr65 <= 1.2)
                + 1.5 * (features["upper_right"] > features["upper_left"])
                + 1.0 * (col_top_50 < 0.091)
                + 1.0 * (row_width_80 < 0.24)
                + 1.0 * (bottom < top * 0.75)
                - 2.0 * (row_width_50 > 0.42)
                - 1.5 * (vc50 > 2.15)
                - 1.5 * (repaired_holes > 0.45)
                - 1.0 * (col_top_50 > 0.12)
            )
        if digit == 8:
            return (
                9.0 * (holes >= 2.0)
                + 5.0 * (holes == 1.0)
                + 3.0 * (repaired_holes >= 2.0)
                + 1.5 * (holes == 1.0 and left >= right * 1.05 and hr50 <= 1.2)
                + 1.5 * (holes == 1.0 and repaired_holes >= 2.0)
                + 2.0 * features["vertical_symmetry"]
                + 2.0 * features["horizontal_symmetry"]
                + 2.0 * (hr20 >= 1.5)
                + 2.0 * (hr80 >= 1.5)
                + 1.5 * (vc50 >= 2.2)
                + 1.5 * (largest_hole > 0.025)
                + 1.5 * (0.35 <= hole_y <= 0.65)
                + 1.5 * (row_width_80 < 0.52)
                + 1.5 * (features["lower_left"] > features["lower_right"] * 1.08)
                - 1.5 * (holes == 1.0 and largest_hole > 0.1)
            )
        if digit == 9:
            return (
                8.0 * (holes == 1.0)
                + 3.5 * broken_loop_nine
                + 3.0 * slashed_nine
                + 3.5 * (hole_y < 0.42)
                + 2.5 * (top > bottom * 1.35)
                + 2.0 * (right >= left * 1.05)
                + 2.0 * (hr20 >= 1.6)
                + 1.5 * (hr80 <= 1.2)
                + 1.5 * (row_left_80 > 0.35)
                + 1.5 * (row_width_80 < 0.32)
                + 1.0 * (largest_hole > 0.03)
                + 1.5 * (row_width_50 > 0.4)
                + 1.5 * (repaired_holes > 0.45)
                + 0.5 * (col_top_50 < 0.11)
                - 1.0 * (vc50 <= 1.6)
                - 1.0 * (col_top_50 > 0.18)
            )
        raise ValueError(f"Unexpected digit: {digit}")

    def classify(self, image: np.ndarray) -> int:
        features = extract_features(image)
        scores = [self.score_digit(features, digit) for digit in range(10)]
        order = sorted(range(10), key=lambda digit: scores[digit], reverse=True)
        best_digit = order[0]
        second_digit = order[1]
        return self.break_tie(features, scores, best_digit, second_digit)

    def break_tie(
        self,
        features: dict[str, float],
        scores: list[float],
        best_digit: int,
        second_digit: int,
    ) -> int:
        pair = {best_digit, second_digit}
        if pair == {3, 5} and abs(scores[3] - scores[5]) <= 2.0:
            if (
                features["upper_left"] > features["upper_right"] * 1.08
                and features["row_left_50"] < 0.24
            ):
                return 5
            if (
                features["left"] > features["right"] * 1.08
                and features["anti_diag_runs"] > features["main_diag_runs"] + 0.25
            ):
                return 5
            if features["right"] > features["left"] * 1.1 and features["row_left_50"] > 0.22:
                return 3
            if features["anti_diag_runs"] > features["main_diag_runs"] + 0.2 and features["row_left_50"] < 0.26:
                return 5
            if features["row_left_50"] < 0.14:
                return 5
            return 3
        if pair == {2, 8} and abs(scores[2] - scores[8]) <= 2.5:
            if features["row_left_50"] > 0.45 and (
                features["top"] < 0.34 or features["hole_y"] > 0.62
            ):
                return 2
            return 8
        if pair == {2, 3} and abs(scores[2] - scores[3]) <= 3.0:
            if features["lower_left"] >= 0.415:
                return 2
            return 3
        if pair == {2, 6} and abs(scores[2] - scores[6]) <= 3.0:
            if features["row_left_50"] >= 0.357:
                return 2
            return 6
        if pair == {4, 9} and abs(scores[4] - scores[9]) <= 3.5:
            if features["col_top_50"] >= 0.182:
                return 4
            return 9
        if pair == {5, 6} and abs(scores[5] - scores[6]) <= 2.5:
            if features["repaired_holes"] == 1.0 and features["repaired_largest_hole"] > 0.02:
                return 6
            if (
                features["bottom"] > features["top"] * 1.25
                and features["hr80"] >= 1.8
                and features["main_diag_runs"] >= features["anti_diag_runs"]
            ):
                return 6
            if features["row_left_20"] < 0.25:
                return 5
            return 6
        if pair == {6, 8} and abs(scores[6] - scores[8]) <= 3.0:
            if features["hr35"] <= 1.0:
                return 6
            return 8
        if pair == {7, 9} and abs(scores[7] - scores[9]) <= 3.0:
            if (
                features["top"] > features["bottom"] * 1.45
                and features["row_left_50"] > 0.45
                and features["row_width_50"] < 0.24
                and features["vc50"] >= 2.0
            ):
                return 9
            return 7
        if pair == {8, 9} and abs(scores[8] - scores[9]) <= 3.0:
            if features["holes"] >= 2.0 or features["repaired_holes"] >= 2.0:
                return 8
            if features["holes"] == 1.0 and features["left"] >= features["right"] * 1.05:
                return 8
            if (
                features["holes"] == 1.0
                and features["hr80"] >= 1.2
                and features["row_left_80"] < 0.24
                and features["repaired_largest_hole"] > 0.035
                and features["lower_left"] > features["lower_right"] * 1.35
            ):
                return 8
            return 9
        return best_digit


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
