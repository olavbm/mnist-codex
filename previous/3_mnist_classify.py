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
    threshold = max(32, int(image.mean() * 0.75))
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
            "largest_hole": 0.0,
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
        }

    ys, xs = np.where(mask)
    holes = find_holes(mask)
    features = {
        "aspect_ratio": w / h,
        "fill": float(mask.mean()),
        "center_y": float(ys.mean() / h),
        "center_x": float(xs.mean() / w),
        "holes": float(len(holes)),
        "largest_hole": float(max((hole[0] for hole in holes), default=0.0)),
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
    return features


class DigitClassifier:
    def score_digit(self, features: dict[str, float], digit: int) -> float:
        holes = features["holes"]
        largest_hole = features["largest_hole"]
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

        if digit == 0:
            return (
                8.0 * (holes == 1.0)
                + 3.5 * features["vertical_symmetry"]
                + 2.5 * (1.6 <= hr50 <= 2.4)
                + 2.0 * (1.6 <= vc50 <= 2.4)
                + 1.5 * (largest_hole > 0.1)
                + 2.0 * (0.42 <= hole_y <= 0.6)
                + 1.5 * (abs(top - bottom) < 0.12)
                - 2.5 * (hole_y > 0.62)
                - 2.5 * (hole_y < 0.4)
            )
        if digit == 1:
            return (
                8.0 * (holes == 0.0)
                + 4.0 * (aspect_ratio < 0.58)
                + 3.0 * (hr20 <= 1.1 and hr50 <= 1.1 and hr80 <= 1.1)
                + 2.5 * (vc50 <= 1.2)
                + 2.0 * (features["center"] > max(left, right) * 1.7)
                - 2.0 * fill
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
                - 1.5 * (row_left_50 < 0.3)
            )
        if digit == 3:
            return (
                6.0 * (holes == 0.0)
                + 3.0 * (right > left * 1.35)
                + 2.0 * (top > 0.22)
                + 2.0 * (bottom > 0.22)
                + 2.0 * (hr20 <= 1.6 and hr80 <= 1.6)
                + 1.5 * (vc50 >= 2.3)
                + 2.0 * (row_width_50 > 0.35)
                + 1.5 * (row_left_80 > 0.16)
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
            )
        if digit == 6:
            return (
                8.0 * (holes == 1.0)
                + 3.5 * (hole_y > 0.56)
                + 2.5 * (bottom > top * 1.35)
                + 2.0 * (left >= right * 1.05)
                + 2.0 * (hr65 >= 1.6)
                + 1.5 * (hr20 <= 1.2)
                + 1.5 * (largest_hole > 0.015)
                - 1.0 * (top > 0.3)
            )
        if digit == 7:
            return (
                6.0 * (holes == 0.0)
                + 3.0 * (top > middle * 1.3)
                + 3.0 * (top > bottom * 1.7)
                + 2.5 * (hr80 <= 1.1)
                + 2.0 * (hr65 <= 1.2)
                + 1.5 * (features["upper_right"] > features["upper_left"])
            )
        if digit == 8:
            return (
                9.0 * (holes >= 2.0)
                + 5.0 * (holes == 1.0)
                + 2.0 * features["vertical_symmetry"]
                + 2.0 * features["horizontal_symmetry"]
                + 2.0 * (hr20 >= 1.5)
                + 2.0 * (hr80 >= 1.5)
                + 1.5 * (vc50 >= 2.2)
                + 1.5 * (largest_hole > 0.025)
                + 1.5 * (0.35 <= hole_y <= 0.65)
                + 1.5 * (row_width_80 < 0.52)
            )
        if digit == 9:
            return (
                8.0 * (holes == 1.0)
                + 3.5 * (hole_y < 0.42)
                + 2.5 * (top > bottom * 1.35)
                + 2.0 * (right >= left * 1.05)
                + 2.0 * (hr20 >= 1.6)
                + 1.5 * (hr80 <= 1.2)
                + 1.5 * (row_left_80 > 0.35)
                + 1.5 * (row_width_80 < 0.32)
                + 1.0 * (largest_hole > 0.03)
            )
        raise ValueError(f"Unexpected digit: {digit}")

    def classify(self, image: np.ndarray) -> int:
        features = extract_features(image)
        best_digit = 0
        best_score = self.score_digit(features, 0)
        for digit in range(1, 10):
            score = self.score_digit(features, digit)
            if score > best_score:
                best_score = score
                best_digit = digit
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
