"""Oriented grasp rectangles in image space (Cornell convention) and the rectangle metric.

A rectangle is (cx, cy, w, h, theta) in pixels, image x right / y down:
  w      jaw opening: distance between the two gripper plates
  h      plate size: how much of the object each finger covers
  theta  direction of the opening (the w axis) from the image x axis, in [-pi/2, pi/2)

Following Chu, Xu & Vela (2018), orientation is predicted as a class: [-pi/2, pi/2) is
split into NUM_ANGLE_BINS bins and the detector's box regressor predicts the
*unrotated* rectangle (cx +- w/2, cy +- h/2) for each angle class.
"""

from dataclasses import dataclass

import numpy as np

NUM_ANGLE_BINS = 19
BIN_WIDTH = np.pi / NUM_ANGLE_BINS


def wrap_angle(theta: float) -> float:
    """Map to [-pi/2, pi/2): a parallel gripper is symmetric under a 180 deg turn."""
    return (theta + np.pi / 2) % np.pi - np.pi / 2


def angle_to_bin(theta: float) -> int:
    return int(np.clip(np.floor((wrap_angle(theta) + np.pi / 2) / BIN_WIDTH), 0, NUM_ANGLE_BINS - 1))


def bin_to_angle(k: int) -> float:
    return -np.pi / 2 + (k + 0.5) * BIN_WIDTH


def angle_difference(a: float, b: float) -> float:
    """Smallest difference between two grasp angles, accounting for the 180 deg symmetry."""
    d = abs(wrap_angle(a) - wrap_angle(b))
    return min(d, np.pi - d)


@dataclass
class GraspRect:
    cx: float
    cy: float
    w: float
    h: float
    theta: float
    score: float = 1.0

    def corners(self) -> np.ndarray:
        """4x2 (x, y) corners, counter-clockwise in a y-up sense."""
        c, s = np.cos(self.theta), np.sin(self.theta)
        u = np.array([c, s]) * self.w / 2  # along the opening
        v = np.array([-s, c]) * self.h / 2  # along the plates
        ctr = np.array([self.cx, self.cy])
        return np.stack([ctr - u - v, ctr + u - v, ctr + u + v, ctr - u + v])

    @classmethod
    def from_cornell(cls, pts: np.ndarray) -> "GraspRect":
        """From 4 Cornell vertices (x, y). Edge 0-1 runs along a gripper plate, so the
        opening direction is edge 1-2."""
        pts = np.asarray(pts, dtype=float)
        center = pts.mean(axis=0)
        plate = pts[1] - pts[0]
        opening = pts[2] - pts[1]
        theta = wrap_angle(np.arctan2(opening[1], opening[0]))
        return cls(center[0], center[1], float(np.linalg.norm(opening)), float(np.linalg.norm(plate)), theta)

    def canonical_box(self) -> np.ndarray:
        """Unrotated (x1, y1, x2, y2) box regressed by the detector for this angle class."""
        return np.array([self.cx - self.w / 2, self.cy - self.h / 2, self.cx + self.w / 2, self.cy + self.h / 2])

    @classmethod
    def from_detection(cls, box, label: int, score: float = 1.0) -> "GraspRect":
        """Inverse of (canonical_box, angle_to_bin + 1); label 0 is background."""
        x1, y1, x2, y2 = (float(v) for v in box)
        return cls((x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1, bin_to_angle(int(label) - 1), float(score))

    def angle_bin(self) -> int:
        return angle_to_bin(self.theta)

    def scaled(self, sx: float, sy: float = None) -> "GraspRect":
        sy = sx if sy is None else sy
        return GraspRect(self.cx * sx, self.cy * sy, self.w * sx, self.h * sy, self.theta, self.score)

    def shifted(self, dx: float, dy: float) -> "GraspRect":
        return GraspRect(self.cx + dx, self.cy + dy, self.w, self.h, self.theta, self.score)

    def flipped_x(self, image_width: int) -> "GraspRect":
        return GraspRect(image_width - 1 - self.cx, self.cy, self.w, self.h, wrap_angle(-self.theta), self.score)


def _cross(a: np.ndarray, b: np.ndarray) -> float:
    return float(a[0] * b[1] - a[1] * b[0])


def polygon_area(p: np.ndarray) -> float:
    x, y = p[:, 0], p[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def _clip(subject: np.ndarray, clipper: np.ndarray) -> np.ndarray:
    """Sutherland-Hodgman clipping of a polygon by a convex polygon."""
    # Orientation of the clipper decides which side is "inside".
    sign = np.sign(_cross(clipper[1] - clipper[0], clipper[2] - clipper[1])) or 1.0
    out = list(subject)
    for i in range(len(clipper)):
        a, b = clipper[i], clipper[(i + 1) % len(clipper)]
        inp, out = out, []
        if not inp:
            break
        for j in range(len(inp)):
            p, q = inp[j], inp[(j + 1) % len(inp)]
            p_in = sign * _cross(b - a, p - a) >= 0
            q_in = sign * _cross(b - a, q - a) >= 0
            if p_in:
                out.append(p)
            if p_in != q_in:
                d = q - p
                denom = _cross(b - a, d)
                if abs(denom) > 1e-12:
                    t = _cross(b - a, a - p) / denom
                    out.append(p + t * d)
    return np.array(out) if out else np.zeros((0, 2))


def rect_iou(a: GraspRect, b: GraspRect) -> float:
    pa, pb = a.corners(), b.corners()
    inter = _clip(pa, pb)
    inter_area = polygon_area(inter) if len(inter) >= 3 else 0.0
    union = polygon_area(pa) + polygon_area(pb) - inter_area
    return inter_area / union if union > 0 else 0.0


def rectangle_metric(pred: GraspRect, ground_truth: list, max_angle=np.deg2rad(30), min_iou=0.25) -> bool:
    """Cornell "rectangle metric": correct if, for some ground-truth rectangle, the angle is
    within 30 deg and the Jaccard index exceeds 25%."""
    return any(angle_difference(pred.theta, g.theta) < max_angle and rect_iou(pred, g) > min_iou
               for g in ground_truth)
