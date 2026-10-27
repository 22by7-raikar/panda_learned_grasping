"""Turn an image-space grasp rectangle plus aligned depth into a 3D top-down grasp.

The detector is 2D, so its grasps are 4-DoF: position, opening direction in the image
plane and width. The approach is along the camera's optical axis, which is why the
camera looks straight down from the observation pose. (The classical EFD pipeline has
the same top-down restriction but measures the object in 3D first.)
"""

from dataclasses import dataclass

import numpy as np

from grasp_learning.rectangle import GraspRect


@dataclass
class ProjectionParams:
    max_grasp_depth: float = 0.04   # fingertips at most this far below the object top
    min_contact_depth: float = 0.006
    table_clearance: float = 0.004
    surround_scale: float = 1.8     # ring around the rectangle used to find the table
    x_axis_hint: tuple = (1.0, 0.0, 0.0)


@dataclass
class Grasp3D:
    pose: np.ndarray        # 4x4 TCP pose in the base frame (z approach, y closing)
    width: float            # [m]
    object_center: np.ndarray
    object_dimensions: np.ndarray
    contact_a: np.ndarray
    contact_b: np.ndarray


def rect_mask(rect: GraspRect, shape) -> np.ndarray:
    """Boolean mask of pixels inside the (rotated) rectangle."""
    h, w = shape
    ys, xs = np.mgrid[0:h, 0:w]
    c, s = np.cos(rect.theta), np.sin(rect.theta)
    dx, dy = xs - rect.cx, ys - rect.cy
    along = dx * c + dy * s
    across = -dx * s + dy * c
    return (np.abs(along) <= rect.w / 2) & (np.abs(across) <= rect.h / 2)


def deproject(u: float, v: float, depth: float, k: np.ndarray) -> np.ndarray:
    return np.array([(u - k[0, 2]) / k[0, 0] * depth, (v - k[1, 2]) / k[1, 1] * depth, depth])


def rect_to_grasp(rect: GraspRect, depth: np.ndarray, k: np.ndarray, base_t_cam: np.ndarray,
                  params: ProjectionParams = ProjectionParams()):
    """depth: (H, W) metres along the optical axis (0 or NaN = invalid).
    Returns a Grasp3D, or None if the depth inside the rectangle is unusable."""
    depth = np.where(np.isfinite(depth) & (depth > 0), depth, np.nan)
    inside = rect_mask(rect, depth.shape)
    d_in = depth[inside]
    d_in = d_in[np.isfinite(d_in)]
    if d_in.size < 10:
        return None

    big = GraspRect(rect.cx, rect.cy, rect.w * params.surround_scale, rect.h * params.surround_scale, rect.theta)
    ring = rect_mask(big, depth.shape) & ~inside
    d_ring = depth[ring]
    d_ring = d_ring[np.isfinite(d_ring)]

    top = float(np.percentile(d_in, 10))  # object top, robust to flying pixels
    table = float(np.percentile(d_ring, 90)) if d_ring.size else float(np.percentile(d_in, 95))
    height = table - top
    if height < params.min_contact_depth:
        return None

    reach = min(params.max_grasp_depth, height - params.table_clearance)
    tcp_depth = top + reach
    contact_depth = top + 0.5 * reach

    center_cam = deproject(rect.cx, rect.cy, tcp_depth, k)
    closing_cam = np.array([np.cos(rect.theta), np.sin(rect.theta), 0.0])
    r = base_t_cam[:3, :3]
    z = r @ np.array([0.0, 0.0, 1.0])
    y = r @ closing_cam
    y -= y.dot(z) * z
    y /= np.linalg.norm(y)
    x = np.cross(y, z)
    if x.dot(np.asarray(params.x_axis_hint)) < 0:
        x, y = -x, -y

    pose = np.eye(4)
    pose[:3, 0], pose[:3, 1], pose[:3, 2] = x, y, z
    pose[:3, 3] = base_t_cam[:3, :3] @ center_cam + base_t_cam[:3, 3]

    width = rect.w * top / k[0, 0]
    plate = rect.h * top / k[1, 1]
    half = 0.5 * rect.w * np.array([np.cos(rect.theta), np.sin(rect.theta)])
    to_base = lambda p: base_t_cam[:3, :3] @ p + base_t_cam[:3, 3]  # noqa: E731
    contact_a = to_base(deproject(rect.cx - half[0], rect.cy - half[1], contact_depth, k))
    contact_b = to_base(deproject(rect.cx + half[0], rect.cy + half[1], contact_depth, k))
    # Rough object box: the rectangle footprint and the measured height, resting on the table.
    object_center = to_base(deproject(rect.cx, rect.cy, top + 0.5 * height, k))
    dims = np.array([max(width, plate), max(width, plate), height])
    return Grasp3D(pose, width, object_center, dims, contact_a, contact_b)


def rotation_to_quaternion(r: np.ndarray):
    """3x3 rotation -> (x, y, z, w), numerically safe for any rotation."""
    tr = np.trace(r)
    if tr > 0:
        s = 2.0 * np.sqrt(tr + 1.0)
        w, x, y, z = 0.25 * s, (r[2, 1] - r[1, 2]) / s, (r[0, 2] - r[2, 0]) / s, (r[1, 0] - r[0, 1]) / s
    else:
        i = int(np.argmax(np.diag(r)))
        j, k = (i + 1) % 3, (i + 2) % 3
        s = 2.0 * np.sqrt(1.0 + r[i, i] - r[j, j] - r[k, k])
        q = np.zeros(3)
        q[i] = 0.25 * s
        q[j] = (r[j, i] + r[i, j]) / s
        q[k] = (r[k, i] + r[i, k]) / s
        w = (r[k, j] - r[j, k]) / s
        x, y, z = q
    return float(x), float(y), float(z), float(w)
