import numpy as np
import pytest

from grasp_learning.projection import ProjectionParams, rect_mask, rect_to_grasp, rotation_to_quaternion
from grasp_learning.rectangle import GraspRect

K = np.array([[300.0, 0, 160.0], [0, 300.0, 120.0], [0, 0, 1]])


def camera_above(z=0.5):
    """Optical frame looking straight down; image x = base x, image y = base -y."""
    t = np.eye(4)
    t[:3, :3] = [[1, 0, 0], [0, -1, 0], [0, 0, -1]]
    t[:3, 3] = [0.5, 0.0, z]
    return t


def box_depth(top=0.42, table=0.5, x0=140, x1=180, y0=100, y1=140):
    d = np.full((240, 320), table)
    d[y0:y1, x0:x1] = top
    return d


def test_mask():
    m = rect_mask(GraspRect(50, 40, 20, 10, 0.0), (100, 100))
    assert m.sum() == pytest.approx(21 * 11, abs=25)
    assert m[40, 50] and not m[40, 70]


def test_top_down_grasp_on_box():
    depth = box_depth()  # 8 cm tall box under the image centre
    rect = GraspRect(160, 120, 60, 20, np.deg2rad(90))  # opening along image y
    g = rect_to_grasp(rect, depth, K, camera_above())
    assert g is not None
    # Approach straight down, closing along base y (image y = base -y).
    assert g.pose[:3, 2] == pytest.approx([0, 0, -1], abs=1e-9)
    assert abs(g.pose[1, 1]) == pytest.approx(1.0, abs=1e-9)
    assert g.pose[:3, 0].dot([1, 0, 0]) > 0  # x-axis hint respected
    # Fingertips 4 cm below the top (top at z = 0.08), above the table.
    assert g.pose[2, 3] == pytest.approx(0.08 - 0.04, abs=1e-6)
    assert g.width == pytest.approx(60 * 0.42 / 300)
    assert g.object_dimensions[2] == pytest.approx(0.08)
    R = g.pose[:3, :3]
    assert R.T @ R == pytest.approx(np.eye(3))
    assert np.linalg.det(R) == pytest.approx(1.0)


def test_shallow_object_limits_depth():
    depth = box_depth(top=0.48)  # 2 cm tall
    g = rect_to_grasp(GraspRect(160, 120, 60, 20, 0.0), depth, K, camera_above(),
                      ProjectionParams(table_clearance=0.004))
    assert g.pose[2, 3] == pytest.approx(0.004, abs=1e-6)


def test_flat_or_invalid_regions_are_rejected():
    flat = np.full((240, 320), 0.5)
    assert rect_to_grasp(GraspRect(160, 120, 60, 20, 0.0), flat, K, camera_above()) is None
    holes = np.full((240, 320), np.nan)
    assert rect_to_grasp(GraspRect(160, 120, 60, 20, 0.0), holes, K, camera_above()) is None


def quat_to_matrix(x, y, z, w):
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def test_rotation_to_quaternion_round_trip():
    rng = np.random.default_rng(0)
    rotations = [np.diag([1.0, -1.0, -1.0]), np.diag([-1.0, 1.0, -1.0]), np.diag([-1.0, -1.0, 1.0]), np.eye(3)]
    for _ in range(50):
        q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
        rotations.append(q * np.sign(np.linalg.det(q)))
    for r in rotations:
        q = rotation_to_quaternion(r)
        assert np.linalg.norm(q) == pytest.approx(1.0)
        assert quat_to_matrix(*q) == pytest.approx(r, abs=1e-9)
