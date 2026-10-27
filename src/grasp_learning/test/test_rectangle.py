import numpy as np
import pytest

from grasp_learning.rectangle import (
    NUM_ANGLE_BINS, GraspRect, angle_difference, angle_to_bin, bin_to_angle, rect_iou, rectangle_metric, wrap_angle)


def test_angle_bins_round_trip():
    for k in range(NUM_ANGLE_BINS):
        assert angle_to_bin(bin_to_angle(k)) == k
    assert angle_to_bin(np.pi / 2 - 1e-6) == NUM_ANGLE_BINS - 1
    assert angle_to_bin(-np.pi / 2) == 0
    # 180 deg symmetry
    assert angle_to_bin(0.3) == angle_to_bin(0.3 + np.pi)


def test_angle_difference_is_symmetric_mod_pi():
    assert angle_difference(np.deg2rad(85), np.deg2rad(-85)) == pytest.approx(np.deg2rad(10))
    assert angle_difference(0.2, 0.2 + np.pi) == pytest.approx(0.0, abs=1e-12)
    assert wrap_angle(np.pi / 2) == pytest.approx(-np.pi / 2)


def test_cornell_vertices():
    # Plate edge (0-1) vertical, opening along +x: 40 px opening, 20 px plates.
    pts = np.array([[80, 90], [80, 110], [120, 110], [120, 90]], dtype=float)
    r = GraspRect.from_cornell(pts)
    assert (r.cx, r.cy) == pytest.approx((100, 100))
    assert r.w == pytest.approx(40)
    assert r.h == pytest.approx(20)
    assert angle_difference(r.theta, 0.0) < 1e-9
    # corners() reproduces the same rectangle (as a set)
    assert sorted(map(tuple, np.round(r.corners()))) == sorted(map(tuple, pts))


def test_detection_round_trip():
    r = GraspRect(150.0, 80.0, 50.0, 18.0, np.deg2rad(37))
    back = GraspRect.from_detection(r.canonical_box(), r.angle_bin() + 1)
    assert (back.cx, back.cy, back.w, back.h) == pytest.approx((r.cx, r.cy, r.w, r.h))
    assert angle_difference(back.theta, r.theta) <= np.pi / NUM_ANGLE_BINS / 2 + 1e-9


def test_iou():
    a = GraspRect(0, 0, 10, 10, 0)
    assert rect_iou(a, a) == pytest.approx(1.0)
    assert rect_iou(a, GraspRect(5, 0, 10, 10, 0)) == pytest.approx(1 / 3)
    assert rect_iou(a, GraspRect(100, 0, 10, 10, 0)) == 0.0
    # Square rotated 45 deg inside itself: intersection = octagon
    rot = GraspRect(0, 0, 10, 10, np.pi / 4)
    octagon = 100 * 2 * (np.sqrt(2) - 1)
    assert rect_iou(a, rot) == pytest.approx(octagon / (200 - octagon), rel=1e-6)


def test_rectangle_metric():
    gt = [GraspRect(100, 100, 40, 20, 0.0)]
    assert rectangle_metric(GraspRect(103, 100, 40, 20, np.deg2rad(20)), gt)
    assert not rectangle_metric(GraspRect(100, 100, 40, 20, np.deg2rad(40)), gt)  # angle
    assert not rectangle_metric(GraspRect(140, 100, 40, 20, 0.0), gt)  # IoU


def test_flip():
    r = GraspRect(10, 20, 30, 10, np.deg2rad(30))
    f = r.flipped_x(100)
    assert f.cx == 89 and f.cy == 20
    assert angle_difference(f.theta, np.deg2rad(-30)) < 1e-9
