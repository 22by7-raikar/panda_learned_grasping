import numpy as np
import pytest

from grasp_sim.renderer import (
    Intrinsics, Primitive, Table, intersect_box, intersect_cylinder, object_footprint_contains,
    pose_xyz_yaw, primitives_from_scene, render)


def look_down(x=0.5, y=0.0, z=0.5):
    """Camera above (x, y) looking straight down; image x = world x, image y = world -y."""
    pose = np.eye(4)
    pose[:3, :3] = [[1, 0, 0], [0, -1, 0], [0, 0, -1]]
    pose[:3, 3] = [x, y, z]
    return pose


def centre_depth(frame, intr):
    return frame.depth[int(intr.cy), int(intr.cx)]


def test_empty_table_depth_equals_camera_height():
    intr = Intrinsics()
    frame = render(look_down(z=0.5), [], intr=intr, depth_noise=0.0, depth_noise_quadratic=0.0, dropout=0.0,
                   rng=np.random.default_rng(0))
    assert centre_depth(frame, intr) == pytest.approx(0.5, abs=1e-3)
    # Points are in the optical frame: all in front of the camera.
    assert (frame.points[:, 2] > 0).all()


def test_box_and_cylinder_tops():
    intr = Intrinsics()
    for prim, top in [
        (Primitive("box", pose_xyz_yaw(0.5, 0.0, 0.0, 0.3), (0.05, 0.08, 0.12)), 0.12),
        (Primitive("cylinder", pose_xyz_yaw(0.5, 0.0), (0.03, 0.09)), 0.09),
    ]:
        frame = render(look_down(z=0.5), [prim], intr=intr, depth_noise=0.0, depth_noise_quadratic=0.0, dropout=0.0)
        assert centre_depth(frame, intr) == pytest.approx(0.5 - top, abs=1e-3)
        assert tuple(frame.color[int(intr.cy), int(intr.cx)]) == prim.color


def test_box_side_hit_from_oblique_ray():
    origins = np.array([[0.0, 0.0, 0.05]])
    dirs = np.array([[1.0, 0.0, 0.0]])
    t = intersect_box(pose_xyz_yaw(0.5, 0.0), (0.1, 0.1, 0.1), origins, dirs)
    assert t[0] == pytest.approx(0.45)
    t = intersect_cylinder(pose_xyz_yaw(0.5, 0.0), (0.05, 0.1), origins, dirs)
    assert t[0] == pytest.approx(0.45)
    # Above the cylinder: miss.
    t = intersect_cylinder(pose_xyz_yaw(0.5, 0.0), (0.05, 0.1), np.array([[0.0, 0.0, 0.2]]), dirs)
    assert np.isinf(t[0])


def test_points_back_project_to_world():
    intr = Intrinsics()
    cam = look_down(0.45, 0.05, 0.6)
    frame = render(cam, [], table=Table(height=0.0), intr=intr, depth_noise=0.0, depth_noise_quadratic=0.0, dropout=0.0)
    world = frame.points @ cam[:3, :3].T + cam[:3, 3]
    assert np.abs(world[:, 2]).max() < 1e-6


def test_compound_object_and_footprint():
    scene = {"objects": {"clamp": {"parts": [
        {"type": "box", "size": [0.14, 0.025, 0.025], "offset": [0.07, 0.0, 0.0]},
        {"type": "box", "size": [0.14, 0.025, 0.025], "offset": [0.0, 0.07, 1.5708]},
    ]}}}
    pose = pose_xyz_yaw(0.4, 0.0)
    prims = primitives_from_scene(scene, {"clamp": pose})
    assert len(prims) == 2
    assert object_footprint_contains(scene["objects"]["clamp"], pose, np.array([0.45, 0.0, 0.01]))
    assert not object_footprint_contains(scene["objects"]["clamp"], pose, np.array([0.45, 0.05, 0.01]))


def test_ycb_library_objects_render():
    from grasp_sim.scene import load_ycb_library, parse_scene
    library = load_ycb_library()
    assert "006_mustard_bottle" in library
    scene = {"objects": {name: {"ycb": name, "pose": [0.5, 0.0, 0.3]} for name in library}}
    for name in library:
        single = {"objects": {name: dict(scene["objects"][name])}}
        sc, table, poses = parse_scene(single, library)
        frame = render(look_down(z=0.6), primitives_from_scene(sc, poses), table, Intrinsics(),
                       depth_noise=0.0, depth_noise_quadratic=0.0, dropout=0.0)
        # Something sticks up above the table under the camera.
        assert np.nanmin(frame.depth) < 0.6 - 0.02, name
