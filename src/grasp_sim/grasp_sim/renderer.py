"""Tiny ray-casting RGB-D renderer for tabletop scenes of boxes and cylinders.

Stands in for the wrist RealSense so the full pipeline (perception -> planning ->
execution on MoveIt mock hardware) runs without Gazebo or a GPU. Depth follows the
camera's z axis like a real depth sensor, with noise that grows with distance and
random dropouts.
"""

from dataclasses import dataclass, field

import numpy as np


@dataclass
class Intrinsics:
    width: int = 320
    height: int = 240
    fx: float = 307.5
    fy: float = 307.5
    cx: float = 159.5
    cy: float = 119.5

    @property
    def k(self) -> np.ndarray:
        return np.array([[self.fx, 0, self.cx], [0, self.fy, self.cy], [0, 0, 1]], dtype=float)


@dataclass
class Primitive:
    """A box or upright cylinder. pose is the 4x4 world pose of its bottom centre."""

    kind: str  # "box" or "cylinder"
    pose: np.ndarray
    size: tuple = (0.04, 0.04, 0.04)  # box: (sx, sy, sz); cylinder: (radius, height)
    color: tuple = (200, 60, 60)


@dataclass
class Table:
    height: float = 0.0
    x_range: tuple = (0.15, 0.9)
    y_range: tuple = (-0.55, 0.55)
    color: tuple = (150, 115, 80)


@dataclass
class Frame:
    depth: np.ndarray  # (H, W) float32 metres along the optical axis, NaN = no return
    color: np.ndarray  # (H, W, 3) uint8
    points: np.ndarray = field(default=None)  # (N, 3) in the camera optical frame
    point_colors: np.ndarray = field(default=None)  # (N, 3) uint8


def pose_xyz_yaw(x: float, y: float, z: float = 0.0, yaw: float = 0.0) -> np.ndarray:
    c, s = np.cos(yaw), np.sin(yaw)
    t = np.eye(4)
    t[:3, :3] = [[c, -s, 0], [s, c, 0], [0, 0, 1]]
    t[:3, 3] = [x, y, z]
    return t


def camera_rays(intr: Intrinsics) -> np.ndarray:
    """Unnormalised ray directions with z = 1 in the optical frame, shape (H*W, 3)."""
    u, v = np.meshgrid(np.arange(intr.width), np.arange(intr.height))
    return np.stack([(u - intr.cx) / intr.fx, (v - intr.cy) / intr.fy, np.ones_like(u, dtype=float)], -1).reshape(-1, 3)


def _to_local(pose: np.ndarray, origins: np.ndarray, dirs: np.ndarray):
    r, t = pose[:3, :3], pose[:3, 3]
    return (origins - t) @ r, dirs @ r


def intersect_box(pose, size, origins, dirs) -> np.ndarray:
    """Slab test. Returns ray parameter t (inf for misses)."""
    o, d = _to_local(pose, origins, dirs)
    half = np.array([size[0] / 2, size[1] / 2, size[2] / 2])
    lo = np.array([-half[0], -half[1], 0.0])
    hi = np.array([half[0], half[1], size[2]])
    with np.errstate(divide="ignore", invalid="ignore"):
        inv = 1.0 / d
        t1 = (lo - o) * inv
        t2 = (hi - o) * inv
    t_near = np.nanmax(np.minimum(t1, t2), axis=1)
    t_far = np.nanmin(np.maximum(t1, t2), axis=1)
    hit = (t_near <= t_far) & (t_far > 0)
    return np.where(hit, np.where(t_near > 0, t_near, t_far), np.inf)


def intersect_cylinder(pose, size, origins, dirs) -> np.ndarray:
    """Upright finite cylinder (side + top cap). Returns ray parameter t."""
    radius, height = size
    o, d = _to_local(pose, origins, dirs)
    t = np.full(len(o), np.inf)

    a = d[:, 0] ** 2 + d[:, 1] ** 2
    b = 2 * (o[:, 0] * d[:, 0] + o[:, 1] * d[:, 1])
    c = o[:, 0] ** 2 + o[:, 1] ** 2 - radius ** 2
    disc = b * b - 4 * a * c
    ok = (disc >= 0) & (a > 1e-12)
    with np.errstate(invalid="ignore", divide="ignore"):
        t_side = (-b - np.sqrt(np.where(ok, disc, 0))) / (2 * a)
    z = o[:, 2] + t_side * d[:, 2]
    side = ok & (t_side > 0) & (z >= 0) & (z <= height)
    t = np.where(side, t_side, t)

    with np.errstate(divide="ignore", invalid="ignore"):
        t_cap = (height - o[:, 2]) / d[:, 2]
        p = o + t_cap[:, None] * d
    cap = (t_cap > 0) & (p[:, 0] ** 2 + p[:, 1] ** 2 <= radius ** 2)
    return np.where(cap & (t_cap < t), t_cap, t)


def render(
    cam_pose: np.ndarray,
    primitives: list,
    table: Table = Table(),
    intr: Intrinsics = Intrinsics(),
    depth_noise: float = 0.001,
    depth_noise_quadratic: float = 0.002,
    dropout: float = 0.01,
    rng: np.random.Generator = None,
    background=(110, 110, 110),
) -> Frame:
    """Render depth and colour seen from cam_pose (world <- optical frame, 4x4)."""
    rng = rng or np.random.default_rng()
    rays_cam = camera_rays(intr)
    dirs = rays_cam @ cam_pose[:3, :3].T
    origins = np.broadcast_to(cam_pose[:3, 3], dirs.shape)

    best = np.full(len(dirs), np.inf)
    color = np.tile(np.array(background, dtype=np.uint8), (len(dirs), 1))

    with np.errstate(divide="ignore", invalid="ignore"):
        t_table = (table.height - origins[:, 2]) / dirs[:, 2]
    p = origins + t_table[:, None] * dirs
    on_table = (t_table > 0) & (p[:, 0] >= table.x_range[0]) & (p[:, 0] <= table.x_range[1]) \
        & (p[:, 1] >= table.y_range[0]) & (p[:, 1] <= table.y_range[1])
    best = np.where(on_table, t_table, best)
    grain = (10 * np.sin(p[:, 0] * 180) + 6 * np.sin(p[:, 1] * 37)).astype(int)[:, None]
    table_color = np.clip(np.array(table.color)[None, :] + np.nan_to_num(grain), 0, 255).astype(np.uint8)
    color = np.where(on_table[:, None], table_color, color)

    for prim in primitives:
        fn = intersect_box if prim.kind == "box" else intersect_cylinder
        t = fn(prim.pose, prim.size, origins, dirs)
        closer = t < best
        best = np.where(closer, t, best)
        color = np.where(closer[:, None], np.array(prim.color, dtype=np.uint8)[None, :], color)

    # rays_cam has z = 1, so the ray parameter is the depth along the optical axis.
    depth = best.copy()
    hit = np.isfinite(depth)
    depth[hit] += rng.normal(0.0, 1.0, hit.sum()) * (depth_noise + depth_noise_quadratic * depth[hit] ** 2)
    drop = rng.random(len(depth)) < dropout
    depth[drop | ~hit] = np.nan

    valid = np.isfinite(depth)
    points = rays_cam[valid] * depth[valid, None]
    return Frame(
        depth=depth.reshape(intr.height, intr.width).astype(np.float32),
        color=color.reshape(intr.height, intr.width, 3),
        points=points.astype(np.float32),
        point_colors=color[valid],
    )


def primitives_from_scene(scene: dict, object_poses: dict) -> list:
    """Expand a scene description into world-frame primitives.

    scene["objects"] maps a name to {"parts": [...]} or a single part. A part has
    "type", "size" (or "radius"/"height"), optional "offset" [x, y, yaw] in the object
    frame and "color". object_poses maps the same names to 4x4 world poses.
    """
    prims = []
    for name, obj in scene["objects"].items():
        parts = obj.get("parts", [obj])
        for part in parts:
            ox, oy, oyaw = part.get("offset", [0.0, 0.0, 0.0])
            pose = object_poses[name] @ pose_xyz_yaw(ox, oy, 0.0, oyaw)
            if part["type"] == "box":
                size = tuple(part["size"])
            else:
                size = (part["radius"], part["height"])
            prims.append(Primitive(part["type"], pose, size, tuple(part.get("color", obj.get("color", (200, 60, 60))))))
    return prims


def object_footprint_contains(scene_obj: dict, pose: np.ndarray, point_world: np.ndarray) -> bool:
    """True if point_world lies inside any part of the object (used for simulated grasps)."""
    for part in scene_obj.get("parts", [scene_obj]):
        ox, oy, oyaw = part.get("offset", [0.0, 0.0, 0.0])
        local = np.linalg.inv(pose @ pose_xyz_yaw(ox, oy, 0.0, oyaw)) @ np.append(point_world, 1.0)
        if part["type"] == "box":
            sx, sy, sz = part["size"]
            if abs(local[0]) <= sx / 2 and abs(local[1]) <= sy / 2 and -0.01 <= local[2] <= sz:
                return True
        elif np.hypot(local[0], local[1]) <= part["radius"] and -0.01 <= local[2] <= part["height"]:
            return True
    return False
