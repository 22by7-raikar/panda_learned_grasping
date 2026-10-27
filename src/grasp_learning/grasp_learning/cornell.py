"""Cornell Grasping Dataset loader.

Expected layout (the common "archive" version with depth TIFFs):

    root/01/pcd0100r.png       RGB
    root/01/pcd0100d.tiff      depth (float, metres or millimetres)
    root/01/pcd0100cpos.txt    positive rectangles, 4 lines (x y) per rectangle

Images are 640x480; objects sit near the centre, so samples are cropped to a square
around the centre (with a random shift when augmenting) and resized to image_size.
"""

import glob
import os
import random

import numpy as np
import torch
from PIL import Image

from grasp_learning.model import make_rgd
from grasp_learning.rectangle import GraspRect


def read_cornell_rects(path):
    vals = []
    with open(path) as f:
        for line in f:
            parts = line.split()
            if len(parts) != 2:
                continue
            x, y = float(parts[0]), float(parts[1])
            if not (np.isfinite(x) and np.isfinite(y)):
                vals.append(None)
                continue
            vals.append((x, y))
    rects = []
    for i in range(0, len(vals) - 3, 4):
        quad = vals[i:i + 4]
        if any(v is None for v in quad):
            continue  # a few Cornell files contain NaN vertices
        rects.append(GraspRect.from_cornell(np.array(quad)))
    return rects


def find_samples(root):
    samples = []
    for rgb in sorted(glob.glob(os.path.join(root, "**", "pcd*r.png"), recursive=True)):
        stem = rgb[:-len("r.png")]
        depth, pos = stem + "d.tiff", stem + "cpos.txt"
        if os.path.exists(depth) and os.path.exists(pos):
            samples.append((rgb, depth, pos))
    return samples


def image_wise_split(samples, fold=0, folds=5, seed=0):
    """Image-wise k-fold split (the usual Cornell protocol)."""
    idx = list(range(len(samples)))
    random.Random(seed).shuffle(idx)
    test = set(idx[fold::folds])
    return [samples[i] for i in idx if i not in test], [samples[i] for i in sorted(test)]


class CornellDataset(torch.utils.data.Dataset):
    def __init__(self, samples, image_size=320, crop=400, augment=False, depth_channel=2):
        self.samples = samples
        self.depth_channel = depth_channel
        self.image_size = image_size
        self.crop = crop
        self.augment = augment

    def __len__(self):
        return len(self.samples)

    def load(self, i):
        rgb_path, depth_path, pos_path = self.samples[i]
        rgb = np.asarray(Image.open(rgb_path).convert("RGB"))
        depth = np.asarray(Image.open(depth_path), dtype=np.float32)
        if np.nanmax(depth) > 10.0:
            depth = depth / 1000.0  # millimetres
        return rgb, depth, read_cornell_rects(pos_path)

    def __getitem__(self, i):
        rgb, depth, rects = self.load(i)
        h, w = depth.shape
        x0 = (w - self.crop) // 2
        y0 = (h - self.crop) // 2
        if self.augment:
            x0 += random.randint(-40, 40)
            y0 += random.randint(-30, 30)
            x0 = int(np.clip(x0, 0, w - self.crop))
            y0 = int(np.clip(y0, 0, h - self.crop))
        rgb = rgb[y0:y0 + self.crop, x0:x0 + self.crop]
        depth = depth[y0:y0 + self.crop, x0:x0 + self.crop]
        scale = self.image_size / self.crop
        rects = [r.shifted(-x0, -y0).scaled(scale) for r in rects]

        rgd = make_rgd(rgb, depth, depth_channel=self.depth_channel)
        rgd = torch.nn.functional.interpolate(rgd[None], size=(self.image_size, self.image_size),
                                              mode="bilinear", align_corners=False)[0]
        if self.augment and random.random() < 0.5:
            rgd = torch.flip(rgd, dims=[2])
            rects = [r.flipped_x(self.image_size) for r in rects]

        boxes = torch.tensor(np.array([r.canonical_box() for r in rects]), dtype=torch.float32).reshape(-1, 4)
        labels = torch.tensor([r.angle_bin() + 1 for r in rects], dtype=torch.int64)
        target = {"boxes": boxes, "labels": labels, "rects": rects}
        return rgd, target


def collate(batch):
    return tuple(zip(*batch))
