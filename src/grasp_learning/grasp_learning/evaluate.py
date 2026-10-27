"""Evaluate a checkpoint on the Cornell held-out fold with the rectangle metric.

    ros2 run grasp_learning evaluate --data ~/datasets/cornell --model frcnn.pt

Reports top-1 accuracy (the highest-scoring grasp per image is correct if within 30 deg
and Jaccard > 0.25 of any ground-truth rectangle) and inference time, the numbers used
to compare against the classical EFD pipeline.
"""

import argparse
import time

import torch

from grasp_learning.cornell import CornellDataset, find_samples, image_wise_split
from grasp_learning.model import build_model, predict
from grasp_learning.rectangle import rectangle_metric


@torch.no_grad()
def evaluate(model, dataset, device, top_k=1):
    model.eval()
    correct = 0
    elapsed = 0.0
    for i in range(len(dataset)):
        rgd, target = dataset[i]
        t0 = time.perf_counter()
        rects = predict(model, rgd.to(device), top_k=top_k, min_score=0.0)
        if device.type == "cuda":
            torch.cuda.synchronize()
        elapsed += time.perf_counter() - t0
        if rects and any(rectangle_metric(r, target["rects"]) for r in rects[:top_k]):
            correct += 1
    n = max(1, len(dataset))
    return {"accuracy": correct / n, "ms_per_image": elapsed / n, "images": len(dataset)}


def load_model(path, device):
    ckpt = torch.load(path, map_location=device)
    model = build_model(pretrained=False, image_size=ckpt.get("image_size", 320))
    model.load_state_dict(ckpt["model"])
    model.depth_channel = ckpt.get("depth_channel", 2)
    return model.to(device).eval()


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--fold", type=int, default=0)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args(argv)

    device = torch.device(args.device)
    model = load_model(args.model, device)
    _, test_s = image_wise_split(find_samples(args.data), fold=args.fold)
    m = evaluate(model, CornellDataset(test_s, image_size=model.transform.min_size[0],
                                       depth_channel=model.depth_channel), device)
    print(f"{m['images']} images: top-1 accuracy {100 * m['accuracy']:.1f}%, {1e3 * m['ms_per_image']:.1f} ms/image")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
