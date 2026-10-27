"""Train the multi-grasp Faster R-CNN on the Cornell dataset.

    ros2 run grasp_learning train --data ~/datasets/cornell --out frcnn.pt --epochs 20

Image-wise 5-fold split; fold 0 is held out and evaluated with the rectangle metric
after every epoch. The checkpoint stores the weights plus the settings needed to rebuild
the model.
"""

import argparse
import time

import torch

from grasp_learning.cornell import CornellDataset, collate, find_samples, image_wise_split
from grasp_learning.evaluate import evaluate
from grasp_learning.model import build_model, freeze_batchnorm


def train_one_epoch(model, loader, optimizer, device, log_every=20):
    model.train()
    freeze_batchnorm(model)
    totals = {}
    for step, (images, targets) in enumerate(loader):
        images = [im.to(device) for im in images]
        targets = [{"boxes": t["boxes"].to(device), "labels": t["labels"].to(device)} for t in targets]
        losses = model(images, targets)
        loss = sum(losses.values())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        for k, v in losses.items():
            totals[k] = totals.get(k, 0.0) + float(v)
        if step % log_every == 0:
            print(f"  step {step:4d} loss {float(loss):.3f} " + " ".join(f"{k}={float(v):.3f}" for k, v in losses.items()))
    return {k: v / max(1, len(loader)) for k, v in totals.items()}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", required=True, help="Cornell dataset root")
    p.add_argument("--out", default="frcnn_grasp.pt")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--lr", type=float, default=0.005)
    p.add_argument("--image-size", type=int, default=320)
    p.add_argument("--fold", type=int, default=0)
    p.add_argument("--depth-channel", type=int, default=2, choices=[1, 2],
                   help="colour channel replaced by depth: 2 = blue (Chu et al.), 1 = green")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args(argv)

    samples = find_samples(args.data)
    if not samples:
        raise SystemExit(f"no Cornell samples (pcd*r.png + d.tiff + cpos.txt) under {args.data}")
    train_s, test_s = image_wise_split(samples, fold=args.fold)
    print(f"{len(train_s)} training / {len(test_s)} test images")

    train_ds = CornellDataset(train_s, image_size=args.image_size, augment=True, depth_channel=args.depth_channel)
    test_ds = CornellDataset(test_s, image_size=args.image_size, augment=False, depth_channel=args.depth_channel)
    loader = torch.utils.data.DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                                         num_workers=args.workers, collate_fn=collate)

    device = torch.device(args.device)
    model = build_model(pretrained=True, image_size=args.image_size).to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=args.lr, momentum=0.9, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[int(args.epochs * 0.7)], gamma=0.1)

    best = -1.0
    for epoch in range(args.epochs):
        t0 = time.time()
        losses = train_one_epoch(model, loader, optimizer, device)
        scheduler.step()
        metrics = evaluate(model, test_ds, device)
        print(f"epoch {epoch + 1}/{args.epochs} ({time.time() - t0:.0f}s) losses {losses} "
              f"top-1 accuracy {100 * metrics['accuracy']:.1f}% ({1e3 * metrics['ms_per_image']:.1f} ms/img)")
        if metrics["accuracy"] > best:
            best = metrics["accuracy"]
            torch.save({"model": model.state_dict(), "image_size": args.image_size, "accuracy": best,
                        "epoch": epoch + 1, "depth_channel": args.depth_channel}, args.out)
            print(f"  saved {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
