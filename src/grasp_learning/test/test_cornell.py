import numpy as np
import pytest

torch = pytest.importorskip("torch")
from PIL import Image  # noqa: E402

from grasp_learning.cornell import CornellDataset, find_samples, image_wise_split, read_cornell_rects  # noqa: E402
from grasp_learning.rectangle import angle_difference  # noqa: E402


@pytest.fixture
def dataset_root(tmp_path):
    d = tmp_path / "01"
    d.mkdir()
    for i in range(5):
        stem = d / f"pcd010{i}"
        Image.fromarray(np.full((480, 640, 3), 100, np.uint8)).save(f"{stem}r.png")
        Image.fromarray(np.full((480, 640), 0.6, np.float32)).save(f"{stem}d.tiff")
        # One 40 x 20 px rectangle at the image centre, opening along x, plus a NaN one.
        with open(f"{stem}cpos.txt", "w") as f:
            f.write("300 230\n300 250\n340 250\n340 230\n")
            f.write("NaN NaN\n1 1\n2 2\n3 3\n")
    return tmp_path


def test_reader_skips_nan_rectangles(dataset_root):
    rects = read_cornell_rects(dataset_root / "01" / "pcd0100cpos.txt")
    assert len(rects) == 1
    assert rects[0].w == pytest.approx(40) and rects[0].h == pytest.approx(20)


def test_dataset_crops_and_scales_targets(dataset_root):
    samples = find_samples(str(dataset_root))
    assert len(samples) == 5
    train, test = image_wise_split(samples, fold=0, folds=5)
    assert len(train) == 4 and len(test) == 1

    ds = CornellDataset(samples, image_size=200, crop=400)
    rgd, target = ds[0]
    assert rgd.shape == (3, 200, 200)
    r = target["rects"][0]
    # Crop starts at (120, 40); centre (320, 240) -> (200, 200) * 0.5 = (100, 100).
    assert (r.cx, r.cy) == pytest.approx((100, 100))
    assert r.w == pytest.approx(20)
    assert angle_difference(r.theta, 0.0) < 1e-9
    assert target["boxes"].shape == (1, 4)
    assert int(target["labels"][0]) == r.angle_bin() + 1
