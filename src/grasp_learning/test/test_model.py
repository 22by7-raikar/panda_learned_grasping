import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torchvision")

from grasp_learning.model import build_model, freeze_batchnorm, make_rgd, predict  # noqa: E402
from grasp_learning.rectangle import NUM_ANGLE_BINS, GraspRect  # noqa: E402


@pytest.fixture(scope="module")
def model():
    torch.manual_seed(0)
    return build_model(pretrained=False, image_size=160, box_score_thresh=0.0)


def test_architecture_matches_paper(model):
    # res4 trunk (1024 ch), RoIAlign 14x14, res5 + avgpool head (2048), fc 20 / fc 80.
    assert model.backbone.out_channels == 1024
    assert model.roi_heads.box_roi_pool.output_size == (14, 14)
    assert model.roi_heads.box_predictor.cls_score.out_features == NUM_ANGLE_BINS + 1
    assert model.roi_heads.box_predictor.bbox_pred.out_features == 4 * (NUM_ANGLE_BINS + 1)
    feats = model.roi_heads.box_head(torch.zeros(2, 1024, 14, 14))
    assert feats.shape == (2, 2048)


def test_training_step_gives_finite_losses(model):
    model.train()
    freeze_batchnorm(model)
    rects = [GraspRect(80, 80, 40, 15, 0.3), GraspRect(60, 100, 30, 12, -1.0)]
    target = {
        "boxes": torch.tensor(np.array([r.canonical_box() for r in rects]), dtype=torch.float32),
        "labels": torch.tensor([r.angle_bin() + 1 for r in rects]),
    }
    losses = model([torch.rand(3, 160, 160)], [target])
    assert set(losses) == {"loss_classifier", "loss_box_reg", "loss_objectness", "loss_rpn_box_reg"}
    total = sum(losses.values())
    assert torch.isfinite(total)
    total.backward()


def test_predict_returns_rects(model):
    rects = predict(model, torch.rand(3, 160, 160), top_k=5, min_score=0.0)
    assert len(rects) <= 5
    for r in rects:
        assert isinstance(r, GraspRect)
        assert -np.pi / 2 <= r.theta < np.pi / 2


def test_make_rgd():
    rgb = np.zeros((4, 5, 3), np.uint8)
    rgb[..., 0] = 255
    depth = np.linspace(0.4, 0.6, 20).reshape(4, 5)
    depth[0, 0] = np.nan
    rgd = make_rgd(rgb, depth)
    assert rgd.shape == (3, 4, 5)
    assert float(rgd[0].min()) == 1.0
    assert float(rgd[2].max()) == 1.0 and float(rgd[2, 0, 1]) == pytest.approx(0.0, abs=1e-6)
    assert float(rgd[2, 0, 0]) == 1.0  # hole -> far


def test_make_rgd_green_variant():
    rgb = np.full((2, 2, 3), 255, np.uint8)
    rgd = make_rgd(rgb, np.array([[0.4, 0.5], [0.6, 0.7]]), depth_channel=1)
    assert float(rgd[2].min()) == 1.0  # blue kept
    assert float(rgd[1, 0, 0]) == pytest.approx(0.0)  # green is depth
