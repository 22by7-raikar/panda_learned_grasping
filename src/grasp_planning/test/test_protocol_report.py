import os
import sys

import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
from protocol_report import load, per_object, summary  # noqa: E402


@pytest.fixture
def results(tmp_path):
    rows = []
    # efd: 2 trials on the bottle (3/3 and 1/3), 1 failed trial on the ball.
    for obj, pose, lift, yaw, shake in [("bottle", "P1", 1, 1, 1), ("bottle", "P2", 1, 0, 0), ("ball", "P1", 0, 0, 0)]:
        rows.append({"algorithm": "efd", "config": "sim", "object": obj, "pose_id": pose,
                     "grasp_success": bool(lift), "lift_passed": bool(lift), "yaw_passed": bool(yaw),
                     "shake_passed": bool(shake), "points": lift + yaw + shake, "failure": "" if lift else "no_grasp"})
    path = tmp_path / "r.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return load([path])


def test_summary_scores(results):
    s = summary(results).loc[("efd", "sim")]
    assert s["trials"] == 3
    assert s["GSS_%"] == pytest.approx(66.7, abs=0.05)
    assert s["GS_%"] == pytest.approx(100 * 4 / 9, abs=0.05)
    assert s["shake_%"] == pytest.approx(33.3, abs=0.05)


def test_per_object(results):
    t = per_object(results, "GS_%")
    assert t.loc[("efd", "sim"), "bottle"] == pytest.approx(66.7, abs=0.05)
    assert t.loc[("efd", "sim"), "ball"] == 0.0
