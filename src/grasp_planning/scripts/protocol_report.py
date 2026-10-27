#!/usr/bin/env python3
"""GS / GSS tables from benchmark_protocol.py results.

    ros2 run grasp_planning protocol_report.py results/*.csv [--markdown]

Grasp Success Score (GSS): % of trials where the object was picked up.
Grasp Score (GS):          points scored / points possible, one point each for
                           lift, yaw and shake (3 per trial).
"""

import argparse
import sys

import pandas as pd

STAGES = ["lift_passed", "yaw_passed", "shake_passed"]


def load(paths):
    df = pd.concat([pd.read_csv(p) for p in paths], ignore_index=True)
    for col in STAGES + ["grasp_success"]:
        df[col] = df[col].astype(str).str.lower().isin(["1", "true", "yes"])
    df["points"] = df[STAGES].sum(axis=1)
    return df


def scores(group: pd.DataFrame) -> pd.Series:
    n = len(group)
    return pd.Series({
        "trials": n,
        "GSS_%": 100.0 * group["grasp_success"].mean(),
        "GS_%": 100.0 * group["points"].sum() / (3 * n),
        "lift_%": 100.0 * group["lift_passed"].mean(),
        "yaw_%": 100.0 * group["yaw_passed"].mean(),
        "shake_%": 100.0 * group["shake_passed"].mean(),
    })


def summary(df: pd.DataFrame) -> pd.DataFrame:
    return df.groupby(["algorithm", "config"]).apply(scores, include_groups=False).round(1)


def per_object(df: pd.DataFrame, metric: str = "GS_%") -> pd.DataFrame:
    table = df.groupby(["algorithm", "config", "object"]).apply(scores, include_groups=False)[metric]
    return table.unstack("object").round(1)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("csv", nargs="+")
    parser.add_argument("--markdown", action="store_true", help="print Markdown tables")
    args = parser.parse_args(argv)

    df = load(args.csv)
    fmt = (lambda t: t.to_markdown()) if args.markdown else (lambda t: t.to_string())
    print("Overall\n" + fmt(summary(df)))
    print("\nGS % per object\n" + fmt(per_object(df, "GS_%")))
    print("\nGSS % per object\n" + fmt(per_object(df, "GSS_%")))
    failures = df[~df["grasp_success"]].groupby(["algorithm", "config", "failure"]).size()
    if not failures.empty:
        print("\nFailed trials by reason\n" + fmt(failures.to_frame("trials")))
    return 0


if __name__ == "__main__":
    sys.exit(main())
