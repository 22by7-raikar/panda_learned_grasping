#!/usr/bin/env python3
"""Summarise the per-attempt CSV written by pick_place_node.

    ros2 run grasp_planning analyze_benchmark.py /tmp/pick_place_benchmark.csv

Groups by (grasp source, mode, planner) and reports planning success, pick success,
planning times, joint-space path length and how straight the final approach was.
"""

import argparse
import sys

import pandas as pd


def summarise(df: pd.DataFrame) -> pd.DataFrame:
    keys = ["grasp_source", "mode", "planner"]
    # A cycle is one detection; its first successful (or last tried) attempt counts.
    g = df.groupby(keys)
    out = pd.DataFrame({
        "attempts": g.size(),
        "plan_success_%": 100 * g["planned"].mean(),
        "pick_success_%": 100 * g["lifted"].mean(),
        "transit_plan_ms_p50": 1e3 * g["transit_plan_time"].median(),
        "approach_plan_ms_p50": 1e3 * g["approach_plan_time"].median(),
        "transit_len_rad_p50": g["transit_joint_length"].median(),
        "approach_len_rad_p50": g["approach_joint_length"].median(),
        "approach_line_dev_mm_p95": 1e3 * g["approach_max_line_deviation"].quantile(0.95),
        "detect_ms_p50": 1e3 * g["detect_time"].median(),
    })
    return out.round(2)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("csv", nargs="+", help="benchmark CSV file(s)")
    parser.add_argument("--failures", action="store_true", help="also break down failure reasons")
    args = parser.parse_args(argv)

    df = pd.concat([pd.read_csv(p) for p in args.csv], ignore_index=True)
    for col in ("planned", "executed", "lifted"):
        df[col] = df[col].astype(bool)
    planned = df[df["planned"]]
    print(summarise(df).to_string())
    if not planned.empty:
        print("\nPlanned attempts only (path metrics are meaningful here):")
        print(summarise(planned).to_string())
    if args.failures:
        print("\nFailures:")
        print(df[df["failure"].notna()].groupby(["mode", "planner", "failure"]).size().to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())
