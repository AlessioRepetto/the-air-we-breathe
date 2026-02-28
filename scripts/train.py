#!/usr/bin/env python
from __future__ import annotations


import sys
from pathlib import Path as _Path

# Ensure 'src' is on PYTHONPATH when running from repo root
ROOT = _Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

import argparse
from pathlib import Path

import pandas as pd

from airq.modeling import AirQualityModel


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Path to input Excel/CSV containing the raw data.")
    ap.add_argument("--target", required=True, choices=["pm25", "o3"])
    ap.add_argument("--output", required=True, help="Path to save the trained model (joblib).")
    args = ap.parse_args()

    inp = Path(args.input)
    if inp.suffix.lower() in [".xlsx", ".xls"]:
        df = pd.read_excel(inp)
    else:
        df = pd.read_csv(inp)

    model = AirQualityModel.create_empty()
    model.fit(df, target=args.target)
    model.save(args.output)
    print(f"Saved model to: {args.output}")


if __name__ == "__main__":
    main()