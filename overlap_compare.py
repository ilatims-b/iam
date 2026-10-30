#!/usr/bin/env python3
"""Overlaps of the measured vehicle rectangles under the earlier and the current
class dimensions (report, Appendix "Vehicle dimensions and overlaps").

Every fifth frame, each vehicle is drawn as its class rectangle at its measured
centroid and every intersecting pair is counted.

    python overlap_compare.py
"""
import numpy as np, pandas as pd
from iam_calib import overlap_figs as o
from iam_calib.config import PipelineConfig, dims_for_class
NEW = {"Auto": (2.7, 1.4), "Bike": (1.8, 0.65), "Car": (4.0, 1.7), "LCV": (4.5, 1.75),
       "Truck": (8.5, 2.4), "MultiAxle": (12.0, 2.5), "Trailer": (15.5, 2.5)}
d = o.load(); dims = PipelineConfig().vehicle_dims
cls = sorted(d.cls.unique())
print("class      old (L,W)      new (L,W)")
for c in cls: print(f"{c:10s} {str(dims_for_class(c, dims)):14s} {NEW.get(c)}")
res = {}
for tag in ("old", "new"):
    e = d.copy()
    if tag == "new":
        e["L"] = e.cls.map(lambda c: NEW[c][0]); e["W"] = e.cls.map(lambda c: NEW[c][1])
    p, nveh = o.pairs(e, 5)
    inv = pd.unique(np.r_[p.frame.astype(str) + "_" + p.va.astype(str), p.frame.astype(str) + "_" + p.vb.astype(str)])
    res[tag] = dict(pairs=len(p), veh_frames_in_overlap=len(inv) / nveh.sum(),
                    frames_any=p.frame.nunique() / len(nveh),
                    vehicles_ever=len(set(p.va) | set(p.vb)) / d.vid.nunique(),
                    med_ox=p.ox.median(), med_oy=p.oy.median(),
                    share=p.pair.value_counts(normalize=True))
for k in ("pairs", "veh_frames_in_overlap", "frames_any", "vehicles_ever", "med_ox", "med_oy"):
    print(f"{k:24s} {res['old'][k]:10.4f} {res['new'][k]:10.4f}")
s = pd.concat({t: res[t]["share"] * res[t]["pairs"] for t in res}, axis=1).fillna(0).astype(int)
print(s.sort_values("old", ascending=False).to_string())
