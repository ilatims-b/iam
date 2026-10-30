#!/usr/bin/env python3
"""Where and when the MEASURED vehicles' model boxes overlap.

Each vehicle is drawn as the rectangle the model gives it: its measured
centroid, and the fixed length and width of its class. Two boxes overlap when
they intersect in both x and y. No simulation is involved; this is a property
of the data plus the fixed class dimensions.

    python overlap_figs.py --out paper_joint
"""
import argparse, json, os
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Patch

from iam_calib.config import PipelineConfig, dims_for_class

EDGES = (0.0, 8.9)
SEC = 204.0
DT = 0.04
ORDER = ["Bike", "Car", "LCV", "Truck", "Auto", "MultiAxle", "Trailer"]
CCOL = {"Bike": "#7FA7C9", "Car": "#9BC59D", "LCV": "#E3C16F", "Truck": "#C9A27F",
        "Auto": "#B9A3D0", "MultiAxle": "#A0A0A0", "Trailer": "#6F6F6F"}
HIT = "#C0392B"
plt.rcParams.update({"font.size": 7, "axes.linewidth": .6, "figure.dpi": 200,
                     "axes.spines.top": False, "axes.spines.right": False})


def load():
    df = pd.read_csv("D3F1_lcl_smtF.csv")
    dims = PipelineConfig().vehicle_dims
    d = pd.DataFrame({"vid": df.VehicleID, "cls": df.Class,
                      "frame": np.rint((df.GTime_sec - df.GTime_sec.min()) / DT).astype(int),
                      "x": df.X_Smt - df.X_Smt.min(), "y": df.Y_Smt})
    d["L"] = d.cls.map(lambda c: dims_for_class(c, dims)[0])
    d["W"] = d.cls.map(lambda c: dims_for_class(c, dims)[1])
    return d.sort_values("frame").reset_index(drop=True)


def pairs(d, step=5):
    """Every overlapping pair on every `step`-th frame."""
    out, nveh = [], {}
    for f, g in d[d.frame % step == 0].groupby("frame"):
        x, y, L, W = (g[k].to_numpy() for k in ("x", "y", "L", "W"))
        nveh[f] = len(g)
        ox = 0.5 * (L[:, None] + L[None]) - np.abs(x[:, None] - x[None])
        oy = 0.5 * (W[:, None] + W[None]) - np.abs(y[:, None] - y[None])
        i, j = np.where(np.triu((ox > 0) & (oy > 0), 1))
        if i.size == 0: continue
        c = g.cls.to_numpy(); v = g.vid.to_numpy()
        for a, b in zip(i, j):
            ca, cb = sorted((c[a], c[b]), key=ORDER.index)
            out.append((f, v[a], v[b], f"{ca}--{cb}", 0.5 * (x[a] + x[b]),
                        min(ox[a, b], L[a], L[b]), min(oy[a, b], W[a], W[b])))
    p = pd.DataFrame(out, columns=["frame", "va", "vb", "pair", "x", "ox", "oy"])
    return p, pd.Series(nveh)


def _draw_where_when(fig, ax, p, s5, occ):
    """Draw the four where/when panels on ax[0..3]; a None slot is skipped."""
    ax = list(ax)
    # when: share of vehicles in at least one overlap, with occupancy
    a = ax[0]
    if a is not None:
        a.plot(s5.index * 5, 100 * s5.values, color=HIT, lw=.9)
        a.set_xlabel("time (s)"); a.set_ylabel("vehicles in an overlap (%)", color=HIT)
        a.set_title("when: share of vehicles overlapping a neighbour (5 s means)", fontsize=7)
        a2 = a.twinx(); a2.plot(occ.index * 5, occ.values, color="#555", lw=.6, alpha=.7)
        a2.set_ylabel("vehicles on section", color="#555"); a2.spines["top"].set_visible(False)
        a.set_xlim(0, 781)
    # where: overlapping pairs along the section
    a = ax[1]
    if a is not None:
        a.hist(p.x, bins=np.arange(0, SEC + 4, 4), color=HIT, alpha=.8, lw=0)
        a.set_xlabel("position along section (m)"); a.set_ylabel("overlapping pairs (frames x pairs)")
        a.set_title("where: location of overlapping pairs", fontsize=7); a.set_xlim(0, SEC)
    # where and when together
    a = ax[2]
    if a is not None:
        h = a.hist2d(p.frame * DT, p.x, bins=[np.arange(0, 800, 20), np.arange(0, SEC + 8, 8)],
                     cmap="Reds", cmin=1, rasterized=True)
        cb = fig.colorbar(h[3], ax=a, pad=.02, fraction=.05); cb.set_label("pairs", fontsize=6)
        cb.ax.tick_params(labelsize=5, length=2)
        a.set_xlabel("time (s)"); a.set_ylabel("position (m)")
        a.set_title("where and when (20 s x 8 m cells)", fontsize=7)
    # which classes
    a = ax[3]
    if a is not None:
        top = p.pair.value_counts().head(10)[::-1]
        a.barh(range(len(top)), 100 * top.values / len(p), color=HIT, alpha=.8)
        a.set_yticks(range(len(top))); a.set_yticklabels([t.replace("--", " / ") for t in top.index], fontsize=6)
        a.set_xlabel("share of overlapping pairs (%)"); a.set_title("which classes overlap", fontsize=7)


def where_when(d, p, nveh, out, step=5):
    # when: share of vehicles in at least one overlap, 5 s means
    inv = pd.concat([p[["frame", "va"]].rename(columns={"va": "v"}),
                     p[["frame", "vb"]].rename(columns={"vb": "v"})]).drop_duplicates()
    share = (inv.groupby("frame").size() / nveh).reindex(nveh.index).fillna(0)
    b = (share.index * DT // 5).astype(int)
    s5 = share.groupby(b).mean(); occ = nveh.groupby(b).mean()
    fig, ax = plt.subplots(2, 2, figsize=(7.1, 4.6))
    _draw_where_when(fig, ax.flat, p, s5, occ)
    for a in ax.flat: a.tick_params(length=2, pad=1.5)
    fig.tight_layout(); fig.savefig(f"{out}/fig_overlap_where.pdf", bbox_inches="tight"); plt.close(fig)
    print("wrote fig_overlap_where.pdf")
    # the same four panels as two figures of two, for layouts that need to split it
    for k, part in enumerate("ab"):
        fig, ax2 = plt.subplots(1, 2, figsize=(7.1, 2.3))
        slots = [None] * 4; slots[2 * k], slots[2 * k + 1] = ax2
        _draw_where_when(fig, slots, p, s5, occ)
        for a in ax2: a.tick_params(length=2, pad=1.5)
        fig.tight_layout(); fig.savefig(f"{out}/fig_overlap_where_{part}.pdf", bbox_inches="tight"); plt.close(fig)
        print(f"wrote fig_overlap_where_{part}.pdf")
    return share


def pick_snapshots(p, n, win, min_gap_s=20.0):
    """The (frame, window) cells with the most overlapping pairs, at least
    `min_gap_s` apart in time so that no two snapshots show the same queue."""
    starts = np.arange(0, SEC - win + 1, 4.0)
    rows = []
    for f, g in p.groupby("frame"):
        cnt = [((g.x >= s0) & (g.x < s0 + win)).sum() for s0 in starts]
        k = int(np.argmax(cnt)); rows.append((cnt[k], f, starts[k]))
    rows.sort(reverse=True); chosen = []
    for c, f, s0 in rows:
        if all(abs(f - g) * DT >= min_gap_s for _, g, _ in chosen):
            chosen.append((c, f, s0))
        if len(chosen) == n: break
    return sorted(chosen, key=lambda r: r[1])


def draw_frame(a, d, p, f, s0, win):
    g = d[(d.frame == f) & (d.x + d.L / 2 > s0 - 2) & (d.x - d.L / 2 < s0 + win + 2)]
    pf = p[p.frame == f]; hit = set(pf.va) | set(pf.vb)
    a.axhspan(*EDGES, color="#EEEEEE", lw=0, zorder=0)
    for yy, ls in ((EDGES[0], "-"), (EDGES[1], "-"), (0.2, ":")):
        a.axhline(yy, color="#555", lw=.5, ls=ls, zorder=1)
    rect = {}
    for r in g.itertuples():
        on = r.vid in hit
        R = Rectangle((r.x - r.L / 2, r.y - r.W / 2), r.L, r.W, fc=CCOL.get(r.cls, "#ccc"),
                      ec=HIT if on else "#333", lw=.9 if on else .35, alpha=.85, zorder=2)
        a.add_patch(R); rect[r.vid] = (r.x, r.y, r.L, r.W)
    for q in pf.itertuples():                  # shade the intersection itself
        if q.va not in rect or q.vb not in rect: continue
        (xa, ya, La, Wa), (xb, yb, Lb, Wb) = rect[q.va], rect[q.vb]
        x0 = max(xa - La / 2, xb - Lb / 2); x1 = min(xa + La / 2, xb + Lb / 2)
        y0 = max(ya - Wa / 2, yb - Wb / 2); y1 = min(ya + Wa / 2, yb + Wb / 2)
        a.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fc=HIT, ec="none", alpha=.75, zorder=3))
    a.set_xlim(s0, s0 + win); a.set_ylim(-0.3, EDGES[1] + 0.3); a.set_aspect("equal")
    a.set_yticks([0, 4, 8]); a.tick_params(length=2, pad=1.5, labelsize=6)
    n_in = len(g[(g.x >= s0) & (g.x < s0 + win)])
    a.set_title(f"t = {f * DT:.1f} s,  {len(pf[(pf.x >= s0) & (pf.x < s0 + win)])} overlapping "
                f"pairs among {n_in} vehicles", fontsize=6.5, loc="left", pad=2)


def snapshots(d, p, snaps, win, out, per_page=6):
    pages = [snaps[i:i + per_page] for i in range(0, len(snaps), per_page)]
    names = []
    for k, page in enumerate(pages, 1):
        fig, ax = plt.subplots(len(page), 1, figsize=(7.1, 1.42 * len(page) + .4))
        for a, (_, f, s0) in zip(np.atleast_1d(ax), page):
            draw_frame(a, d, p, f, s0, win)
            a.set_ylabel("y (m)", fontsize=6)
        np.atleast_1d(ax)[-1].set_xlabel("position along section, x (m); traffic moves left to right")
        h = [Patch(fc=CCOL[c], ec="#333", lw=.35, label=c) for c in ORDER]
        h += [Patch(fc="white", ec=HIT, lw=.9, label="box in an overlap"),
              Patch(fc=HIT, alpha=.75, label="overlapping area")]
        fig.legend(handles=h, loc="lower center", ncol=9, frameon=False, fontsize=6,
                   bbox_to_anchor=(.5, -.01), handlelength=1.2, columnspacing=.9)
        fig.tight_layout(rect=[0, .025, 1, 1]); nm = f"fig_overlap_snap{k}.pdf"
        fig.savefig(f"{out}/{nm}", bbox_inches="tight"); plt.close(fig); names.append(nm)
        print("wrote", nm)
    return names


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="paper_joint")
    ap.add_argument("--snaps", type=int, default=12)
    ap.add_argument("--win", type=float, default=50.0)
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    d = load(); p, nveh = pairs(d)
    share = where_when(d, p, nveh, a.out)
    snaps = pick_snapshots(p, a.snaps, a.win)
    snapshots(d, p, snaps, a.win, a.out)
    vf = (share * nveh).sum() / nveh.sum()
    pc = p.pair.value_counts()
    summ = {"frames_sampled": int(len(nveh)), "vehicle_frames": int(nveh.sum()),
            "share_vehicle_frames_in_overlap": float(vf),
            "frames_with_any_overlap": float((share > 0).mean()),
            "pair_share": {k: float(v / len(p)) for k, v in pc.items()},
            "median_long_overlap_m": float(p.ox.median()), "median_lat_overlap_m": float(p.oy.median()),
            "x_quartiles": [float(q) for q in np.percentile(p.x, [25, 50, 75])],
            "snapshots": [{"t": float(f * DT), "x0": float(s0), "pairs": int(c)} for c, f, s0 in snaps]}
    json.dump(summ, open("results/overlap_summary.json", "w"), indent=1)
    print(json.dumps({k: v for k, v in summ.items() if k not in ("pair_share", "snapshots")}, indent=1))
    print("top pairs:", {k: round(v, 3) for k, v in list(summ["pair_share"].items())[:6]})
