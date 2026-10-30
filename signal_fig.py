#!/usr/bin/env python3
"""Is the congestion imposed by a signal beyond the exit? Measured data only.

Four signatures of a signal, each one panel of fig_exit_signal.pdf:
  (a) a regular cycle of stops at the exit (the stop periods the simulation's
      exit stop line uses are shaded);
  (b) outflow that switches off and on: cumulative vehicles past 200 m,
      non-bikes flat during stops, bikes filtering through;
  (c) the cycle is strongest at the exit and fades upstream (autocorrelation of
      mean speed against lag);
  (d) after each release, restarts travel upstream from the front (start wave).

    python signal_fig.py --out paper_joint
"""
import argparse, json, os
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

from iam_calib.config import PipelineConfig
from iam_calib.jointsim import prep_frames, exit_stops

DT = 0.04
RED, NB, BK = "#C0392B", "#1F4E79", "#E39B2D"
plt.rcParams.update({"font.size": 7, "axes.linewidth": .6, "figure.dpi": 200,
                     "axes.spines.top": False, "axes.spines.right": False})


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", default="paper_joint")
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    df = pd.read_csv("D3F1_lcl_smtF.csv")
    d, _, _ = prep_frames(df, PipelineConfig())
    stop, x_stop, runs = exit_stops(d)
    d["xc"] = d.x - 0.5 * d.L; x0 = d.xc.min(); d["xc"] -= x0; d["xf"] = d.x - x0
    d["t"] = d.frame * DT; sec = d.xc.max()
    per = [(p * DT, q * DT) for p, q in runs]
    shade = lambda ax: [ax.axvspan(p, q, color=RED, alpha=.12, lw=0) for p, q in per]

    fig, ax = plt.subplots(2, 2, figsize=(7.1, 4.8))
    # (a) stopped share in the last 20 m
    z = d[d.xc > sec - 20]
    sh = (z.vx < 0.5).groupby((z.t // 1).astype(int)).mean()
    A = ax[0, 0]; shade(A)
    A.plot(sh.index, 100 * sh.values, color=RED, lw=.7)
    # one cycle per stop: a stop starting within 60 s of the previous one is the
    # same red phase with a brief release (500-565 s), not a new cycle
    on = []
    for p, _ in per:
        if not on or p - on[-1] > 60: on.append(p)
    on = np.array(on)
    for p in on: A.axvline(p, color=RED, lw=.5, ls=":")
    gaps = np.diff(on)
    A.set_xlabel("time (s)"); A.set_ylabel("vehicles stopped (%)"); A.set_xlim(0, 781)
    A.set_title("(a) last 20 m: share below 0.5 m/s (1 s means)\nstops begin at "
                + ", ".join(f"{p:.0f}" for p in on) + " s (dotted)", fontsize=7)
    # (b) cumulative outflow past 200 m
    cross = d[d.xf >= 200].groupby("vid").agg(t=("t", "min"), cls=("cls", "first"))
    B = ax[0, 1]; shade(B)
    for sel, col, lab in ((cross.cls != "Bike", NB, "all classes except Bike"),
                          (cross.cls == "Bike", BK, "Bike")):
        tt = np.sort(cross.t[sel].values)
        B.step(tt, np.arange(1, len(tt) + 1), where="post", color=col, lw=.9, label=lab)
    B.set_xlabel("time (s)"); B.set_ylabel("vehicles past 200 m (cumulative)"); B.set_xlim(0, 781)
    B.set_title("(b) outflow at the exit: flat steps = nothing leaves", fontsize=7)
    B.legend(fontsize=5.5, frameon=False, loc="upper left")
    # (c) autocorrelation of mean speed at three positions
    C = ax[1, 0]
    lags = np.arange(5, 300, 5)
    for X, col in ((195, RED), (100, "#555"), (20, "#AAA")):
        w = d[(d.xc > X - 10) & (d.xc < X + 10)]
        s = w.groupby((w.t // 5).astype(int)).vx.mean().reindex(range(157)).interpolate().bfill().ffill()
        q = (s - s.mean()).values
        r = [np.corrcoef(q[:-k // 5], q[k // 5:])[0, 1] for k in lags]
        C.plot(lags, r, color=col, lw=1, label=f"at {X} m")
    C.axvline(gaps.mean(), color=RED, lw=.6, ls=":")
    C.axhline(0, color="#999", lw=.4)
    C.set_xlabel("lag (s)"); C.set_ylabel("autocorrelation of mean speed")
    C.set_title(f"(c) a {gaps.mean():.0f} s cycle, strongest at the exit", fontsize=7)
    C.legend(fontsize=5.5, frameon=False)
    # (d) start wave: when each position restarts after each release
    D = ax[1, 1]
    xs = np.arange(120, 201, 10)
    for k, (p, q) in enumerate(per):
        if q - p < 30: continue
        rt = []
        for X in xs:
            w = d[(abs(d.xc - X) < 5) & (d.t >= q - 5) & (d.t < q + 60)]
            sp = w.groupby(w.t.round(0)).vx.mean()
            m = sp[(sp.index >= q) & (sp > 2)]
            rt.append(m.index.min() - q if len(m) else np.nan)
        D.plot(rt, xs, "o-", ms=2.2, lw=.8, label=f"release at {q:.0f} s")
    D.set_xlabel("time after release until mean speed > 2 m/s (s)")
    D.set_ylabel("position along section (m)")
    D.set_title("(d) restarts move upstream from the front", fontsize=7)
    D.legend(fontsize=5, frameon=False, loc="lower left")
    for x in ax.flat: x.tick_params(length=2, pad=1.5)
    fig.tight_layout(); fig.savefig(f"{a.out}/fig_exit_signal.pdf", bbox_inches="tight"); plt.close(fig)
    nb = cross[cross.cls != "Bike"]; bk = cross[cross.cls == "Bike"]
    inr = lambda t: np.any([(t >= p) & (t < q) for p, q in per], axis=0)
    tr = sum(q - p for p, q in per); tg = 781 - tr
    summ = {"stops_s": per, "onset_gaps_s": gaps.tolist(), "cycle_mean_s": float(gaps.mean()),
            "cycle_cv": float(gaps.std() / gaps.mean()), "x_stop_m": x_stop - x0,
            "outflow_per_min": {"nonbike_stop": 60 * inr(nb.t.values).sum() / tr,
                                "nonbike_other": 60 * (~inr(nb.t.values)).sum() / tg,
                                "bike_stop": 60 * inr(bk.t.values).sum() / tr,
                                "bike_other": 60 * (~inr(bk.t.values)).sum() / tg}}
    json.dump(summ, open("results/signal_summary.json", "w"), indent=1)
    print("wrote fig_exit_signal.pdf"); print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    main()
