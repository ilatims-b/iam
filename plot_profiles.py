#!/usr/bin/env python3
"""Figure and summary of one-at-a-time sensitivity profiles (oat_profiles.py).

Figure: one panel per parameter, one line per class, the change in the
training objective F against the parameter value across its box; dots mark the
fitted values. Summary, per parameter and class:
  range  max F - min F across the box: how much the parameter can matter;
  flat   share of the box within 1 % of the base F of the best point on the
         profile: near 1 means the data barely constrain it;
  edge   whether the best point on the profile lies at an end of the box.

    python plot_profiles.py --run results/j12_dims
"""
import argparse, json
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.ticker
import matplotlib.pyplot as plt

from iam_calib.config import REALISTIC_BOUNDS

# Okabe-Ito colour-blind-safe palette, one strong colour per class
CCOL = {"Auto": "#E69F00", "Bike": "#0072B2", "Car": "#009E73", "LCV": "#D55E00",
        "MultiAxle": "#000000", "Truck": "#CC79A7"}
LABEL = {"v0": "$v_0$ (m/s)", "T": "$T$ (s)", "s0": "$s_0$ (m)", "a": "$a$ (m/s$^2$)",
         "b": "$b$ (m/s$^2$)", "delta": "$\\delta$", "s0y": "$s_{0y}$ (m)",
         "s0yLat": "$s_{0y,lat}$ (m)", "sensLat": "$\\sigma$ (s)", "tauLatOVM": "$\\tau$ (s)",
         "pushLong": "$\\lambda_\\parallel$", "pushLat": "$\\lambda_\\perp$",
         "accBiasLat": "$\\beta$ (m/s$^2$)", "accLongBRef": "$f_B$ (m/s$^2$)",
         "accLatBRef": "$g_B$ (m/s$^2$)", "s0yB": "$s_{0y,B}$ (m)", "s0yLatB": "$s_{0y,latB}$ (m)"}
plt.rcParams.update({"font.size": 6.5, "axes.linewidth": .5, "figure.dpi": 200,
                     "axes.spines.top": False, "axes.spines.right": False})


METRIC = {"F": "training objective $F$", "long": "pooled RMSE, longitudinal position (m)", "lat": "pooled RMSE, lateral position (m)",
          "speed": "pooled RMSE, speed (m/s)"}


def rmse_profiles(d, base_row, cp, params, col, out):
    """Same layout as the F profiles, but the absolute pooled RMSE `col` on the
    training vehicles; the grey line is its value at the fit."""
    nc = 3; nr = int(np.ceil(len(params) / nc))
    fig, ax = plt.subplots(nr, nc, figsize=(7.1, 1.6 * nr))
    for i, p in enumerate(params):
        A = ax.flat[i]; lo, hi, log = REALISTIC_BOUNDS[p]
        for c, g in d[d.param == p].groupby("cls"):
            g = g.sort_values("value")
            A.plot(g.value, g[col], color=CCOL.get(c, "#333"), lw=1.2, label=c)
            v = cp[c].get(p)
            if v is not None:
                A.plot([v], [np.interp(v, g.value, g[col])], "o", color=CCOL.get(c, "#333"),
                       ms=3.2, mec="white", mew=.6, zorder=5)
        if log:
            A.set_xscale("log"); A.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        A.axhline(base_row[col], color="#666", lw=.6, ls="--")
        A.set_xlim(lo, hi); A.set_title(LABEL.get(p, p), fontsize=8, pad=3)
        A.tick_params(length=2, pad=1, labelsize=6.5)
    for j in range(len(params), nr * nc): ax.flat[j].axis("off")
    h, l = ax.flat[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=6, frameon=False, fontsize=6.5, bbox_to_anchor=(.5, -.01))
    fig.supylabel(f"{METRIC[col]}, training vehicles (dashed: value at the fit, {base_row[col]:.2f})",
                  fontsize=7, x=.005)
    fig.tight_layout(rect=[0.01, .03, 1, 1]); fig.savefig(out, bbox_inches="tight"); plt.close(fig)


KEY_PARAMS = ["T", "s0", "a", "b", "s0y", "sensLat"]                 # largest effect on F
WEAK_PARAMS = ["pushLat", "delta", "accLongBRef", "accLatBRef", "s0yB", "s0yLatB"]


def panel_profiles(d, base_row, cp, panels, out, ylabel, nc=3):
    """A chosen set of profiles, `panels` = [(parameter, measure)], `nc` per row,
    each panel titled by its parameter (and its measure when they differ)."""
    nr = int(np.ceil(len(panels) / nc))
    fig, ax = plt.subplots(nr, nc, figsize=(7.1, 2.0 * nr + (0.4 if nr == 1 else 0)), squeeze=False)
    one_metric = len({c for _, c in panels}) == 1
    for i, (p, col) in enumerate(panels):
        A = ax.flat[i]; lo, hi, log = REALISTIC_BOUNDS[p]
        for c, g in d[d.param == p].groupby("cls"):
            g = g.sort_values("value")
            A.plot(g.value, g[col], color=CCOL.get(c, "#333"), lw=1.3, label=c)
            v = cp[c].get(p)
            if v is not None:
                A.plot([v], [np.interp(v, g.value, g[col])], "o", color=CCOL.get(c, "#333"),
                       ms=3.4, mec="white", mew=.6, zorder=5)
        if log:
            A.set_xscale("log"); A.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        A.axhline(base_row[col], color="#666", lw=.6, ls="--")
        A.set_xlim(lo, hi); A.tick_params(length=2, pad=1, labelsize=7)
        short = {"F": "$F$", "long": "longitudinal position (m)", "lat": "lateral position (m)",
                 "speed": "speed (m/s)"}
        A.set_title(LABEL.get(p, p) if one_metric else short[col], fontsize=8.5, pad=3)
        if not one_metric: A.set_xlabel(LABEL.get(p, p), fontsize=7.5)
    for j in range(len(panels), nr * nc): ax.flat[j].axis("off")
    h, l = ax.flat[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=6, frameon=False, fontsize=7.5, bbox_to_anchor=(.5, -.02))
    if ylabel: fig.supylabel(ylabel, fontsize=7.5, x=.005)
    fig.tight_layout(rect=[0.01, .1 if nr == 1 else .05, 1, 1])
    fig.savefig(out, bbox_inches="tight"); plt.close(fig)


def spread_boxes(d, base, params, out):
    """Distribution of F over all values tried, one box per class, per parameter."""
    nc = 4; nr = int(np.ceil(len(params) / nc))
    fig, ax = plt.subplots(nr, nc, figsize=(7.1, 1.75 * nr), sharey=True)
    classes = sorted(d.cls.dropna().unique())
    for i, p in enumerate(params):
        A = ax.flat[i]
        data = [d[(d.param == p) & (d.cls == c)].F.to_numpy() for c in classes]
        bp = A.boxplot(data, widths=.6, patch_artist=True, showfliers=True,
                       flierprops=dict(ms=1.2, mew=.3), medianprops=dict(color="k", lw=.6),
                       whiskerprops=dict(lw=.5), capprops=dict(lw=.5), boxprops=dict(lw=.4))
        for b, c in zip(bp["boxes"], classes): b.set_facecolor(CCOL.get(c, "#333"))
        A.axhline(base, color="#C0392B", lw=.5, ls="--")
        A.set_xticks(range(1, len(classes) + 1))
        A.set_xticklabels(classes, rotation=45, ha="right", fontsize=6)
        A.set_title(LABEL.get(p, p), fontsize=8, pad=3); A.tick_params(length=2, pad=1, labelsize=6.5)
        A.set_yscale("log")
    for j in range(len(params), nr * nc): ax.flat[j].axis("off")
    fig.supylabel(f"training $F$ over all values tried (log scale); dashed: $F$ at the fit = {base:.2f}",
                  fontsize=7, x=.005)
    fig.tight_layout(rect=[0.01, 0, 1, 1]); fig.savefig(out, bbox_inches="tight"); plt.close(fig)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--run", required=True)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    d = pd.read_csv(f"{a.run}/oat_profiles.csv")
    base_row = d[d.param == "base"].iloc[0]
    base = float(base_row.F); d = d[d.param != "base"]
    cp = json.load(open(f"{a.run}/class_params.json"))
    params = [p for p in REALISTIC_BOUNDS if p in set(d.param)]
    for col in METRIC:
        rmse_profiles(d, base_row, cp, params, col, f"{a.run}/fig_oat_rmse_{col}.pdf")
    spread_boxes(d, base, params, f"{a.run}/fig_oat_spread.pdf")
    lab = f"training objective $F$ (dashed: value at the fit, {base:.2f})"
    panel_profiles(d, base_row, cp, [(p, "F") for p in KEY_PARAMS], f"{a.run}/fig_oat_key.pdf", lab)
    panel_profiles(d, base_row, cp, [(p, "F") for p in WEAK_PARAMS], f"{a.run}/fig_oat_weak.pdf", lab)
    panel_profiles(d, base_row, cp, [("T", m) for m in ("long", "lat", "speed")],
                   f"{a.run}/fig_oat_T_rmse.pdf", "pooled RMSE, training vehicles (dashed: value at the fit)")
    nc = 4; nr = int(np.ceil(len(params) / nc))
    fig, ax = plt.subplots(nr, nc, figsize=(7.1, 1.75 * nr))
    rows = []
    for i, p in enumerate(params):
        A = ax.flat[i]; lo, hi, log = REALISTIC_BOUNDS[p]
        for c, g in d[d.param == p].groupby("cls"):
            g = g.sort_values("value")
            A.plot(g.value, g.F - base, color=CCOL.get(c, "#333"), lw=.9, label=c)
            v = cp[c].get(p)
            if v is not None:
                A.plot([v], [np.interp(v, g.value, g.F) - base], "o", color=CCOL.get(c, "#333"),
                       ms=2.2, mec="k", mew=.3)
            m = g.F.min(); k = int(np.argmin(g.F.values))
            rows.append(dict(param=p, cls=c, fitted=v, best_on_profile=float(g.value.iloc[k]),
                             range=float(g.F.max() - m),
                             flat=float((g.F <= m + 0.01 * base).mean()),
                             edge=k in (0, len(g) - 1)))
        if log:
            A.set_xscale("log"); A.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        A.axhline(0, color="#999", lw=.4)
        A.set_xlim(lo, hi); A.set_title(LABEL.get(p, p), fontsize=8, pad=3)
        A.tick_params(length=2, pad=1, labelsize=6.5)
        top = d[d.param == p].F.max() - base
        A.set_ylim(min(-0.02 * base, (d[d.param == p].F.min() - base) * 1.1), max(top * 1.05, 0.02 * base))
    for j in range(len(params), nr * nc): ax.flat[j].axis("off")
    h, l = ax.flat[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=6, frameon=False, fontsize=6.5, bbox_to_anchor=(.5, -.01))
    fig.supylabel(f"change in training objective $F$ (base $F$ = {base:.3f})", fontsize=7, x=.005)
    fig.tight_layout(rect=[0.01, .03, 1, 1])
    out = a.out or f"{a.run}/fig_oat_profiles.pdf"
    fig.savefig(out, bbox_inches="tight"); plt.close(fig); print("wrote", out)
    s = pd.DataFrame(rows); s.to_csv(f"{a.run}/oat_summary.csv", index=False)
    piv = s.pivot(index="param", columns="cls", values="range").reindex(params)
    print("\nrange of F across the box (importance), by parameter and class:")
    print(piv.round(3).to_string())
    print("\nflat share of the box (1 = unconstrained):")
    print(s.pivot(index="param", columns="cls", values="flat").reindex(params).round(2).to_string())
    print("\nbest point at a box edge:", ", ".join(f"{r.param}/{r.cls}" for r in s.itertuples() if r.edge))


if __name__ == "__main__":
    main()
