"""Joint simulation: every vehicle integrated simultaneously, no neighbour replay.

Differs from objective.simulate in exactly the two ways that matter:
  * a vehicle's neighbours are other SIMULATED vehicles, not measured ones;
  * there is no mid-scene reset, so a scene runs free for its whole length.
A vehicle is initialised from measurement at the frame it first appears and
then runs free until it leaves.
"""
from __future__ import annotations
import numpy as np, pandas as pd
from numba import njit

from iam_calib.config import (PipelineConfig, MODEL_CONSTANTS as C,
                              dims_for_class, FREE_PARAMS)
from iam_calib.fastcore import _accel, PARAM_ORDER


def prep_frames(df, pcfg):
    """Frame-indexed, frame-sorted vehicle records shared by every scene."""
    cfg = pcfg.data; dims = pcfg.vehicle_dims
    d = df[[cfg.col_id, cfg.col_class, cfg.col_time, cfg.col_x, cfg.col_y,
            cfg.col_vx, cfg.col_vy, cfg.col_ax]].copy()
    d.columns = ["vid", "cls", "t", "x", "y", "vx", "vy", "ax"]
    t0 = d["t"].min()
    d["frame"] = np.rint((d["t"] - t0) / cfg.dt).astype(np.int64)
    d["L"] = d["cls"].map(lambda c: dims_for_class(c, dims)[0]).astype(float)
    d["W"] = d["cls"].map(lambda c: dims_for_class(c, dims)[1]).astype(float)
    if getattr(cfg, "x_is_front", False) is False:
        d["x"] = d["x"] + 0.5 * d["L"]
    d = d.sort_values("frame", kind="mergesort").reset_index(drop=True)
    # End of the measured section (largest centroid position in the record). A
    # simulated vehicle is retired when it crosses this line, not when its
    # measured twin leaves. Using the end of the section, rather than where
    # vehicles typically drop out of tracking (about 200.6 m), also means no
    # measured vehicle is ever beyond the line a retired vehicle is held at.
    d.attrs["x_exit_c"] = float((d.x - 0.5 * d.L).max())
    return d, d["frame"].to_numpy(), cfg.dt


def exit_stops(d, zone=20.0, vstop=0.5, share=0.5, smooth_s=1.0, merge_s=3.0,
               min_s=10.0, ahead=0.5, dt=0.04):
    """When and where the measured traffic is held at the exit.

    The record's queues start at the end of the section and grow backwards on
    a regular cycle, the signature of a signal or bottleneck just beyond it,
    which an open-ended simulation cannot contain. This reads that boundary
    condition off the data, the way entry times are already read off it at the
    other end: a frame is a stop frame when, smoothed over `smooth_s`, more than
    `share` of the vehicles whose centroid is in the last `zone` metres are
    slower than `vstop`. Stops closer than `merge_s` are merged and stops
    shorter than `min_s` dropped (single vehicles pausing). The stop line sits
    `ahead` metres beyond the median front bumper of the front-most stopped
    vehicle. Returns (stop[frame] bool, x_stop, [(start, end) frames])."""
    T = int(d.frame.max()) + 1
    xc = d.x - 0.5 * d.L
    z = d[xc > xc.max() - zone]
    sh = (z.vx < vstop).groupby(z.frame).mean().reindex(range(T))
    w = max(1, int(round(smooth_s / dt)))
    on = (sh.rolling(w, center=True, min_periods=max(1, w // 5)).mean() > share).fillna(False).to_numpy()
    e = np.flatnonzero(np.diff(np.r_[0, on.astype(int), 0]))
    runs = []
    for a, b in zip(e[::2], e[1::2]):
        if runs and a - runs[-1][1] < merge_s / dt:
            runs[-1] = (runs[-1][0], b)
        else:
            runs.append((a, b))
    runs = [(int(a), int(b)) for a, b in runs if b - a >= min_s / dt]
    stop = np.zeros(T, np.bool_)
    fronts = []
    for a, b in runs:
        stop[a:b] = True
        q = z[(z.frame >= a) & (z.frame < b) & (z.vx < vstop)]
        fronts.append(float(q.groupby("frame").x.max().median()))
    return stop, float(np.median(fronts)) + ahead, runs


def scene_slice(d, frames, f0, T, dt, min_present=100):
    """One scene: every vehicle present in frames [f0, f0+T). Returns None if
    the span holds too little data to score."""
    lo, hi = np.searchsorted(frames, [f0, f0 + T])
    g = d.iloc[lo:hi]
    if g.empty:
        return None
    vids = g["vid"].unique(); n = len(vids)
    idx = {v: k for k, v in enumerate(vids)}
    X = np.full((n, T), np.nan); Y = X.copy(); VX = X.copy(); VY = X.copy(); AX = X.copy()
    L = np.zeros(n); W = np.zeros(n); cls = [None] * n
    for v, gv in g.groupby("vid"):
        k = idx[v]; f = gv["frame"].to_numpy() - f0
        X[k, f] = gv["x"].to_numpy(); Y[k, f] = gv["y"].to_numpy()
        VX[k, f] = gv["vx"].to_numpy(); VY[k, f] = gv["vy"].to_numpy()
        AX[k, f] = gv["ax"].to_numpy()
        L[k] = gv["L"].iloc[0]; W[k] = gv["W"].iloc[0]; cls[k] = gv["cls"].iloc[0]
    present = ~np.isnan(X)
    if present.sum() < min_present:
        return None
    return dict(X=X, Y=Y, VX=VX, VY=VY, AX=AX, present=present, L=L, W=W,
                cls=cls, vids=list(vids), T=T, dt=dt, f0=f0, len_s=T * dt,
                x_exit_c=d.attrs["x_exit_c"])


def build_scenes(df, pcfg, scene_s=30.0):
    """Chop the record into consecutive scenes of one fixed length."""
    d, frames, dt = prep_frames(df, pcfg)
    spf = int(round(scene_s / dt))
    scenes = []
    for sid in range(int(frames.max()) // spf + 1):
        sc = scene_slice(d, frames, sid * spf, spf, dt)
        if sc is not None:
            scenes.append(sc)
    return scenes


@njit(cache=True)
def _run_scene(X,Y,VX,VY,AX,present,L,W,cid,P,T,dt,K,x_exit_c,
               radius_x,radius_y,
               bmax,longPar,bIntCrit,smaxfollow,accLatIntMaxFactor,
               useSeveralLeaders,accLatBMax,v0max,cool,narrow_on,clamp_on,
               dvdumax,dotdvdumax,road_on,y_left,y_right,
               stop,obey,x_stop):
    """Integrate every vehicle in a scene at once.

    Lifecycle. A vehicle enters from measurement at the first frame its twin is
    present and then runs on its own state. It leaves when its simulated
    centroid crosses `x_exit_c`, the line where real vehicles leave the
    section, and NOT when its measured twin leaves; otherwise a vehicle that
    lagged behind reality would vanish mid-road and free everyone behind it.
    After leaving it is held at the exit line, so frames its twin is still
    present are scored against that, not skipped (skipping would reward
    vehicles that run off early).

    Exit stop line. On frames where `stop[t]` is set, every vehicle with
    `obey[i]` that is still behind `x_stop` sees one extra leader: a stopped
    obstacle whose rear is at `x_stop`, spanning the road. It is given an
    enormous width so that it acts only longitudinally (its lateral desire,
    |dy|/W, is negligible). With `stop` all False this is the open exit.

    Neighbours. Each ego sees the K nearest vehicles by longitudinal distance
    from BOTH sides inside the +-radius window. K is set well above the number
    of vehicles that can share a window, and `ntrunc` counts the ego-steps for
    which it was not, so a silent cap cannot return.
    """
    n = X.shape[0]
    sx=np.full((n,T),np.nan); sy=np.full((n,T),np.nan); svx=np.full((n,T),np.nan)
    x=np.zeros(n); y=np.zeros(n); vx=np.zeros(n); vy=np.zeros(n)
    aprev=np.zeros(n); dvdu=np.zeros(n)
    live=np.zeros(n,np.bool_); exited=np.zeros(n,np.bool_)
    fx=np.zeros(n); fy=np.zeros(n); fv=np.zeros(n)
    nbx=np.empty(K+1); nby=np.empty(K+1); nbvx=np.empty(K+1); nbvy=np.empty(K+1)
    nbax=np.empty(K+1); nbL=np.empty(K+1); nbW=np.empty(K+1)
    val=np.empty(K+1,np.bool_); order=np.empty(n,np.int64)
    y_mid=0.5*(y_left+y_right)
    ntrunc=0
    for t in range(T):
        for i in range(n):
            if present[i,t]:
                if not live[i] and not exited[i]:
                    x[i]=X[i,t]; y[i]=Y[i,t]; vx[i]=VX[i,t]; vy[i]=VY[i,t]
                    aprev[i]=AX[i,t]; dvdu[i]=vy[i]/max(vx[i],1e-4); live[i]=True
            elif not live[i]:
                exited[i]=False                       # twin gone and nothing on the road
        for i in range(n):
            if live[i]: sx[i,t]=x[i]; sy[i,t]=y[i]; svx[i,t]=vx[i]
            elif exited[i] and present[i,t]: sx[i,t]=fx[i]; sy[i,t]=fy[i]; svx[i,t]=fv[i]
        if t==T-1: break
        ax_new=np.zeros(n); ay_new=np.zeros(n)
        nlive=0
        for i in range(n):
            if live[i]: order[nlive]=i; nlive+=1
        for q in range(1,nlive):                      # insertion sort on x
            key=order[q]; kx=x[key]; r=q-1
            while r>=0 and x[order[r]]>kx:
                order[r+1]=order[r]; r-=1
            order[r+1]=key
        for q in range(nlive):
            i=order[q]
            rb=q-1; rf=q+1; m=0
            while True:
                while rb>=0 and x[i]-x[order[rb]]<=radius_x and abs(y[order[rb]]-y[i])>radius_y:
                    rb-=1
                while rf<nlive and x[order[rf]]-x[i]<=radius_x and abs(y[order[rf]]-y[i])>radius_y:
                    rf+=1
                has_b = rb>=0 and x[i]-x[order[rb]]<=radius_x
                has_f = rf<nlive and x[order[rf]]-x[i]<=radius_x
                if not (has_b or has_f): break
                if m==K:
                    ntrunc+=1; break
                if has_b and (not has_f or x[i]-x[order[rb]]<=x[order[rf]]-x[i]):
                    j=order[rb]; rb-=1
                else:
                    j=order[rf]; rf+=1
                nbx[m]=x[j]; nby[m]=y[j]; nbvx[m]=vx[j]; nbvy[m]=vy[j]
                nbax[m]=aprev[j]; nbL[m]=L[j]; nbW[m]=W[j]; val[m]=True
                m+=1
            if stop[t] and obey[i] and x[i]<x_stop and x_stop-x[i]<=radius_x:
                nbx[m]=x_stop+1.0; nbL[m]=1.0; nby[m]=y_mid; nbW[m]=1e4
                nbvx[m]=0.0; nbvy[m]=0.0; nbax[m]=0.0; val[m]=True
                m+=1
            p=P[cid[i]]
            # only the filled slots: _accel loops over whatever length it is given
            a1,a2=_accel(x[i],y[i],vx[i],vy[i],nbx[:m],nby[:m],nbvx[:m],nbvy[:m],
                         nbax[:m],nbL[:m],nbW[:m],val[:m],
                         aprev[i],L[i],W[i],p,bmax,longPar,bIntCrit,smaxfollow,
                         accLatIntMaxFactor,useSeveralLeaders,accLatBMax,v0max,
                         cool,narrow_on,road_on,y_left,y_right)
            ax_new[i]=a1; ay_new[i]=a2
        for i in range(n):
            if not live[i]: continue
            a=ax_new[i]; al=ay_new[i]
            vxn=vx[i]+a*dt; vyn=vy[i]+al*dt
            if clamp_on:
                dn=vyn/max(vxn,1e-4)
                if abs(dn-dvdu[i])>dt*dotdvdumax:
                    dvdu[i]+= (1.0 if dn-dvdu[i]>0 else -1.0)*dotdvdumax*dt
                if dvdu[i]>dvdumax: dvdu[i]=dvdumax
                if dvdu[i]<-dvdumax: dvdu[i]=-dvdumax
                vm=abs(dvdu[i]*vxn)
                if vyn>vm: vyn=vm
                if vyn<-vm: vyn=-vm
            dy=0.5*(vy[i]+vyn)*dt
            if vxn< -1e-6:
                dx=-0.5*vx[i]*vx[i]/a if a!=0.0 else 0.0; vxn=0.0
            else:
                dx=0.5*(vx[i]+vxn)*dt
                if vxn<0.0: vxn=0.0
            x[i]+=dx; y[i]+=dy; vx[i]=vxn; vy[i]=vyn; aprev[i]=a
            if x[i]-0.5*L[i]>=x_exit_c:                # left the section
                live[i]=False; exited[i]=True
                fx[i]=x[i]; fy[i]=y[i]; fv[i]=vx[i]
    return sx,sy,svx,ntrunc
