#!/usr/bin/env python3
"""Evaluate pySLAM runs of the GS-AgriField benchmark against the RTK-GNSS/INS reference (camera frame).

    .venv_eval/bin/python evaluate.py --tag gsaf_benchmark      # or gsaf_slamtest
Writes <out>/runs.csv (one row per run) and, with --plot, top-view trajectory figures per sequence.

Protocol (see README.md):
  reference      groundtruth_cam.tum = /odom/sbg (map->imu_link) * T_imu_link_camera (/tf_static), one pose per image
  estimate       trajectory_online.txt (TUM, one pose per processed frame, as output online by pySLAM)
  association    evo sync.associate_trajectories, max |dt| = 20 ms (image timestamps on both sides)
  alignment      Sim(3) Umeyama on all associated poses (single scale per run; monocular scale is unobservable)
  ATE            evo APE, translation part, after Sim(3)                     [m] and [% of reference path length]
  RPE            evo RPE over 1 m and 10 m of travelled distance, translation [m] and rotation [deg]
  KITTI drift    t_rel [%] and r_rel [deg/100 m] averaged over sub-trajectories of 10..50 m (KITTI devkit
                 procedure: start every 10 frames, error of the relative pose over each length)
  scale drift    coefficient of variation of the per-10 m segment scale |dp_est| / |dp_ref| on the raw
                 (unaligned) estimate: 0 % = constant scale; independent of the global scale
  robustness     status (completed / stopped after STOP_LOST lost frames / crashed / not run),
                 tracked frames (% of all frames of the sequence), lost frames, initialisation delay,
                 loop detections / corrections (pySLAM loop_closing.log), runtime
  success        completed and >= 90 % of the frames tracked
"""
import argparse, glob, json, os, re
import numpy as np
import pandas as pd
from evo.core import metrics, sync
from evo.core.trajectory import PoseTrajectory3D
from evo.tools import file_interface

B = os.path.dirname(os.path.abspath(__file__))
ROOT = os.environ.get("PYSLAM_ROOT", os.path.dirname(os.path.dirname(B)))
DATA = os.path.join(ROOT, "data/kitti/pyslamdata")
MAX_DT = 0.02
KITTI_LENGTHS = [10, 20, 30, 40, 50]
SUCCESS_TRACKED = 90.0
STOP_LOST = int(os.environ.get("STOP_LOST", 50))


def find_runs(tag):
    out = []
    for it in glob.glob(os.path.join(ROOT, "results", "eval_*", tag, "**", "iteration_0"), recursive=True):
        rel = it.split(os.sep + tag + os.sep, 1)[1].split(os.sep)
        seq, method = rel[0], rel[1]
        rep = next((int(p[3:]) for p in rel if re.fullmatch(r"rep\d+", p)), 1)
        out.append(dict(sequence=seq, method=method, rep=rep, path=it, stamp=it.split(os.sep)[-len(rel) - 3]))
    df = pd.DataFrame(out)
    if len(df):  # keep the most recent run of each (sequence, method, rep)
        df = df.sort_values("stamp").groupby(["sequence", "method", "rep"]).tail(1)
    return df


def path_dist(p):
    return np.r_[0, np.cumsum(np.linalg.norm(np.diff(p, axis=0), axis=1))]


def kitti_drift(ref, est):
    """KITTI odometry devkit errors on aligned trajectories (lists of 4x4)."""
    d = path_dist(np.array([T[:3, 3] for T in ref]))
    t_err, r_err = [], []
    for s in range(0, len(ref), 10):
        for L in KITTI_LENGTHS:
            e = np.searchsorted(d, d[s] + L)
            if e >= len(ref):
                continue
            dref = np.linalg.inv(ref[s]) @ ref[e]
            dest = np.linalg.inv(est[s]) @ est[e]
            err = np.linalg.inv(dest) @ dref
            t_err.append(np.linalg.norm(err[:3, 3]) / L)
            ang = np.arccos(np.clip((np.trace(err[:3, :3]) - 1) / 2, -1, 1))
            r_err.append(np.degrees(ang) / L)
    if not t_err:
        return np.nan, np.nan
    return 100 * np.mean(t_err), 100 * np.mean(r_err)


def scale_cv(ref_p, est_p, seg=10.0):
    d = path_dist(ref_p)
    ratios, s = [], 0
    while True:
        e = np.searchsorted(d, d[s] + seg)
        if e >= len(ref_p):
            break
        dr = np.linalg.norm(ref_p[e] - ref_p[s])
        if dr > 0.5 * seg:  # skip segments dominated by turning on the spot
            ratios.append(np.linalg.norm(est_p[e] - est_p[s]) / dr)
        s = e
    if len(ratios) < 3:
        return np.nan
    r = np.array(ratios)
    return 100 * r.std() / r.mean()


def rpe(ref, est, delta, rel):
    m = metrics.RPE(rel, delta=delta, delta_unit=metrics.Unit.meters, all_pairs=False)
    try:
        m.process_data((ref, est))
        return m.get_statistic(metrics.StatisticsType.rmse)
    except Exception:
        return np.nan


def loops(it):
    f = os.path.join(it, "logs", "loop_closing.log")
    if not os.path.isfile(f):
        return np.nan, np.nan
    txt = open(f, errors="ignore").read()
    return txt.count("detected loop candidates"), txt.count("LoopCorrector: optimizing pose graph")


def tracking_times(it):
    """Per-frame tracking time (pySLAM 'Tracking: elapsed_time') split by tracking state, from logs/log.txt."""
    f = os.path.join(it, "logs", "log.txt")
    if not os.path.isfile(f):
        return {}
    ev = re.findall(r"Tracking: elapsed_time:\s+([\d.]+).*?state: (\w+)", open(f, errors="ignore").read(), flags=re.S)
    if not ev:
        return {}
    el = np.array([float(a) for a, _ in ev]); st = np.array([b for _, b in ev])
    ok = st == "OK"
    return dict(track_ms_ok_median=1000 * np.median(el[ok]) if ok.any() else np.nan,
                track_ms_reloc_median=1000 * np.median(el[~ok]) if (~ok).any() else np.nan,
                tracking_time_share_pct=100 * el.sum())  # divided by runtime below


def evaluate(r):
    ds = os.path.join(DATA, f"gsaf_{r['sequence']}")
    meta = json.load(open(os.path.join(ds, "meta.json")))
    row = dict(sequence=r["sequence"], method=r["method"], rep=r["rep"], crop=meta["crop"], bbch=meta["bbch"],
               speed=meta["speed"], illumination=meta["illumination"], rig=meta["rig"], frames=meta["frames"],
               path_length_m=round(meta["path_length_m"], 1), status="crashed", run_dir=r["path"])
    info_p = os.path.join(r["path"], "other_metrics_info.txt")
    if not os.path.isfile(info_p):
        return row
    info = dict(l.split(": ", 1) for l in open(info_p).read().splitlines() if ": " in l)
    if "run_exception" in info and int(info.get("num_processed_frames", 0)) == 0:
        row["error"] = info.get("run_exception_message", "")
        return row
    processed, lost = int(info["num_processed_frames"]), int(info["num_lost_frames"])
    row.update(status="completed" if processed >= meta["frames"] else "stopped",
               processed_frames=processed, lost_frames=lost,
               fps=round(processed / max(float(info["runtime_seconds"]), 1e-9), 2))
    row["loop_detections"], row["loop_corrections"] = loops(r["path"])
    tt = tracking_times(r["path"])
    if tt:
        tt["tracking_time_share_pct"] /= max(float(info["runtime_seconds"]), 1e-9)
        row.update({k: round(v, 2) for k, v in tt.items()})
    traj_p = os.path.join(r["path"], "trajectory_online.txt")
    if not os.path.isfile(traj_p) or os.path.getsize(traj_p) == 0:
        row["tracked_pct"] = 0.0
        return row
    ref = file_interface.read_tum_trajectory_file(os.path.join(ds, "groundtruth_cam.tum"))
    est = file_interface.read_tum_trajectory_file(traj_p)
    t0 = ref.timestamps[0]
    row["init_delay_s"] = round(float(est.timestamps[0] - t0), 2)
    ref_a, est_a = sync.associate_trajectories(ref, est, max_diff=MAX_DT)
    n = est_a.num_poses
    row["poses_associated"] = n
    # tracked = frames in state OK over all frames of the sequence (frames after a stop count as failures).
    # Exact per-frame states from pySLAM frame_log.csv (PYSLAM_EXPORT_FRAME_LOGS=1); otherwise processed frames
    # minus LOST frames minus the STOP_LOST lost/relocalizing frames that triggered the stop.
    fl = glob.glob(os.path.join(r["path"], "**", "frame_log.csv"), recursive=True)
    if fl:
        st = pd.read_csv(fl[0])["tracking_state"]
        ok = int((st == "OK").sum())
        row["relocalizing_frames"] = int((st == "RELOCALIZING").sum())
        row["lost_episodes"] = int(((st != "OK") & (st.shift(fill_value="OK") == "OK") & (st != "INIT")).sum())
        row["tracked_source"] = "frame_log"
    else:
        ok = processed - lost - (STOP_LOST if row["status"] == "stopped" else 0)
        row["tracked_source"] = "approx"
    row["tracked_pct"] = round(100.0 * max(ok, 0) / meta["frames"], 2)
    row["success"] = int(row["status"] == "completed" and row["tracked_pct"] >= SUCCESS_TRACKED)
    if n < 20:
        return row
    raw_p = est_a.positions_xyz.copy()
    est_al = PoseTrajectory3D(poses_se3=est_a.poses_se3, timestamps=est_a.timestamps)
    _, _, s = est_al.align(ref_a, correct_scale=True)
    row["sim3_scale"] = round(float(s), 4)
    ape = metrics.APE(metrics.PoseRelation.translation_part)
    ape.process_data((ref_a, est_al))
    st = ape.get_all_statistics()
    ref_len = float(path_dist(ref_a.positions_xyz)[-1])
    row.update(ate_rmse_m=st["rmse"], ate_median_m=st["median"], ate_max_m=st["max"],
               ate_rmse_pct=100 * st["rmse"] / max(ref_len, 1e-9), evaluated_length_m=round(ref_len, 1))
    for d in (1, 10):
        row[f"rpe_t_{d}m_m"] = rpe(ref_a, est_al, d, metrics.PoseRelation.translation_part)
        row[f"rpe_r_{d}m_deg"] = rpe(ref_a, est_al, d, metrics.PoseRelation.rotation_angle_deg)
    row["kitti_t_rel_pct"], row["kitti_r_rel_deg_100m"] = kitti_drift(ref_a.poses_se3, est_al.poses_se3)
    row["scale_drift_cv_pct"] = scale_cv(ref_a.positions_xyz, raw_p)
    row["_ref_xy"], row["_est_xy"] = ref_a.positions_xyz[:, :2], est_al.positions_xyz[:, :2]
    return row


def plot(rows, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    for seq, g in pd.DataFrame(rows).groupby("sequence"):
        ds = os.path.join(DATA, f"gsaf_{seq}")
        gt = np.loadtxt(os.path.join(ds, "groundtruth_cam.tum"))[:, 1:3]
        fig, ax = plt.subplots(figsize=(4.2, 6))
        ax.plot(gt[:, 0], gt[:, 1], "k-", lw=2.2, label="RTK-GNSS/INS (camera)")
        for _, r in g[g["rep"] == g["rep"].min()].iterrows():
            if isinstance(r.get("_est_xy"), np.ndarray):
                ax.plot(r["_est_xy"][:, 0], r["_est_xy"][:, 1], lw=1.1,
                        label=f"{r['method']} ({r['tracked_pct']:.0f} % tracked)")
        ax.set_aspect("equal"); ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]"); ax.set_title(seq, fontsize=9)
        ax.legend(fontsize=7, loc="best"); ax.grid(alpha=.3); fig.tight_layout()
        for ext in ("png", "pdf"):
            fig.savefig(os.path.join(out, f"traj_{seq}.{ext}"), dpi=200)
        plt.close(fig)


def check(tag, seq, method, rep):
    """Exit status 0 if the most recent run of (seq, method, rep) is a success (used by run_all.sh)."""
    runs = find_runs(tag)
    r = runs[(runs.sequence == seq) & (runs.method == method) & (runs.rep == rep)]
    if r.empty:
        return 2
    row = evaluate(r.iloc[0])
    print(f"{seq} {method} attempt {rep}: {row['status']} tracked {row.get('tracked_pct', 0)} % -> "
          f"{'SUCCESS' if row.get('success') else 'FAILURE'}")
    return 0 if row.get("success") else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", nargs=3, metavar=("SEQ", "METHOD", "REP"))
    ap.add_argument("--tag", default="gsaf_benchmark")
    ap.add_argument("--out", default=None)
    ap.add_argument("--plot", action="store_true")
    a = ap.parse_args()
    if a.check:
        raise SystemExit(check(a.tag, a.check[0], a.check[1], int(a.check[2])))
    out = a.out or os.path.join(B, "results" if a.tag == "gsaf_benchmark" else a.tag)
    os.makedirs(out, exist_ok=True)
    runs = find_runs(a.tag)
    rows = [evaluate(r) for _, r in runs.iterrows()]
    df = pd.DataFrame(rows)
    df.drop(columns=[c for c in df.columns if c.startswith("_")]).to_csv(os.path.join(out, "runs.csv"), index=False)
    if a.plot:
        plot(rows, out)
    cols = ["sequence", "method", "rep", "status", "tracked_pct", "loop_corrections", "ate_rmse_m", "ate_rmse_pct",
            "rpe_r_1m_deg", "kitti_t_rel_pct", "kitti_r_rel_deg_100m", "scale_drift_cv_pct", "track_ms_ok_median", "fps"]
    print(df[[c for c in cols if c in df.columns]].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
