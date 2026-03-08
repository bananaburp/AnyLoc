#!/usr/bin/env python3
"""
prepare_custom_dataset.py
Prepares an AnyLoc-ready dataset from either:
  (a) a raw TUM pose file + image directory (full pipeline), or
  (b) an existing mixvpr_evalset/ directory (conversion only).

Usage examples:
  # From existing evalset (most common case):
  python ./scripts_hilti/prepare_custom_dataset.py \
      --evalset_dir  /Volumes/T9/DevSpace/Github/hilti-trimble-slam-challenge-2026/challenge_tools_ros/vpr/data/floor_1_2025-05-05_run_1/eval/mixvpr_evalset \
      --out_dir      datasets_vg/datasets/hilti/floor_1 \
      --undistort

  # From raw TUM + images (fresh split):
  python ./scripts_hilti/prepare_custom_dataset.py \
      --images_dir   vpr/data/floor_1_2025-05-05_run_1/raw_frames/cam0 \
      --gt_tum       path/to/groundtruth.txt \
      --out_dir      anyloc_datasets/floor_1 \
      --db_frac 0.70 --pos_dist_m 5.0 --undistort
"""

import os, shutil, argparse, csv, glob
import numpy as np
import cv2
from pathlib import Path
from tqdm import tqdm

# ── Fisheye calibration ─────────────────────────────────────────────────────
# Replace with your actual calibration. The script exits with a clear error
# if --undistort is requested but calibration is not filled in.
CAM0_K = np.array([[461.6398879418857,   0.0,               732.9460954720965],
                   [  0.0,               459.7153043295965, 720.5410566713475],
                   [  0.0,               0.0,               1.0             ]], dtype=np.float64)
CAM0_D = np.array([[0.03442918444998219], [-0.02155491263917851],
                   [0.0031292637056308044], [-0.0005356957576266091]], dtype=np.float64)
CAL_INPUT_SIZE  = (1472, 1440)  # (W, H) as calibrated
CAL_OUTPUT_SIZE = (392,  392)   # DINOv2-friendly (divisible by 14)
CALIB_FILLED    = True          # kalibr_imucam_chain.yaml cam0 equidistant
# ────────────────────────────────────────────────────────────────────────────


def build_undistort_maps():
    Knew = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
        CAM0_K, CAM0_D, CAL_INPUT_SIZE, np.eye(3),
        balance=0.0, new_size=CAL_OUTPUT_SIZE, fov_scale=1.0)
    map1, map2 = cv2.fisheye.initUndistortRectifyMap(
        CAM0_K, CAM0_D, np.eye(3), Knew, CAL_OUTPUT_SIZE, cv2.CV_16SC2)
    return map1, map2


def process_image(src: str, dst: str, map1, map2, undistort: bool):
    img = cv2.imread(src)
    if undistort:
        img = cv2.remap(img, map1, map2,
                        interpolation=cv2.INTER_LINEAR,
                        borderMode=cv2.BORDER_CONSTANT)
    img = cv2.rotate(img, cv2.ROTATE_180)
    cv2.imwrite(dst, img, [cv2.IMWRITE_JPEG_QUALITY, 95])


# ── TUM helpers (from 1_align_frames_to_gt.py) ──────────────────────────────
def read_tum(path):
    t, p, q = [], [], []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 8:
                continue
            tt, tx, ty, tz, qx, qy, qz, qw = parts[:8]
            t.append(float(tt)); p.append([float(tx), float(ty), float(tz)])
            q.append([float(qx), float(qy), float(qz), float(qw)])
    t = np.asarray(t); p = np.asarray(p); q = np.asarray(q)
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    return t, p, q


def slerp(q0, q1, a):
    dot = float(np.clip(np.dot(q0, q1), -1, 1))
    if dot < 0:
        q1, dot = -q1, -dot
    if dot > 0.9995:
        return (q0 + a * (q1 - q0)) / np.linalg.norm(q0 + a * (q1 - q0))
    th0 = np.arccos(dot); sth0 = np.sin(th0); th = th0 * a
    return (np.sin(th0 - th) * q0 + np.sin(th) * q1) / sth0


def interp_pose(gt_t, gt_p, gt_q, t):
    j = int(np.searchsorted(gt_t, t))
    if j <= 0:   return gt_p[0], gt_q[0]
    if j >= len(gt_t): return gt_p[-1], gt_q[-1]
    dt = gt_t[j] - gt_t[j-1]
    a = 0.0 if dt <= 0 else (t - gt_t[j-1]) / dt
    return (1-a)*gt_p[j-1] + a*gt_p[j], slerp(gt_q[j-1], gt_q[j], a)


def align_images_to_tum(images_dir, gt_tum):
    """Returns list of (ts_ns, t_s, x, y, z, qx, qy, qz, qw, img_path)."""
    imgs = sorted([p for p in Path(images_dir).glob("*.jpg") if not p.name.startswith("._")],
                  key=lambda p: int(p.stem))
    gt_t, gt_p, gt_q = read_tum(gt_tum)
    rows = []
    for ip in imgs:
        ts_ns = int(ip.stem)
        t_s   = ts_ns * 1e-9
        if t_s < gt_t[0] or t_s > gt_t[-1]:
            continue
        pos, quat = interp_pose(gt_t, gt_p, gt_q, t_s)
        rows.append((ts_ns, t_s, *pos, *quat, str(ip)))
    return rows


def split_and_gt(rows, db_frac, skip_first_s, pos_dist_m):
    rows = sorted(rows, key=lambda r: r[1])
    t0   = rows[0][1]
    rows = [r for r in rows if r[1] >= t0 + skip_first_s]
    n_db = int(len(rows) * db_frac)
    db_rows, q_rows = rows[:n_db], rows[n_db:]

    db_xy = np.array([[r[2], r[3]] for r in db_rows], dtype=np.float32)
    q_xy  = np.array([[r[2], r[3]] for r in q_rows],  dtype=np.float32)
    gt = np.empty(len(q_rows), dtype=object)
    for i in range(len(q_rows)):
        gt[i] = np.where(np.linalg.norm(db_xy - q_xy[i], axis=1) <= pos_dist_m)[0].astype(np.int32)
    return db_rows, q_rows, gt


# ── AnyLoc gt_positives → ground_truth_new ─────────────────────────────────
def convert_gt(gt_positives: np.ndarray) -> np.ndarray:
    """
    gt_positives: shape [n_qu,] dtype=object  (existing mixvpr format)
    returns:      shape [n_qu, 2] dtype=object (AnyLoc VPR-Bench format)
    """
    n = len(gt_positives)
    out = np.empty((n, 2), dtype=object)
    for i, pos in enumerate(gt_positives):
        out[i, 0] = i
        out[i, 1] = np.asarray(pos, dtype=np.int32)
    return out


# ── Main ────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir",      required=True)
    ap.add_argument("--undistort",    action="store_true")
    # Mode A: existing evalset
    ap.add_argument("--evalset_dir",  default=None,
                    help="Path to existing mixvpr_evalset/ directory")
    # Mode B: fresh split from TUM
    ap.add_argument("--images_dir",   default=None)
    ap.add_argument("--gt_tum",       default=None)
    ap.add_argument("--db_frac",      type=float, default=0.70)
    ap.add_argument("--skip_first_s", type=float, default=5.0)
    ap.add_argument("--pos_dist_m",   type=float, default=5.0)
    args = ap.parse_args()

    if args.undistort and not CALIB_FILLED:
        raise RuntimeError(
            "Set CALIB_FILLED=True and fill CAM0_K / CAM0_D before using --undistort")

    map1 = map2 = None
    if args.undistort:
        map1, map2 = build_undistort_maps()
        print(f"Undistortion maps ready: {CAL_INPUT_SIZE} → {CAL_OUTPUT_SIZE}")

    db_dir = Path(args.out_dir) / "images" / "test" / "database"
    qu_dir = Path(args.out_dir) / "images" / "test" / "queries"
    db_dir.mkdir(parents=True, exist_ok=True)
    qu_dir.mkdir(parents=True, exist_ok=True)

    # ── Mode A ──────────────────────────────────────────────────────────────
    if args.evalset_dir:
        evalset = Path(args.evalset_dir)

        db_srcs = sorted([p for p in (evalset / "db_cam0").glob("*.jpg") if not p.name.startswith("._")],
                          key=lambda p: int(p.stem))
        qu_srcs = sorted([p for p in (evalset / "query_cam0").glob("*.jpg") if not p.name.startswith("._")],
                          key=lambda p: int(p.stem))

        print(f"Copying/undistorting {len(db_srcs)} DB images …")
        for src in tqdm(db_srcs):
            process_image(str(src), str(db_dir / src.name), map1, map2, args.undistort)

        print(f"Copying/undistorting {len(qu_srcs)} query images …")
        for src in tqdm(qu_srcs):
            process_image(str(src), str(qu_dir / src.name), map1, map2, args.undistort)

        gt_pos = np.load(evalset / "gt_positives.npy", allow_pickle=True)

    # ── Mode B ──────────────────────────────────────────────────────────────
    else:
        assert args.images_dir and args.gt_tum, "Provide --images_dir and --gt_tum"
        rows = align_images_to_tum(args.images_dir, args.gt_tum)
        db_rows, q_rows, gt_pos = split_and_gt(
            rows, args.db_frac, args.skip_first_s, args.pos_dist_m)

        print(f"Copying/undistorting {len(db_rows)} DB images …")
        for row in tqdm(db_rows):
            src = row[-1]
            dst = str(db_dir / Path(src).name)
            process_image(src, dst, map1, map2, args.undistort)

        print(f"Copying/undistorting {len(q_rows)} query images …")
        for row in tqdm(q_rows):
            src = row[-1]
            dst = str(qu_dir / Path(src).name)
            process_image(src, dst, map1, map2, args.undistort)

    # ── Write ground_truth_new.npy ──────────────────────────────────────────
    gt_anyloc = convert_gt(gt_pos)
    np.save(Path(args.out_dir) / "ground_truth_new.npy", gt_anyloc)

    n_empty = sum(1 for i in range(len(gt_pos)) if len(gt_pos[i]) == 0)
    print(f"\nDataset written to: {args.out_dir}")
    print(f"  DB:     {len(list(db_dir.glob('*.jpg')))} images")
    print(f"  Query:  {len(list(qu_dir.glob('*.jpg')))} images")
    print(f"  GT:     ground_truth_new.npy  (queries with zero positives: {n_empty})")


if __name__ == "__main__":
    main()
