#!/usr/bin/env python3
"""
run_anyloc_eval.py
End-to-end AnyLoc-VLAD-DINOv2 recall evaluation on a prepared dataset.

Usage:
  python demo/run_anyloc_eval.py \
      --dataset_dir  datasets_vg/datasets/hilti/floor_1 \
      --domain       indoor \
      --out_dir      results/floor_1
"""

import os, glob, argparse, warnings
warnings.filterwarnings("ignore")
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms as tvf
from PIL import Image
from tqdm import tqdm
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# AnyLoc imports (run from AnyLoc repo root)
import sys
import importlib.util as _ilu
sys.path.insert(0, str(Path(__file__).parent))
from utilities import DinoV2ExtractFeatures, VLAD
_root_utils_spec = _ilu.spec_from_file_location(
    "root_utilities", Path(__file__).parent.parent / "utilities.py")
_root_utils = _ilu.module_from_spec(_root_utils_spec)
_root_utils_spec.loader.exec_module(_root_utils)
get_top_k_recall = _root_utils.get_top_k_recall


def load_images_sorted(folder: str):
    paths = sorted(glob.glob(f"{folder}/*.jpg"),
                   key=lambda p: os.path.basename(p))
    assert paths, f"No .jpg found in {folder}"
    return paths


def extract_vlad_descriptors(img_paths, extractor, vlad,
                              device, max_img_size=1024, desc=""):
    base_tf = tvf.Compose([
        tvf.ToTensor(),
        tvf.Normalize(mean=[0.485, 0.456, 0.406],
                      std=[0.229, 0.224, 0.225])
    ])
    all_gd = []
    for p in tqdm(img_paths, desc=desc, leave=False):
        with torch.no_grad():
            img_pt = base_tf(Image.open(p).convert("RGB")).to(device)
            c, h, w = img_pt.shape
            # Downscale if needed to avoid OOM
            if max(h, w) > max_img_size:
                scale  = max_img_size / max(h, w)
                h, w   = int(h * scale), int(w * scale)
                img_pt = tvf.functional.resize(
                    img_pt, (h, w),
                    interpolation=tvf.InterpolationMode.BICUBIC)
            # Crop to 14-patch boundary
            h_new, w_new = (h // 14) * 14, (w // 14) * 14
            img_pt = tvf.CenterCrop((h_new, w_new))(img_pt)[None]
            ret    = extractor(img_pt)              # [1, n_patches, 1536]
        gd = vlad.generate(ret.cpu().squeeze())     # [n_clusters * 1536]
        all_gd.append(gd.numpy())
    return np.stack(all_gd, axis=0)                 # [N, agg_dim]


def save_top1_grid(qu_paths, db_paths, indices, out_path, n_show=20):
    n = min(n_show, len(qu_paths))
    fig = plt.figure(figsize=(6, n * 3))
    gs  = gridspec.GridSpec(n, 2, figure=fig,
                            hspace=0.05, wspace=0.05)
    for i in range(n):
        ax_q = fig.add_subplot(gs[i, 0])
        ax_r = fig.add_subplot(gs[i, 1])
        ax_q.imshow(Image.open(qu_paths[i]))
        ax_r.imshow(Image.open(db_paths[int(indices[i, 0])]))
        ax_q.axis("off"); ax_r.axis("off")
        if i == 0:
            ax_q.set_title("Query", fontsize=9)
            ax_r.set_title("Top-1 Retrieval", fontsize=9)
    plt.savefig(out_path, dpi=100, bbox_inches="tight")
    plt.close()
    print(f"Saved retrieval grid → {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_dir",  required=True,
                    help="Root produced by prepare_custom_dataset.py")
    ap.add_argument("--domain",       default="indoor",
                    choices=["indoor", "urban", "aerial", "custom_floor1"])
    ap.add_argument("--num_clusters", type=int, default=32)
    ap.add_argument("--cache_dir",    default=None,
                    help="Override default demo/cache path")
    ap.add_argument("--out_dir",      default="./results")
    ap.add_argument("--max_img_size", type=int, default=1024)
    ap.add_argument("--top_k",        nargs="+", type=int,
                    default=[1, 5, 10])
    ap.add_argument("--use_gpu_index",action="store_true")
    ap.add_argument("--device", default="auto",
                    choices=["auto", "cuda", "mps", "cpu"],
                    help="Force a specific device (default: auto-detect)")
    args = ap.parse_args()

    if args.device != "auto":
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}", flush=True)
    demo_dir  = Path(__file__).parent
    cache_dir = args.cache_dir or str(demo_dir / "cache")

    # ── Vocabulary ──────────────────────────────────────────────────────────
    ext_spec    = f"dinov2_vitg14/l31_value_c{args.num_clusters}"
    c_ctr_file  = os.path.join(cache_dir, "vocabulary",
                               ext_spec, args.domain, "c_centers.pt")
    assert os.path.isfile(c_ctr_file), \
        f"Vocabulary not found: {c_ctr_file}\nRun the cache download first."

    print(f"Loading VLAD vocabulary ({args.num_clusters} clusters)...", flush=True)
    vlad = VLAD(args.num_clusters, desc_dim=None,
                cache_dir=os.path.dirname(c_ctr_file))
    vlad.fit(None)    # loads c_centers.pt from cache_dir
    print(f"Loaded VLAD vocabulary: {c_ctr_file}", flush=True)

    # ── Model ───────────────────────────────────────────────────────────────
    print("Loading DINOv2 ViT-G/14 model (may download weights on first run)...", flush=True)
    print(f"[DEBUG] torch hub cache dir: {torch.hub.get_dir()}", flush=True)
    print(f"[DEBUG] Weights cache: {os.path.join(torch.hub.get_dir(), 'checkpoints')}", flush=True)
    import glob as _glob
    _ckpts = _glob.glob(os.path.join(torch.hub.get_dir(), "checkpoints", "*vitg14*"))
    print(f"[DEBUG] Existing vitg14 checkpoints: {_ckpts}", flush=True)
    extractor = DinoV2ExtractFeatures(
        "dinov2_vitg14", layer=31, facet="value", device=device)
    print("DINOv2 ViT-G/14  layer=31  facet=value  ready", flush=True)

    # ── Images ──────────────────────────────────────────────────────────────
    db_dir = os.path.join(args.dataset_dir, "images", "test", "database")
    qu_dir = os.path.join(args.dataset_dir, "images", "test", "queries")
    db_paths = load_images_sorted(db_dir)
    qu_paths = load_images_sorted(qu_dir)
    print(f"DB: {len(db_paths)}  Query: {len(qu_paths)}", flush=True)

    # ── Descriptors ─────────────────────────────────────────────────────────
    print(f"Extracting DB descriptors ({len(db_paths)} images)...", flush=True)
    db_descs = extract_vlad_descriptors(
        db_paths, extractor, vlad, device,
        args.max_img_size, desc="DB   ")
    print(f"DB descriptors done. Shape: {db_descs.shape}", flush=True)

    print(f"Extracting query descriptors ({len(qu_paths)} images)...", flush=True)
    qu_descs = extract_vlad_descriptors(
        qu_paths, extractor, vlad, device,
        args.max_img_size, desc="Query")
    print(f"Query descriptors done. Shape: {qu_descs.shape}", flush=True)

    # ── Recall ──────────────────────────────────────────────────────────────
    print("Computing recall...", flush=True)
    gt_path    = os.path.join(args.dataset_dir, "ground_truth_new.npy")
    gt_anyloc  = np.load(gt_path, allow_pickle=True)
    gt_pos     = gt_anyloc[:, 1]   # shape [n_qu,] of int arrays

    db_t = torch.from_numpy(db_descs.astype(np.float32))
    qu_t = torch.from_numpy(qu_descs.astype(np.float32))

    distances, indices, recalls = get_top_k_recall(
        top_k      = args.top_k,
        db         = db_t,
        qu         = qu_t,
        gt_pos     = gt_pos,
        method     = "cosine",
        norm_descs = True,
        use_gpu    = args.use_gpu_index,
        use_percentage = True,
    )

    # ── Print results ────────────────────────────────────────────────────────
    print("\n" + "="*40)
    print(f"  AnyLoc-VLAD-DINOv2 Recall  [{Path(args.dataset_dir).name}]")
    print(f"  Domain: {args.domain}   Clusters: {args.num_clusters}")
    print("="*40)
    for k, v in sorted(recalls.items()):
        bar = "#" * int(v / 2)
        print(f"  R@{k:<3d}: {v:6.2f}%  {bar}")
    print("="*40)

    # ── Save ─────────────────────────────────────────────────────────────────
    print(f"Saving results to {args.out_dir}/...", flush=True)
    os.makedirs(args.out_dir, exist_ok=True)
    np.save(os.path.join(args.out_dir, "recalls.npy"),     recalls)
    np.save(os.path.join(args.out_dir, "ret_indices.npy"), indices)
    np.save(os.path.join(args.out_dir, "ret_distances.npy"), distances)

    save_top1_grid(
        qu_paths, db_paths, indices,
        out_path=os.path.join(args.out_dir, "top1_retrievals.png"),
        n_show=30
    )

    # Summary text
    with open(os.path.join(args.out_dir, "recall_summary.txt"), "w") as f:
        f.write(f"Dataset: {args.dataset_dir}\n")
        f.write(f"Domain:  {args.domain}   Clusters: {args.num_clusters}\n\n")
        for k, v in sorted(recalls.items()):
            f.write(f"R@{k}: {v:.4f}%\n")
    print(f"Results saved to {args.out_dir}/", flush=True)


if __name__ == "__main__":
    main()
