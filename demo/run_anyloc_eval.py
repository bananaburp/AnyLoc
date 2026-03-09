#!/usr/bin/env python3
"""
run_anyloc_eval.py
End-to-end AnyLoc-VLAD-DINOv2 recall evaluation on a prepared dataset.

Usage:

    # floor_2 cross-run eval (macOS, default paths)
    /Users/Aryan/miniforge-arm64/envs/anyloc-arm64/bin/python demo/run_anyloc_eval.py \
        --db_dir    /Volumes/T9/DevSpace/Github/hilti-trimble-slam-challenge-2026/challenge_tools_ros/vpr/data/floor_2_2025-05-05_run_1/eval/mixvpr_evalset_2m/db_full_cam0 \
        --query_dir /Volumes/T9/DevSpace/Github/hilti-trimble-slam-challenge-2026/challenge_tools_ros/vpr/data/floor_2_2025-10-28_run_1/eval/mixvpr_evalset_2m/query_full_cam0 \
        --gt_path   /Volumes/T9/DevSpace/Github/hilti-trimble-slam-challenge-2026/challenge_tools_ros/vpr/data/floor_2_2025-10-28_run_1/eval/mixvpr_evalset_2m/gt_positives.npy \
        --domain    indoor \
        --out_dir   results/floor_2_cross_run \
        --device    mps \
        --batch_size 1 --max_img_size 224 --max_images 20

    # Windows PowerShell (recommended: GPU + fp16 for speed)
    python demo/run_anyloc_eval.py `
            --db_dir    'path\to\floor_2_2025-05-05_run_1\eval\mixvpr_evalset_2m\db_full_cam0' `
            --query_dir 'path\to\floor_2_2025-10-28_run_1\eval\mixvpr_evalset_2m\query_full_cam0' `
            --gt_path   'path\to\floor_2_2025-10-28_run_1\eval\mixvpr_evalset_2m\gt_positives.npy' `
            --domain indoor `
            --out_dir results/floor_2_cross_run `
            --device cuda --fp16 --use_gpu_index

    # Quick test (limit images)
    python demo/run_anyloc_eval.py `
            --db_dir    'path\to\db_full_cam0' `
            --query_dir 'path\to\query_full_cam0' `
            --gt_path   'path\to\gt_positives.npy' `
            --domain indoor `
            --out_dir results/floor_2_quick_test `
            --max_img_size 224 `
            --max_images 20

Note: --gt_path is a single cross-run file mapping each query index to its
      positive DB indices. Only one GT file is needed (not one per split).
"""

import os, glob, argparse, warnings
from concurrent.futures import ThreadPoolExecutor
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
    # Support mixed formats and nested folders common in custom eval sets.
    exts = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
    folder_path = Path(folder)

    try:
        paths = sorted(
            [str(p) for p in folder_path.rglob("*")
             if p.is_file() and p.suffix.lower() in exts],
            key=lambda p: os.path.basename(p).lower())
    except PermissionError as e:
        raise PermissionError(
            f"Access denied while reading '{folder}'. "
            "If this is a network/UNC path, ensure this shell has permission "
            "to the share (or copy/map the dataset locally and pass that path)."
        ) from e

    if not paths:
        if not folder_path.exists():
            raise FileNotFoundError(f"Folder not found: {folder}")
        try:
            present_exts = sorted({p.suffix.lower() for p in folder_path.rglob("*")
                                  if p.is_file() and p.suffix})
        except PermissionError as e:
            raise PermissionError(
                f"Access denied while reading '{folder}'. "
                "If this is a network/UNC path, ensure this shell has permission "
                "to the share (or copy/map the dataset locally and pass that path)."
            ) from e
        raise AssertionError(
            f"No supported images found in {folder}. "
            f"Supported: {exts}. Present extensions: {present_exts[:20]}"
        )
    return paths


def _preprocess_image(p, base_tf, max_img_size):
    """Load, optionally downscale, and crop one image to a 14-px boundary."""
    img = base_tf(Image.open(p).convert("RGB"))
    _, h, w = img.shape
    if max(h, w) > max_img_size:
        scale = max_img_size / max(h, w)
        h, w  = int(h * scale), int(w * scale)
        img   = tvf.functional.resize(
            img, (h, w), interpolation=tvf.InterpolationMode.BICUBIC)
    h_new, w_new = (img.shape[1] // 14) * 14, (img.shape[2] // 14) * 14
    return tvf.CenterCrop((h_new, w_new))(img)


def extract_vlad_descriptors(img_paths, extractor, vlad,
                              device, max_img_size=1024, desc="",
                              use_fp16=False, batch_size=8):
    base_tf = tvf.Compose([
        tvf.ToTensor(),
        tvf.Normalize(mean=[0.485, 0.456, 0.406],
                      std=[0.229, 0.224, 0.225])
    ])
    use_autocast = use_fp16 and device.type == "cuda"
    all_gd = []

    # Parallel image loading: threads load/decode/crop the next batch
    # on CPU while the GPU processes the current one.
    n_workers = min(batch_size, 8)
    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        for i in tqdm(range(0, len(img_paths), batch_size), desc=desc, leave=False):
            imgs = list(pool.map(
                lambda p: _preprocess_image(p, base_tf, max_img_size),
                img_paths[i:i + batch_size]))

            # If all images share the same spatial size run one batched
            # forward pass; otherwise fall back to per-image.
            if len({img.shape for img in imgs}) == 1:
                batch = torch.stack(imgs).to(device)      # [B, C, H, W]
                with torch.no_grad():
                    with torch.cuda.amp.autocast(enabled=use_autocast):
                        ret = extractor(batch)             # [B, n_patches, 1536]
                ret_cpu = ret.cpu().float()
                for j in range(len(imgs)):
                    all_gd.append(vlad.generate(ret_cpu[j]).numpy())
            else:
                for img in imgs:
                    with torch.no_grad():
                        with torch.cuda.amp.autocast(enabled=use_autocast):
                            ret = extractor(img[None].to(device))
                    all_gd.append(vlad.generate(ret.cpu().float().squeeze()).numpy())

    return np.stack(all_gd, axis=0)                   # [N, agg_dim]


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
    ap.add_argument("--dataset_dir",  default=None,
                    help="Root of the eval set (unused when --db_dir / --query_dir are set)")
    ap.add_argument("--db_dir",    required=True,
                    help="Path to database images folder")
    ap.add_argument("--query_dir", required=True,
                    help="Path to query images folder")
    ap.add_argument("--gt_path",   required=True,
                    help="Path to ground-truth .npy file")
    ap.add_argument("--domain",       default="indoor",
                    choices=["indoor", "urban", "aerial", "custom_floor1"])
    ap.add_argument("--num_clusters", type=int, default=32)
    ap.add_argument("--cache_dir",    default=None,
                    help="Override default demo/cache path")
    ap.add_argument("--out_dir",      default="./results")
    ap.add_argument("--max_img_size", type=int, default=1024)
    ap.add_argument("--top_k",        nargs="+", type=int,
                    default=[1, 5, 10])
    ap.add_argument("--batch_size",   type=int, default=8,
                    help="Images per DINOv2 forward pass (increase for faster GPU use)")
    ap.add_argument("--use_gpu_index",action="store_true")
    ap.add_argument("--fp16",         action="store_true",
                    help="Use fp16 autocast for faster GPU inference (CUDA only)")
    ap.add_argument("--device", default="mps",
                    choices=["auto", "cuda", "mps", "cpu"],
                    help="Force a specific device (default: auto-detect)")
    ap.add_argument("--max_images", type=int, default=None,
                    help="Limit DB and query to first N images (for quick tests)")
    ap.add_argument("--recompute",  action="store_true",
                    help="Ignore cached descriptors and recompute from scratch")
    ap.add_argument("--no_compile", action="store_true",
                    help="Skip torch.compile (recommended for small datasets < ~500 images)")
    args = ap.parse_args()

    if args.device == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        if args.device == "cuda" and not torch.cuda.is_available():
            if torch.backends.mps.is_available():
                print("WARNING: --device cuda requested but CUDA not available, falling back to MPS", flush=True)
                device = torch.device("mps")
            else:
                print("WARNING: --device cuda requested but CUDA not available, falling back to CPU", flush=True)
                device = torch.device("cpu")
        else:
            device = torch.device(args.device)
    if args.fp16 and device.type != "cuda":
        print("WARNING: --fp16 has no effect outside CUDA, ignoring", flush=True)
    print(f"Using device: {device}  fp16={args.fp16 and device.type == 'cuda'}", flush=True)
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

    # ── Images ──────────────────────────────────────────────────────────────
    db_paths = load_images_sorted(args.db_dir)
    qu_paths = load_images_sorted(args.query_dir)
    if args.max_images is not None:
        db_paths = db_paths[:args.max_images]
        qu_paths = qu_paths[:args.max_images]
        print(f"[quick-test] Limiting to {args.max_images} images per split", flush=True)
    print(f"DB: {len(db_paths)}  Query: {len(qu_paths)}", flush=True)

    # ── Model ───────────────────────────────────────────────────────────────
    print("Loading DINOv2 ViT-G/14 model (may download weights on first run)...", flush=True)
    print(f"[DEBUG] torch hub cache dir: {torch.hub.get_dir()}", flush=True)
    print(f"[DEBUG] Weights cache: {os.path.join(torch.hub.get_dir(), 'checkpoints')}", flush=True)
    import glob as _glob
    _ckpts = _glob.glob(os.path.join(torch.hub.get_dir(), "checkpoints", "*vitg14*"))
    print(f"[DEBUG] Existing vitg14 checkpoints: {_ckpts}", flush=True)
    extractor = DinoV2ExtractFeatures(
        "dinov2_vitg14", layer=31, facet="value", device=device)
    total_images = len(db_paths) + len(qu_paths)
    _can_compile = (
        hasattr(torch, "compile")
        and device.type == "cuda"
        and torch.cuda.get_device_capability(device)[0] >= 7
        and not args.no_compile
        and total_images >= 500   # compilation overhead ~10-20 min; not worth it for small sets
    )
    if args.no_compile:
        print("torch.compile skipped (--no_compile)", flush=True)
    elif total_images < 500:
        print(f"torch.compile skipped (only {total_images} images; overhead outweighs benefit)", flush=True)
    elif _can_compile:
        extractor.dino_model = torch.compile(extractor.dino_model)
        print("torch.compile applied to DINOv2 (first batch will be slow)", flush=True)
    else:
        print("torch.compile skipped (requires CUDA capability >= 7.0)", flush=True)
    print("DINOv2 ViT-G/14  layer=31  facet=value  ready", flush=True)

    # ── Descriptors (with caching) ───────────────────────────────────────────
    # Cache key encodes everything that affects descriptor values.
    desc_cache_key = (f"vitg14_l31_value"
                      f"_c{args.num_clusters}_{args.domain}"
                      f"_s{args.max_img_size}")
    desc_cache_dir = os.path.join(args.out_dir, "desc_cache")
    os.makedirs(desc_cache_dir, exist_ok=True)
    db_cache  = os.path.join(desc_cache_dir, f"db_{desc_cache_key}.npy")
    qu_cache  = os.path.join(desc_cache_dir, f"qu_{desc_cache_key}.npy")

    if os.path.isfile(db_cache) and not args.recompute:
        print(f"Loading cached DB descriptors from {db_cache}", flush=True)
        db_descs = np.load(db_cache)
    else:
        print(f"Extracting DB descriptors ({len(db_paths)} images)...", flush=True)
        db_descs = extract_vlad_descriptors(
            db_paths, extractor, vlad, device,
            args.max_img_size, desc="DB   ", use_fp16=args.fp16,
            batch_size=args.batch_size)
        np.save(db_cache, db_descs)
        print(f"DB descriptors cached → {db_cache}", flush=True)
    print(f"DB descriptors ready. Shape: {db_descs.shape}", flush=True)

    if os.path.isfile(qu_cache) and not args.recompute:
        print(f"Loading cached query descriptors from {qu_cache}", flush=True)
        qu_descs = np.load(qu_cache)
    else:
        print(f"Extracting query descriptors ({len(qu_paths)} images)...", flush=True)
        qu_descs = extract_vlad_descriptors(
            qu_paths, extractor, vlad, device,
            args.max_img_size, desc="Query", use_fp16=args.fp16,
            batch_size=args.batch_size)
        np.save(qu_cache, qu_descs)
        print(f"Query descriptors cached → {qu_cache}", flush=True)
    print(f"Query descriptors ready. Shape: {qu_descs.shape}", flush=True)

    # ── Recall ──────────────────────────────────────────────────────────────
    print("Computing recall...", flush=True)
    gt_anyloc  = np.load(args.gt_path, allow_pickle=True)
    if gt_anyloc.ndim == 2:
        gt_pos = gt_anyloc[:, 1]   # shape [n_qu,] — (N,2) format with col0=query_idx
    else:
        gt_pos = gt_anyloc         # shape [n_qu,] — already array-of-arrays
    if args.max_images is not None:
        # Slice to match the truncated query set, and drop any GT positive
        # indices that fall outside the truncated DB range so the recall
        # denominator only counts queries that can actually be retrieved.
        gt_pos = gt_pos[:args.max_images].copy()
        for i in range(len(gt_pos)):
            gt_pos[i] = gt_pos[i][gt_pos[i] < args.max_images]

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
        bar = "#" * int(v / 2)   # 50 hashes = 100 %
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
