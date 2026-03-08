# undistort_fisheye.py
#
'''
Usage:
  python undistort_fisheye.py <input_image_or_dir> [output_dir]

Examples:
  python undistort_fisheye.py frame.png                  # saves frame_undist.png
  python undistort_fisheye.py frames/ undistorted/       # processes all images in dir

Before running, update K, D, INPUT_SIZE, and OUTPUT_SIZE below
to match your camera calibration parameters.
'''

import sys
import cv2
import numpy as np
from pathlib import Path

# ── Fill these from your calibration YAML ──────────────────────────────────
# Example values — replace with your actual parameters
K = np.array([[461.6398879418857,   0.0,               732.9460954720965],
              [  0.0,               459.7153043295965, 720.5410566713475],
              [  0.0,               0.0,               1.0             ]], dtype=np.float64)  # fx,fy,cx,cy
D = np.array([[0.03442918444998219], [-0.02155491263917851],
              [0.0031292637056308044], [-0.0005356957576266091]], dtype=np.float64)  # k1..k4 (equidistant)
INPUT_SIZE  = (1472, 1440)   # (width, height) as calibrated
OUTPUT_SIZE = (560, 560)     # see note on resolution below
# ───────────────────────────────────────────────────────────────────────────

# Build the optimal new camera matrix (removes black borders, keeps all pixels)
Knew = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
    K, D, INPUT_SIZE, np.eye(3),
    balance=0.0,           # 0 = crop all black, 1 = keep all pixels
    new_size=OUTPUT_SIZE,
    fov_scale=1.0
)

# Pre-compute the remap LUT once — reuse for every frame
map1, map2 = cv2.fisheye.initUndistortRectifyMap(
    K, D, np.eye(3), Knew,
    OUTPUT_SIZE, cv2.CV_16SC2
)

def undistort(img_bgr: np.ndarray) -> np.ndarray:
    return cv2.remap(img_bgr, map1, map2,
                     interpolation=cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_CONSTANT)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python undistort_fisheye.py <input_image_or_dir> [output_dir]")
        sys.exit(1)

    input_path = Path(sys.argv[1])
    output_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else None

    if input_path.is_dir():
        exts = {".png", ".jpg", ".jpeg", ".bmp", ".tiff"}
        images = [p for p in sorted(input_path.iterdir()) if p.suffix.lower() in exts and not p.name.startswith("._")]
        out_dir = output_dir or input_path
        out_dir.mkdir(parents=True, exist_ok=True)
        print(f"Found {len(images)} image(s) in '{input_path}'. Saving to '{out_dir}'...")
        for img_path in images:
            img = cv2.imread(str(img_path))
            result = undistort(img)
            out_path = out_dir / (img_path.stem + "_undist" + img_path.suffix)
            cv2.imwrite(str(out_path), result)
            print(f"  Saved: {out_path}")
        print(f"Done. Processed {len(images)} image(s).")
    else:
        img = cv2.imread(str(input_path))
        if img is None:
            print(f"Error: could not read '{input_path}'")
            sys.exit(1)
        result = undistort(img)
        out_path = (output_dir / input_path.name) if output_dir else \
                   input_path.with_stem(input_path.stem + "_undist")
        if output_dir:
            output_dir.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out_path), result)
        print(f"Done. Saved undistorted image to '{out_path}'.")
