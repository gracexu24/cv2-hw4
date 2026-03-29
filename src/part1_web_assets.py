"""Build Part 1 report assets under web/assets: L/width ablations on img1 and PSNR curve on img1."""
import os
import sys
from pathlib import Path

_src = Path(__file__).resolve().parent
if str(_src) not in sys.path:
    sys.path.insert(0, str(_src))

from part1 import load_image, train_and_save_part1, train_part1, save_psnr_curve

REPO = _src.parent
ASSETS = REPO / "web" / "assets"
IMG1 = ASSETS / "img1.jpg"
PSNR_OUT = ASSETS / "part1_psnr.png"


def run_ablations():
    if not IMG1.is_file():
        print(f"[ablations] skip: missing {IMG1}")
        return
    os.makedirs(ASSETS, exist_ok=True)
    runs = [
        (2, 128, "img1_L2_w128"),
        (2, 256, "img1_L2_w256"),
        (10, 128, "img1_L10_w128"),
    ]
    num_iters = int(os.environ.get("PART1_ABLATION_ITERS", "2000"))
    for L, width, name in runs:
        print(f"\n===== {name}  (L={L}, width={width}) =====")
        train_and_save_part1(
            str(IMG1),
            str(ASSETS / f"{name}.png"),
            num_iters=num_iters,
            batch_size=10_000,
            lr=1e-2,
            L=L,
            width=width,
        )


def run_psnr_plot():
    if not IMG1.is_file():
        raise SystemExit(f"[psnr] missing {IMG1} (needed for PSNR curve)")
    img = load_image(str(IMG1))

    hist = []
    train_part1(
        img,
        num_iters=1000,
        batch_size=10_000,
        lr=1e-2,
        log_every=50,
        history=hist,
    )
    os.makedirs(ASSETS, exist_ok=True)
    save_psnr_curve(
        hist,
        str(PSNR_OUT),
        title="Part 1: PSNR from batch MSE (img1.jpg)",
    )
    print(f"[psnr] wrote {PSNR_OUT} ({len(hist)} points)")


def main():
    run_ablations()
    run_psnr_plot()


if __name__ == "__main__":
    main()
