# CV2 HW4: Neural Radiance Fields

## Install Dependencies

- Python 3.11+ recommended (3.12 works with the versions below)
- `torch>=2.1.0`
- `numpy>=1.24.3`
- `Pillow>=10.0.0`
- `matplotlib>=3.7.2`
- `imageio>=2.31.0`
- `tyro>=0.8.0` and `viser>=0.2.0` (optional: `visualize_viser.py`)

### Setup

From the repository root:

```bash
pip install -r requirements.txt
```

For a CUDA PyTorch build, install from [pytorch.org](https://pytorch.org) first, then run the line above.

## Run

Paths are resolved from each script’s location; you can run from the repo root as shown or `cd src` and run `python part1.py`, etc.

### Part 1 — image fitting (MLP + positional encoding)

1. Place images in **`web/assets/`**.
2. Expected inputs (missing files are skipped):
   - `img1.jpg`
   - `P1003638.JPG`
3. Run:

```bash
python src/part1.py
```

### Part 1 

1. Requires **`web/assets/img1.jpg`**.
2. Run:

```bash
python src/part1_web_assets.py
```

### Part 2 

1. Place **`data/lego_200x200.npz`** at the repo root (i.e. `data/lego_200x200.npz`).
2. Run (training is slow on CPU; GPU recommended):

```bash
python src/part2.py
```

### Optional — Viser ray demo

```bash
cd src && python visualize_viser.py
```

## The scripts will generate

Outputs are written under **`web/assets/`** unless noted.

### Part 1 (`part1.py`)

- `{stem}_progress_1.png` — render before training (random init)
- `{stem}_progress_2.png` — render after partway training (~25% of iterations)
- `{stem}_final.png` — final reconstruction after full training  
  (`stem` is `img1` or `P1003638` for the inputs above)

### Part 1 web assets (`part1_web_assets.py`)

- `img1_L2_w128.png`, `img1_L2_w256.png`, `img1_L10_w128.png` — ablation reconstructions
- `part1_psnr.png` — PSNR / batch MSE vs step for **`img1.jpg`**

### Part 2 (`part2.py`)

- `part2_brief_description.txt` — short implementation summary
- `part2_rays_and_samples.png` — ray and sample visualization
- `progress_renders/iter_*.png` — validation-view snapshots during training
- `validation_psnr_curve.png` — mean validation PSNR vs step
- `training_progression.png` — montage of progress renders
- `part2_psnr.png` — copy of the validation PSNR curve (for the report page)
- `part2_iter1.png`, `part2_iter100.png`, `part2_iter500.png` — selected progress frames (copied from `progress_renders/`)
- `part2_final.png` — novel view (first test camera)
- `part2_spherical.gif` — orbit / test-trajectory GIF

## View results

Open **`web/index.html`** in a browser. It loads figures from **`web/assets/`**.

## LLM usage

Cursor / LLM assistance was used for HTML layout, README scaffolding, and incremental code edits (e.g. wiring assets, refactors).
