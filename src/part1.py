from __future__ import annotations

import os
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import math
from PIL import Image
import numpy as np

#network for part 1
class Part1MLP(nn.Module):
    def __init__(self, L=10, width=256):
        super().__init__()
        self.L = L
        self.width = width
        input_dim = 2 + 4 * L
        w = width
        self.net = nn.Sequential(
            nn.Linear(input_dim, w),
            nn.ReLU(),
            nn.Linear(w, w),
            nn.ReLU(),
            nn.Linear(w, w),
            nn.ReLU(),
            nn.Linear(w, 3),
            nn.Sigmoid(),
        )

    def positional_encoding(self, x):
        out = [x]  
        for i in range(self.L):
            freq = (2 ** i) * math.pi
            out.append(torch.sin(freq * x))
            out.append(torch.cos(freq * x))
        return torch.cat(out, dim=-1)

    def forward(self, x):
        x = self.positional_encoding(x)
        return self.net(x)

# dataloader for part 1
def load_image(path):
    img = Image.open(path).convert("RGB")
    img = np.array(img) / 255.0  
    return torch.tensor(img, dtype=torch.float32)

class PixelDataset(Dataset):

    def __init__(self, image):
        if isinstance(image, np.ndarray):
            image = torch.from_numpy(image).float()
        self.image = image
        self.H, self.W = image.shape[0], image.shape[1]

        ys = torch.arange(self.H, dtype=torch.float32)
        xs = torch.arange(self.W, dtype=torch.float32)
        yy, xx = torch.meshgrid(ys, xs, indexing="ij")
        self._coords = torch.stack([xx, yy], dim=-1).reshape(-1, 2)
        self._colors = self.image.reshape(-1, 3)

    def __len__(self):
        return self.H * self.W

    def __getitem__(self, idx):
        # Normalize coordinates: x/w, y/h -> [0, 1]
        coord = self._coords[idx] / torch.tensor([self.W, self.H], dtype=torch.float32)
        # Normalize colors: rgb/255 -> [0, 1]
        color = self._colors[idx]
        if color.max() > 1.0:
            color = color / 255.0
        return coord, color


def get_pixel_dataloader(image, num_samples=1024, shuffle=True):
    dataset = PixelDataset(image)
    return DataLoader(dataset, batch_size=num_samples, shuffle=shuffle)


def psnr_from_mse(mse):
    if isinstance(mse, torch.Tensor):
        mse = mse.detach().cpu().item()
    if mse <= 0:
        return float("inf")
    return 10.0 * math.log10(1.0 / mse)


def save_psnr_curve(history, path, title="Part 1 training"):
    """Save a PNG of PSNR (and MSE on secondary axis) vs step. Requires matplotlib."""
    try:
        import matplotlib.pyplot as plt
    except ImportError as e:
        raise ImportError("Install matplotlib to plot: pip install matplotlib") from e

    steps = [h["step"] for h in history]
    psnr = [h["psnr"] for h in history]
    mse = [h["mse"] for h in history]

    fig, ax1 = plt.subplots(figsize=(8, 4))
    ax1.set_xlabel("Step")
    ax1.set_ylabel("PSNR (dB)", color="tab:blue")
    ax1.plot(steps, psnr, color="tab:blue", label="PSNR")
    ax1.tick_params(axis="y", labelcolor="tab:blue")
    ax1.grid(True, alpha=0.3)

    ax2 = ax1.twinx()
    ax2.set_ylabel("MSE (batch)", color="tab:orange")
    ax2.plot(steps, mse, color="tab:orange", alpha=0.7, label="MSE")
    ax2.tick_params(axis="y", labelcolor="tab:orange")
    ax2.set_yscale("log")

    fig.suptitle(title)
    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def train_part1(
    image,
    num_iters: int = 1000,
    batch_size: int = 10_000,
    lr: float = 1e-2,
    L: int = 10,
    width: int = 256,
    log_every: int = 100,
    device=None,
    history=None,
    snapshot_prefix: str | None = None,
):
    """Train Part 1 MLP. If snapshot_prefix is set, save full renders to
    ``{snapshot_prefix}_progress_1.png`` (before training) and
    ``{snapshot_prefix}_progress_2.png`` (after ``max(1, num_iters // 4)`` steps).
    The trained result is saved separately by ``train_and_save_part1`` as ``*_final.png``.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)

    if isinstance(image, np.ndarray):
        image = torch.from_numpy(image).float()
    image = image.to(device)
    h_img, w_img = int(image.shape[0]), int(image.shape[1])

    model = Part1MLP(L=L, width=width).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    dataloader = get_pixel_dataloader(
        image.cpu(), num_samples=batch_size, shuffle=True
    )

    loader_iter = iter(dataloader)

    def next_batch():
        nonlocal loader_iter
        try:
            return next(loader_iter)
        except StopIteration:
            loader_iter = iter(dataloader)
            return next(loader_iter)

    def save_progress(tag: int):
        if snapshot_prefix is None:
            return
        path = f"{snapshot_prefix}_progress_{tag}.png"
        arr = render_part1_image(model, h_img, w_img, device=device)
        save_image_rgb01(arr, path)
        print(f"  saved snapshot: {path}")

    # Snapshot 1 = random init (before any optimizer steps).
    # Snapshot 2 = partway through training (~25% of iters) so it still looks "in progress".
    mid_step = max(1, min(num_iters, num_iters // 4))
    save_progress(1)

    model.train()
    for step in range(1, num_iters + 1):
        coords, rgbs = next_batch()
        coords, rgbs = coords.to(device), rgbs.to(device)

        optimizer.zero_grad()
        pred = model(coords)
        loss = criterion(pred, rgbs)
        loss.backward()
        optimizer.step()

        if step % log_every == 0 or step == 1:
            m = loss.detach().item()
            p = psnr_from_mse(m)
            print(
                f"step {step}/{num_iters}  MSE: {m:.6f}  PSNR: {p:.4f} dB"
            )
            if history is not None:
                history.append({"step": step, "mse": m, "psnr": p})

        if snapshot_prefix is not None and step == mid_step:
            save_progress(2)

    return model


def render_part1_image(
    model: Part1MLP,
    height: int,
    width: int,
    device=None,
    chunk: int = 50_000,
):
    if device is None:
        device = next(model.parameters()).device
    model.eval()
    ys = torch.arange(height, dtype=torch.float32, device=device)
    xs = torch.arange(width, dtype=torch.float32, device=device)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    coords = torch.stack([xx, yy], dim=-1).reshape(-1, 2)
    coords = coords / torch.tensor([width, height], dtype=torch.float32, device=device)

    outs = []
    with torch.no_grad():
        for i in range(0, coords.shape[0], chunk):
            outs.append(model(coords[i : i + chunk]))
    rgb = torch.cat(outs, dim=0).reshape(height, width, 3).cpu().numpy()
    return np.clip(rgb, 0.0, 1.0)


def save_image_rgb01(arr, path):
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    img = Image.fromarray((np.clip(arr, 0.0, 1.0) * 255).astype(np.uint8), mode="RGB")
    img.save(path)


def train_and_save_part1(
    input_path: str,
    output_path: str,
    num_iters: int = 1000,
    batch_size: int = 10_000,
    lr: float = 1e-2,
    L: int = 10,
    width: int = 256,
    psnr_plot_path: str | None = None,
    snapshot_prefix: str | None = None,
):
    img = load_image(input_path)
    H, W = img.shape[0], img.shape[1]
    history = [] if psnr_plot_path else None
    model = train_part1(
        img,
        num_iters=num_iters,
        batch_size=batch_size,
        lr=lr,
        L=L,
        width=width,
        history=history,
        snapshot_prefix=snapshot_prefix,
    )
    if psnr_plot_path and history:
        save_psnr_curve(history, psnr_plot_path, title=f"Part 1: {os.path.basename(input_path)}")
        print(f"Saved PSNR curve: {psnr_plot_path}")
    recon = render_part1_image(model, H, W)
    save_image_rgb01(recon, output_path)
    print(f"Saved reconstruction: {output_path}")


if __name__ == "__main__":
    # Fixed Part 1 demo: train `img1.jpg` and `P1003638.JPG` in `web/assets`,
    # write `{stem}_progress_{1,2}.png` and `{stem}_final.png` alongside them.
    assets = Path(__file__).resolve().parent.parent / "web" / "assets"
    num_iters = 1000
    batch_size = 10_000
    lr = 1e-2
    L, width = 10, 256

    for filename in ("img1.jpg", "P1003638.JPG"):
        in_path = assets / filename
        if not in_path.is_file():
            print(f"skip missing {in_path}")
            continue
        stem = in_path.stem
        print(f"\n=== Part 1: {filename}  L={L}  width={width} ===")
        train_and_save_part1(
            str(in_path),
            str(assets / f"{stem}_final.png"),
            num_iters=num_iters,
            batch_size=batch_size,
            lr=lr,
            L=L,
            width=width,
            snapshot_prefix=str(assets / stem),
        )