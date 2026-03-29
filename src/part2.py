import math
import os
import shutil
import sys
from pathlib import Path

import imageio.v2 as imageio
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

_src = Path(__file__).resolve().parent
if str(_src) not in sys.path:
    sys.path.insert(0, str(_src))

from dataset_3d import load_data, get_rays_dataloader, pixels_to_rays
from rendering import sample_along_rays, predict_rgbs
from part1 import psnr_from_mse, save_image_rgb01


def _gamma_dim(L: int) -> int:
    return 3 + 3 * 2 * L


class Part2MLP(nn.Module):
    """
    NeRF-style MLP:
    - input: 3D world coordinates + 3D viewing direction
    - output: RGB in [0,1] and nonnegative density sigma

    Matches the assignment requirements:
    - positional encoding on xyz and direction
    - deeper MLP than Part 1
    - skip connection by concatenating encoded xyz mid-network
    """

    def __init__(self, L_xyz: int = 10, L_dir: int = 4, hidden: int = 256):
        super().__init__()
        self.L_xyz = L_xyz
        self.L_dir = L_dir

        in_xyz = _gamma_dim(L_xyz)
        in_dir = _gamma_dim(L_dir)

        self.pre_skip = nn.ModuleList(
            [
                nn.Linear(in_xyz, hidden),
                nn.Linear(hidden, hidden),
                nn.Linear(hidden, hidden),
                nn.Linear(hidden, hidden),
                nn.Linear(hidden, hidden),
            ]
        )

        self.post_skip = nn.ModuleList(
            [
                nn.Linear(hidden + in_xyz, hidden),
                nn.Linear(hidden, hidden),
                nn.Linear(hidden, hidden),
                nn.Linear(hidden, hidden),
                nn.Linear(hidden, hidden),
            ]
        )

        self.sigma_head = nn.Linear(hidden, 1)
        self.feature_head = nn.Linear(hidden, hidden)
        self.color_head_1 = nn.Linear(hidden + in_dir, hidden // 2)
        self.color_head_2 = nn.Linear(hidden // 2, 3)

    def positional_encoding(self, x: torch.Tensor, L: int) -> torch.Tensor:
        out = [x]
        for i in range(L):
            freq = (2.0 ** i) * math.pi
            out.append(torch.sin(freq * x))
            out.append(torch.cos(freq * x))
        return torch.cat(out, dim=-1)

    def forward(self, xyz: torch.Tensor, view_dirs: torch.Tensor):
        dtype = next(self.parameters()).dtype
        orig_ndim = xyz.dim()

        if xyz.dim() == 3:
            b, s, _ = xyz.shape
            xyz_flat = xyz.reshape(b * s, 3).to(dtype=dtype)
            vd = (
                view_dirs[:, None, :]
                .expand(b, s, 3)
                .reshape(b * s, 3)
                .to(dtype=dtype)
            )
        else:
            xyz_flat = xyz.to(dtype=dtype)
            vd = view_dirs.to(dtype=dtype)

        vd = F.normalize(vd, dim=-1, eps=1e-6)

        gamma_x = self.positional_encoding(xyz_flat, self.L_xyz)
        gamma_d = self.positional_encoding(vd, self.L_dir)

        h = gamma_x
        for layer in self.pre_skip:
            h = F.relu(layer(h))

        h = torch.cat([h, gamma_x], dim=-1)

        for layer in self.post_skip:
            h = F.relu(layer(h))

        sigma = F.relu(self.sigma_head(h))
        feat = self.feature_head(h)

        hc = torch.cat([feat, gamma_d], dim=-1)
        hc = F.relu(self.color_head_1(hc))
        rgb = torch.sigmoid(self.color_head_2(hc))

        if orig_ndim == 3:
            rgb = rgb.view(b, s, 3)
            sigma = sigma.view(b, s, 1)

        return rgb, sigma


def render_view(
    model: Part2MLP,
    H: int,
    W: int,
    K: torch.Tensor,
    c2w: torch.Tensor,
    near: float,
    far: float,
    num_samples: int,
    device=None,
    chunk_rays: int = 4096,
):
    """
    Render a full image from one camera pose.
    """
    if device is None:
        device = next(model.parameters()).device
    device = torch.device(device)
    dev_str = str(device)

    model.eval()

    yy, xx = torch.meshgrid(
        torch.arange(H, dtype=torch.float32, device=device),
        torch.arange(W, dtype=torch.float32, device=device),
        indexing="ij",
    )
    uvs = torch.stack([xx.reshape(-1), yy.reshape(-1)], dim=1)

    rays_o, rays_d = pixels_to_rays(K, c2w, uvs, verbose=False, device=dev_str)
    rays_o = rays_o.float()
    rays_d = rays_d.float()

    outputs = []
    with torch.no_grad():
        for start in range(0, rays_o.shape[0], chunk_rays):
            ro = rays_o[start : start + chunk_rays]
            rd = rays_d[start : start + chunk_rays]

            xyz = sample_along_rays(
                ro,
                rd,
                near,
                far,
                num_samples,
                perturb=False,
                device=dev_str,
            ).float()

            pred = predict_rgbs(
                model,
                xyz,
                rd,
                near,
                far,
                num_samples,
                device=dev_str,
            )
            outputs.append(pred)

    img = torch.cat(outputs, dim=0).reshape(H, W, 3).cpu().numpy()
    return np.clip(img, 0.0, 1.0)


def evaluate_validation_set(
    model: Part2MLP,
    images_val,
    c2ws_val,
    K: torch.Tensor,
    near: float,
    far: float,
    num_samples: int,
    device=None,
    chunk_rays: int = 4096,
):
    """
    Render all validation images and compute mean PSNR over the 6-val-image set.
    """
    if device is None:
        device = next(model.parameters()).device

    H, W = images_val.shape[1], images_val.shape[2]
    psnrs = []

    for i in range(images_val.shape[0]):
        gt = np.asarray(images_val[i], dtype=np.float32)
        c2w = torch.as_tensor(c2ws_val[i], dtype=torch.float32, device=device)
        pred = render_view(
            model,
            H,
            W,
            K.to(device),
            c2w,
            near,
            far,
            num_samples,
            device=device,
            chunk_rays=chunk_rays,
        )

        mse = np.mean((pred - gt) ** 2)
        psnr = psnr_from_mse(max(mse, 1e-10))
        psnrs.append(psnr)

    return float(np.mean(psnrs)), psnrs


def plot_validation_psnr_curve(steps, psnrs, out_path: str):
    plt.figure(figsize=(7, 4.5))
    plt.plot(steps, psnrs, marker="o")
    plt.xlabel("Training iteration")
    plt.ylabel("Validation PSNR (mean over 6 images)")
    plt.title("Validation PSNR During Training")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def make_training_progress_figure(image_paths, step_labels, out_path: str):
    """
    Create a horizontal strip of saved renders across training.
    """
    imgs = [imageio.imread(p) for p in image_paths]

    n = len(imgs)
    plt.figure(figsize=(4 * n, 4))
    for i, img in enumerate(imgs):
        plt.subplot(1, n, i + 1)
        plt.imshow(img)
        plt.title(f"iter {step_labels[i]}")
        plt.axis("off")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def train_part2(
    images_train,
    K,
    c2ws_train,
    images_val,
    c2ws_val,
    num_iters: int = 1000,
    batch_rays: int = 2048,
    num_samples: int = 64,
    near: float = 2.0,
    far: float = 6.0,
    lr: float = 5e-4,
    log_every: int = 100,
    render_every: int = 500,
    val_every: int = 500,
    out_dir: str = "out",
    device=None,
):
    """
    Train NeRF and save the main deliverables required by the assignment:
    - intermediate renders across iterations
    - validation PSNR curve
    """
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(os.path.join(out_dir, "progress_renders"), exist_ok=True)

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)

    dev_str = str(device)
    K = torch.as_tensor(K, dtype=torch.float32)

    loader = get_rays_dataloader(
        images_train,
        K,
        c2ws_train,
        batch_size=batch_rays,
        shuffle=True,
        device=dev_str,
    )

    model = Part2MLP().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    loader_it = iter(loader)

    def next_batch():
        nonlocal loader_it
        try:
            return next(loader_it)
        except StopIteration:
            loader_it = iter(loader)
            return next(loader_it)

    val_steps = []
    val_psnrs = []
    progress_paths = []
    progress_labels = []

    H_val, W_val = images_val.shape[1], images_val.shape[2]

    model.train()
    for step in range(1, num_iters + 1):
        rays_o, rays_d, target_rgb = next_batch()

        rays_o = rays_o.to(device=device, dtype=torch.float32)
        rays_d = rays_d.to(device=device, dtype=torch.float32)
        target_rgb = target_rgb.to(device=device, dtype=torch.float32)

        xyz = sample_along_rays(
            rays_o,
            rays_d,
            near,
            far,
            num_samples,
            perturb=True,
            device=dev_str,
        ).float()

        pred_rgb = predict_rgbs(
            model,
            xyz,
            rays_d,
            near,
            far,
            num_samples,
            device=dev_str,
        )

        loss = criterion(pred_rgb, target_rgb)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        if step % log_every == 0 or step == 1:
            mse = loss.detach().item()
            print(
                f"step {step}/{num_iters} | train loss {mse:.6f} | train PSNR {psnr_from_mse(mse):.4f} dB"
            )

        # Save progression render of one fixed validation view
        if step % render_every == 0 or step == 1:
            with torch.no_grad():
                val_img = render_view(
                    model,
                    H_val,
                    W_val,
                    K.to(device),
                    torch.as_tensor(c2ws_val[0], dtype=torch.float32, device=device),
                    near,
                    far,
                    num_samples,
                    device=device,
                )
            render_path = os.path.join(out_dir, "progress_renders", f"iter_{step:06d}.png")
            save_image_rgb01(val_img, render_path)
            progress_paths.append(render_path)
            progress_labels.append(step)
            print(f"Saved training progression render: {render_path}")

        # Evaluate PSNR on all 6 validation images
        if step % val_every == 0 or step == 1:
            mean_psnr, per_image_psnrs = evaluate_validation_set(
                model,
                images_val,
                c2ws_val,
                K,
                near,
                far,
                num_samples,
                device=device,
            )
            val_steps.append(step)
            val_psnrs.append(mean_psnr)
            print(
                f"[validation] step {step}: mean PSNR over {len(per_image_psnrs)} val views = {mean_psnr:.4f} dB"
            )

    # Save PSNR curve
    plot_validation_psnr_curve(
        val_steps,
        val_psnrs,
        os.path.join(out_dir, "validation_psnr_curve.png"),
    )

    # Save montage of predicted images across iterations
    if len(progress_paths) > 0:
        make_training_progress_figure(
            progress_paths,
            progress_labels,
            os.path.join(out_dir, "training_progression.png"),
        )

    return model, {
        "val_steps": val_steps,
        "val_psnrs": val_psnrs,
        "progress_paths": progress_paths,
        "progress_labels": progress_labels,
    }


def render_novel_view(
    model,
    K,
    c2w,
    H,
    W,
    near,
    far,
    num_samples,
    out_path,
    device=None,
):
    img = render_view(
        model,
        H,
        W,
        K,
        c2w,
        near,
        far,
        num_samples,
        device=device,
    )
    save_image_rgb01(img, out_path)
    return img


def render_spherical_video(
    model,
    K,
    c2ws_test,
    H,
    W,
    near,
    far,
    num_samples,
    out_path,
    fps: int = 12,
    device=None,
):
    """
    Render all provided test camera poses and save as GIF or MP4 depending on extension.
    """
    frames = []
    for i in range(len(c2ws_test)):
        c2w = torch.as_tensor(c2ws_test[i], dtype=torch.float32)
        img = render_view(
            model,
            H,
            W,
            K,
            c2w,
            near,
            far,
            num_samples,
            device=device,
        )
        frame = np.clip(img * 255.0, 0, 255).astype(np.uint8)
        frames.append(frame)
        print(f"Rendered test frame {i+1}/{len(c2ws_test)}")

    imageio.mimsave(out_path, frames, fps=fps)
    print(f"Saved spherical render video: {out_path}")


if __name__ == "__main__":
    REPO = Path(__file__).resolve().parent.parent
    DATA = REPO / "data" / "lego_200x200.npz"
    OUT_DIR = REPO / "web" / "assets"

    if not DATA.is_file():
        raise SystemExit(f"Missing dataset: {DATA}")

    dp = str(DATA)
    npz = np.load(dp)
    if "c2ws_test" not in npz:
        raise KeyError("Expected c2ws_test in the .npz file for spherical rendering.")
    c2ws_test = npz["c2ws_test"]

    images_train, c2ws_train, images_val, c2ws_val, _, K = load_data(dp)
    K_t = torch.as_tensor(K, dtype=torch.float32)

    num_iters = 1000
    batch_rays = 2048
    num_samples = 64
    near_f, far_f = 2.0, 6.0
    lr = 5e-4
    log_every = 100
    render_every = 500
    val_every = 500
    spherical_fps = 12

    out_dir = str(OUT_DIR)
    os.makedirs(out_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Loaded data: {dp}")
    print(f"Outputs -> {out_dir}")

    model, _logs = train_part2(
        images_train=images_train,
        K=K_t,
        c2ws_train=c2ws_train,
        images_val=images_val,
        c2ws_val=c2ws_val,
        num_iters=num_iters,
        batch_rays=batch_rays,
        num_samples=num_samples,
        near=near_f,
        far=far_f,
        lr=lr,
        log_every=log_every,
        render_every=render_every,
        val_every=val_every,
        out_dir=out_dir,
        device=device,
    )

    curve_src = os.path.join(out_dir, "validation_psnr_curve.png")
    curve_dst = os.path.join(out_dir, "part2_psnr.png")
    if os.path.isfile(curve_src):
        shutil.copyfile(curve_src, curve_dst)
        print(f"Copied validation curve -> {curve_dst}")

    pr_dir = os.path.join(out_dir, "progress_renders")
    for step, name in (
        (1, "part2_iter1.png"),
        (100, "part2_iter100.png"),
        (500, "part2_iter500.png"),
    ):
        src = os.path.join(pr_dir, f"iter_{step:06d}.png")
        dst = os.path.join(out_dir, name)
        if os.path.isfile(src):
            shutil.copyfile(src, dst)
            print(f"Copied {src} -> {dst}")

    H_val, W_val = images_val.shape[1], images_val.shape[2]

    render_novel_view(
        model=model,
        K=K_t,
        c2w=torch.as_tensor(c2ws_test[0], dtype=torch.float32),
        H=H_val,
        W=W_val,
        near=near_f,
        far=far_f,
        num_samples=num_samples,
        out_path=os.path.join(out_dir, "part2_final.png"),
        device=device,
    )
    print(f"Saved novel view: {os.path.join(out_dir, 'part2_final.png')}")

    render_spherical_video(
        model=model,
        K=K_t,
        c2ws_test=c2ws_test,
        H=H_val,
        W=W_val,
        near=near_f,
        far=far_f,
        num_samples=num_samples,
        out_path=os.path.join(out_dir, "part2_spherical.gif"),
        fps=spherical_fps,
        device=device,
    )