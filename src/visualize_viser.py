import sys
from pathlib import Path

# This file lives in `src/`; ensure sibling imports resolve for any cwd.
_src_dir = Path(__file__).resolve().parent
if str(_src_dir) not in sys.path:
    sys.path.insert(0, str(_src_dir))

import viser
import time
import numpy as np
import torch
import tyro
from dataclasses import dataclass

from dataset_3d import load_data, RaysData
from rendering import sample_along_rays


@dataclass
class Config:
    data_path: str = "data/lego_200x200.npz"
    near: float = 2.0
    far: float = 6.0
    num_samples_along_ray: int = 64
    num_rays: int = 128
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    port: int = 8080


def _resolve_data_path(p: str) -> str:
    path = Path(p)
    if path.is_file():
        return str(path)
    parent = Path(__file__).resolve().parent.parent
    alt = parent / p
    if alt.is_file():
        return str(alt)
    return str(path)


def main(cfg: Config):
    images_train, c2ws_train, images_val, c2ws_val, c2ws_test, K = load_data(
        data_path=_resolve_data_path(cfg.data_path)
    )

    H, W = images_train.shape[1], images_train.shape[2]

    device = torch.device(cfg.device)
    images_train_t = torch.as_tensor(images_train, dtype=torch.float32).to(device)
    K_t = torch.as_tensor(K, dtype=torch.float32)
    c2ws_train_t = torch.as_tensor(c2ws_train, dtype=torch.float64)

    dataset = RaysData(images_train_t, K_t, c2ws_train_t, device=str(device))

    near = cfg.near if cfg.near is not None else 2.0
    far = cfg.far if cfg.far is not None else 6.0
    num_samples = cfg.num_samples_along_ray if cfg.num_samples_along_ray is not None else 64
    num_rays = cfg.num_rays if cfg.num_rays is not None else 128

    # Verify uvs aren't flipped
    uvs_start = 0
    uvs_end = min(H * W, 40_000)
    sample_uvs = dataset.uvs[uvs_start:uvs_end].to(device)
    assert torch.allclose(
        images_train_t[0, sample_uvs[:, 1], sample_uvs[:, 0]],
        dataset.gt_rgbs[uvs_start:uvs_end],
        atol=1e-5,
        rtol=0.0,
    )
    print("UVs assertion passed!")

    # Sample random rays from the first image
    num_pixels_per_image = H * W
    indices = np.random.randint(low=0, high=num_pixels_per_image, size=num_rays)

    rays_o = dataset.rays_o[indices]
    rays_d = dataset.rays_d[indices]

    points = sample_along_rays(
        r_os=rays_o,
        r_ds=rays_d,
        near=near,
        far=far,
        num_samples_along_ray=num_samples,
        perturb=True,
        device=cfg.device,
    )  # (num_rays, num_samples, 3)

    # Convert to numpy for viser
    rays_o_np = rays_o.cpu().detach().numpy()
    rays_d_np = rays_d.cpu().detach().numpy()
    points_np = points.cpu().detach().numpy()
    images_np = np.asarray(images_train)
    c2ws_np = np.asarray(c2ws_train)
    K_np = K.detach().cpu().numpy() if isinstance(K, torch.Tensor) else np.asarray(K)

    server = viser.ViserServer(port=cfg.port, share=True)

    fov = float(2 * np.arctan2(H / 2, K_np[0, 0]))
    aspect = float(W / H)

    for i, (image, c2w) in enumerate(zip(images_np, c2ws_np)):
        server.scene.add_camera_frustum(
            f"/cameras/{i}",
            fov=fov,
            aspect=aspect,
            scale=0.15,
            wxyz=viser.transforms.SO3.from_matrix(c2w[:3, :3]).wxyz,
            position=c2w[:3, 3],
            image=image,
        )

    for i, (o, d) in enumerate(zip(rays_o_np, rays_d_np)):

        positions = np.stack((o, o + d * far))
        server.scene.add_spline_catmull_rom(
            f"/rays/{i}",
            positions=positions,
        )

    server.scene.add_point_cloud(
        "/samples",
        colors=np.zeros_like(points_np).reshape(-1, 3),
        points=points_np.reshape(-1, 3),
        point_size=0.03,
    )

    print(f"Viser server running on port {cfg.port}. Press Ctrl+C to stop.")
    while True:
        time.sleep(0.1)


if __name__ == "__main__":
    cfg = tyro.cli(Config)
    main(cfg)

