import torch
import torch.nn as nn
from utils_part2 import batched_T_i


def sample_along_rays(
    r_os: torch.Tensor,
    r_ds: torch.Tensor,
    near: float,
    far: float,
    num_samples_along_ray: int,
    perturb: bool = True,
    device: str = "cuda",
):
    """Sample points along rays.

    Args:
        r_os: torch.Tensor of shape (num_pixels, 3) representing the ray origins
        r_ds: torch.Tensor of shape (num_pixels, 3) representing the ray directions
        near: float representing the near plane distance
        far: float representing the far plane distance
        num_samples_along_ray: int representing the number of samples to take along each ray
        perturb: bool representing whether to perturb the samples (True for training, False for testing)
        device: str representing the device to run on

    Returns:
        samples: torch.Tensor of shape (num_pixels, num_samples_along_ray, 3) representing
        the 3D positions of the samples along each ray
    """
    device = torch.device(device)
    r_os = r_os.to(device)
    r_ds = r_ds.to(device)
    dtype = r_os.dtype

    num_rays = r_os.shape[0]
    n = num_samples_along_ray
    if n < 1:
        raise ValueError("num_samples_along_ray must be >= 1")

    if n == 1:
        t = torch.full((num_rays, 1), 0.5 * (near + far), device=device, dtype=dtype)
    else:
        bin_edges = torch.linspace(near, far, n + 1, device=device, dtype=dtype)
        t_lo, t_hi = bin_edges[:-1], bin_edges[1:]
        if perturb:
            u = torch.rand(num_rays, n, device=device, dtype=dtype)
            t = t_lo.unsqueeze(0) + u * (t_hi - t_lo).unsqueeze(0)
        else:
            t_mid = 0.5 * (t_lo + t_hi)
            t = t_mid.unsqueeze(0).expand(num_rays, n)

    samples = r_os.unsqueeze(1) + t.unsqueeze(-1) * r_ds.unsqueeze(1)
    return samples


def volrend(
    sigmas: torch.Tensor,
    rgbs: torch.Tensor,
    near: float,
    far: float,
    num_samples_along_ray: int,
    device: str = "cuda",
):
    """Volume rendering along rays using the discrete approximation.

    Args:
        sigmas: torch.Tensor of shape (num_pixels, num_samples, 1) representing the density at each sample
        rgbs: torch.Tensor of shape (num_pixels, num_samples, 3) representing the color at each sample
        near: float representing the near plane distance
        far: float representing the far plane distance
        num_samples_along_ray: int representing the number of samples along each ray
        device: str representing the device to run on

    Returns:
        rendered_colors: torch.Tensor of shape (num_pixels, 3) representing the accumulated color for each ray
    """
    step_size = (far - near) / num_samples_along_ray
    delta = torch.tensor([step_size], device=sigmas.device, dtype=sigmas.dtype)
    T = batched_T_i(sigmas, delta)

    weights = T * (1.0 - torch.exp(-sigmas * delta))

    return torch.sum(weights * rgbs, dim=1)


def predict_rgbs(
    model: nn.Module,
    xyzs: torch.Tensor,
    r_ds: torch.Tensor,
    near: float,
    far: float,
    num_samples_along_ray: int,
    device: str = "cuda",
):
    """Predict colors from a model.

    Args:
        model: nn.Module representing the NeRF model
        xyzs: torch.Tensor of shape (num_pixels, num_samples, 3) representing sample positions along rays
        r_ds: torch.Tensor of shape (num_pixels, 3) representing the ray directions
        near: float representing the near plane distance
        far: float representing the far plane distance
        num_samples_along_ray: int representing the number of samples along each ray
        device: str representing the device to run on

    Returns:
        predicted_rgbs: torch.Tensor of shape (num_pixels, 3) representing the predicted colors
    """
    rgbs, sigmas = model(xyzs, r_ds)
    return volrend(
        sigmas,
        rgbs,
        near=near,
        far=far,
        num_samples_along_ray=num_samples_along_ray,
        device=device,
    )
