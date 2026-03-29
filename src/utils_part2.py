import torch


def batched_T_i(sigmas, delta):
    alpha = 1.0 - torch.exp(-sigmas * delta)
    one_minus = torch.cat(
        [
            torch.ones(
                sigmas.shape[0], 1, sigmas.shape[2],
                device=sigmas.device, dtype=sigmas.dtype,
            ),
            1.0 - alpha[:, :-1, :] + 1e-10,
        ],
        dim=1,
    )
    return torch.cumprod(one_minus, dim=1)
