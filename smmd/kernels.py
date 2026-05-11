from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional, Sequence

import torch
import torch.nn.functional as F


class KernelMode(str, Enum):
    """
    Kernel construction modes (used for ablation and analysis).

    - VALUE_DISTANCE:
        K_ij = κ(|v_i - v_j|) using the true numeric values v_i (paper default).
    - RANDOM_KERNEL:
        A value-agnostic PSD kernel used as a stringent control (Appendix C.4).
    - SHUFFLED_MAPPING:
        Same functional form as VALUE_DISTANCE but permute the value↔token mapping,
        breaking semantic alignment while preserving the kernel "shape".
    """
    VALUE_DISTANCE = "value_distance"
    RANDOM_KERNEL = "random_kernel"
    SHUFFLED_MAPPING = "shuffled_mapping"


class DistanceKernelType(str, Enum):
    """
    Distance-to-similarity mapping κ(·).

    The paper uses the Gaussian RBF:
      κ_σ(d) = exp(-d^2 / (2σ^2))

    We keep Laplace as an optional legacy variant:
      κ_σ(d) = exp(-|d| / σ)
    """
    GAUSSIAN = "gaussian"
    LAPLACE = "laplace"


@dataclass(frozen=True)
class KernelSpec:
    """
    Configuration for distance-induced kernels.

    The paper supports a set Σ of bandwidths (multi-kernel average).
    In the main experiments, a single σ=2.0 is the default, and multi-kernel
    mixtures provide limited benefits (Appendix C.5).

    squash_factor:
      Engineering normalization to map raw distances to a consistent dynamic range
      across different V_num choices (e.g., digits vs 0..999). Set to 0 or 1 to
      effectively disable rescaling.
    """
    kernel_type: DistanceKernelType = DistanceKernelType.GAUSSIAN
    sigmas: Sequence[float] = (2.0,)
    squash_factor: float = 9.0


def build_kernel_matrix(
    values: Sequence[int],
    *,
    mode: KernelMode,
    spec: KernelSpec,
    seed: Optional[int] = 42,
    random_feat_dim: int = 4,
) -> torch.Tensor:
    """
    Build the Gram matrix K ∈ R^{N×N} over V_num indices.

    Paper connection:
      - K encodes numeric proximity (Section 3.1).
      - K is used in the discrete MMD term: L_MMD = r^T K r (Section 3.2).
      - K also defines the weighted graph for smoothness (Section 3.3).

    Returns:
      K: float32 tensor of shape [N, N]
    """
    values = list(values)
    N = len(values)
    if N == 0:
        raise ValueError("values must be non-empty")

    if mode == KernelMode.RANDOM_KERNEL:
        # Appendix C.4: Rademacher embeddings -> Gram -> Hadamard power (odd integer p=5)
        return _build_random_psd_kernel(N, seed=seed, random_feat_dim=random_feat_dim).to(torch.float32)

    if mode in (KernelMode.VALUE_DISTANCE, KernelMode.SHUFFLED_MAPPING):
        dist = _build_distance_matrix(values, mode=mode, seed=seed)

        if spec.squash_factor is not None and float(spec.squash_factor) > 1.0:
            dist = squash_distances(dist, squash_factor=float(spec.squash_factor))

        K = torch.zeros_like(dist, dtype=torch.float32)
        sigmas = list(spec.sigmas)
        if len(sigmas) == 0:
            raise ValueError("KernelSpec.sigmas must be non-empty")

        if spec.kernel_type == DistanceKernelType.GAUSSIAN:
            # Paper default: Gaussian RBF over value distance
            for s in sigmas:
                s = float(s)
                K += torch.exp(-(dist ** 2) / (2.0 * (s ** 2)))
        elif spec.kernel_type == DistanceKernelType.LAPLACE:
            for s in sigmas:
                s = float(s)
                K += torch.exp(-torch.abs(dist) / s)
        else:
            raise ValueError(f"Unknown kernel_type: {spec.kernel_type}")

        K /= float(len(sigmas))
        return torch.clamp(K, 0.0, 1.0)

    raise ValueError(f"Unknown kernel mode: {mode}")


def _build_distance_matrix(values: Sequence[int], *, mode: KernelMode, seed: Optional[int]) -> torch.Tensor:
    """
    d(i,j) = |v_i - v_j|  with an optional permutation ablation.
    """
    base = torch.tensor(list(values), dtype=torch.float32)

    if mode == KernelMode.VALUE_DISTANCE:
        labels = base
    elif mode == KernelMode.SHUFFLED_MAPPING:
        gen = torch.Generator(device="cpu")
        if seed is not None:
            gen.manual_seed(int(seed))
        perm = torch.randperm(base.numel(), generator=gen)
        labels = base[perm]
    else:
        raise ValueError(f"Distance matrix only defined for VALUE_DISTANCE/SHUFFLED_MAPPING, got: {mode}")

    return torch.abs(labels.view(-1, 1) - labels.view(1, -1))


def _build_random_psd_kernel(num_count: int, *, seed: Optional[int], random_feat_dim: int) -> torch.Tensor:
    """
    Random PSD kernel (Appendix C.4).

    Construction:
      1) Sample z_i ∈ {−1, +1}^d i.i.d. (Rademacher), normalize to ||z_i||_2 = 1.
      2) K^(0) = Z Z^T is PSD with diag=1.
      3) Apply Hadamard (elementwise) power with odd integer p=5:
            K = (K^(0)) ∘ p
         which remains PSD by the Schur product theorem.

    This removes number-line structure while keeping the MMD form well-defined.
    """
    gen = torch.Generator(device="cpu")
    if seed is not None:
        gen.manual_seed(int(seed))

    d = max(2, int(random_feat_dim))
    Z = torch.randint(0, 2, (num_count, d), generator=gen, dtype=torch.float32)
    Z = Z * 2.0 - 1.0
    Z = F.normalize(Z, p=2, dim=1)  # ensures diag(K^(0)) = 1

    K0 = Z @ Z.t()
    p = 5  # paper ablation uses p=5
    return K0.pow(p)


def squash_distances(dist: torch.Tensor, squash_factor: float) -> torch.Tensor:
    """
    Optional rescaling of distance magnitudes for stability.

    It preserves:
      - dist==0 stays 0
      - min non-zero distance maps to ~1
      - max distance maps to ~squash_factor

    This can help when switching between V_num choices with very different ranges
    (e.g., digits 0..9 vs integers 0..999), so that σ has a similar "locality meaning".
    """
    if dist.numel() == 0 or dist.size(0) <= 1:
        return dist

    mask_nz = dist > 0
    if not mask_nz.any():
        return dist

    min_nz = dist[mask_nz].min()
    max_v = dist.max()
    if max_v <= min_nz:
        return dist

    scale = (float(squash_factor) - 1.0) / (max_v - min_nz + 1e-10)
    out = 1.0 + (dist - min_nz) * scale
    out = out.clone()
    out[dist == 0] = 0.0
    return out
