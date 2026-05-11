from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from .digit_tokens import TokenizerLike, NumericTokenIndex, build_integer_vnum_from_tokenizer
from .kernels import KernelMode, KernelSpec, DistanceKernelType, build_kernel_matrix


class LossMode:
    """
    Controls which component(s) of the paper objective to use.

    - "mmd":      L_MMD = r^T K r
    - "smooth":  L_smooth = r^T L r  (Dirichlet energy / graph Laplacian)
    - "smmd":     L_SMMD = L_MMD + α L_smooth
    """
    MMD = "mmd"
    smooth = "smooth"
    SMMD = "smmd"


@dataclass(frozen=True)
class TauConfig:
    """
    Smoothness scaling α in the paper (Section 3.4).

    Paper sets:
      α = 1 / (2 * mean(deg_i)),
      deg_i = sum_j K_ij

    We implement:
      - mode="auto": compute α from the precomputed kernel degrees (paper default)
      - mode="fixed": use a user-provided constant
    """
    mode: str = "auto"      # "auto" | "fixed"
    fixed_value: float = 1.0


class SMMDLoss(nn.Module):
    """
    Smooth Maximum Mean Discrepancy (SMMD) auxiliary loss.

    Paper pipeline (per numeric-target position t):
      1) Restrict logits ℓ_t to the numeric sub-vocabulary V_num and renormalize:
           p_t = softmax(ℓ_t[V_num]) ∈ Δ^N
      2) One-hot target over V_num:
           q_t = e_{π(y_t)}
      3) Residual:
           r_t = p_t − q_t
      4) MMD alignment:
           L_MMD = r_t^T K r_t
      5) Smoothness via Laplacian L = D − K:
           L_smooth = r_t^T L r_t = 1/2 ∑_{i,j} K_ij (r_i − r_j)^2
      6) Unified objective:
           L_SMMD = L_MMD + α L_smooth

    This module computes the mean auxiliary loss over numeric-target positions in a batch.
    It returns 0 if the batch contains no numeric-target positions.

    Practical instantiation of V_num:
      - In this implementation, V_num is built as standalone integer tokens within [0..max_value],
        matching common tokenizer behaviors discussed in the paper (digits or 0..999 single tokens).
    """

    def __init__(
        self,
        tokenizer: Optional[TokenizerLike] = None,
        *,
        max_value: int = 999,
        # Kernel design (paper default: Gaussian, σ=2.0)
        kernel_mode: str = KernelMode.VALUE_DISTANCE.value,
        kernel_type: str = "gaussian",
        sigmas: Sequence[float] = (2.0,),
        squash_factor: float = 9.0,
        # Ablations
        ablation_seed: Optional[int] = 42,
        random_feat_dim: int = 4,
        # Smoothness scaling α
        tau_mode: str = "auto",
        tau_fixed: float = 1.0,
        # Which term(s) to use
        loss_mode: str = LossMode.SMMD,
    ):
        super().__init__()

        if loss_mode not in (LossMode.MMD, LossMode.smooth, LossMode.SMMD):
            raise ValueError("loss_mode must be 'mmd' | 'smooth' | 'smmd'")

        if tau_mode not in ("auto", "fixed"):
            raise ValueError("tau_mode must be 'auto' | 'fixed'")

        self.max_value = int(max_value)
        self.loss_mode = str(loss_mode)
        self.kernel_mode = KernelMode(kernel_mode)
        self.ablation_seed = ablation_seed
        self.random_feat_dim = int(random_feat_dim)
        self.tau_cfg = TauConfig(mode=str(tau_mode), fixed_value=float(tau_fixed))

        # Buffers to support "no numeric target in batch"
        self.register_buffer("_zero", torch.tensor(0.0, dtype=torch.float32))

        # Buffers are populated by set_tokenizer()
        self._index: Optional[NumericTokenIndex] = None
        self.register_buffer("vnum_token_ids", torch.empty(0, dtype=torch.long))      # V_num token IDs (π-order)
        self.register_buffer("token_is_numeric", torch.empty(0, dtype=torch.bool))   # [|V|]
        self.register_buffer("token_to_index", torch.empty(0, dtype=torch.long))     # [|V|] -> [0..N-1]
        self.register_buffer("kernel_K", torch.empty(0, dtype=torch.float32))        # [N,N]
        self.register_buffer("kernel_deg", torch.empty(0, dtype=torch.float32))      # [N]
        self.register_buffer("alpha", torch.tensor(1.0, dtype=torch.float32))        # α

        if tokenizer is not None:
            self.set_tokenizer(
                tokenizer,
                kernel_type=kernel_type,
                sigmas=sigmas,
                squash_factor=squash_factor,
            )

    @torch.no_grad()
    def set_tokenizer(
        self,
        tokenizer: TokenizerLike,
        *,
        kernel_type: str = "gaussian",
        sigmas: Sequence[float] = (2.0,),
        squash_factor: float = 9.0,
    ) -> None:
        """
        Build V_num, π, and precompute K, deg, and α.

        - V_num: standalone integer tokens in [0..max_value]
        - v_i: integer value associated with each V_num token
        - K: distance-induced kernel matrix (paper Section 3.1)
        - deg_i: ∑_j K_ij
        - α: 1 / (2 * mean(deg_i)) when tau_mode="auto" (paper Section 3.4)
        """
        index = build_integer_vnum_from_tokenizer(tokenizer, max_value=self.max_value)
        self._index = index

        kt = kernel_type.lower().strip()
        if kt == "gaussian":
            dtype = DistanceKernelType.GAUSSIAN
        elif kt == "laplace":
            dtype = DistanceKernelType.LAPLACE
        else:
            raise ValueError("kernel_type must be 'gaussian' or 'laplace'")

        spec = KernelSpec(
            kernel_type=dtype,
            sigmas=list(sigmas),
            squash_factor=float(squash_factor),
        )

        K = build_kernel_matrix(
            index.values,
            mode=self.kernel_mode,
            spec=spec,
            seed=self.ablation_seed,
            random_feat_dim=self.random_feat_dim,
        )
        deg = K.sum(dim=1).to(dtype=torch.float32)

        if self.tau_cfg.mode == "auto":
            # Paper default: α = 1 / (2 * mean(deg_i))
            alpha = 0.5 / deg.mean().clamp(min=1e-12)
        else:
            alpha = torch.tensor(float(self.tau_cfg.fixed_value), dtype=torch.float32)

        # Register buffers (they move with .to(device))
        self.vnum_token_ids = index.token_ids.clone()
        self.token_is_numeric = index.token_is_numeric.clone()
        self.token_to_index = index.token_to_index.clone()
        self.kernel_K = K.to(dtype=torch.float32)
        self.kernel_deg = deg
        self.alpha = alpha.to(dtype=torch.float32)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: [B, T, |V|] unnormalized LM logits (ℓ in the paper).
            targets: [B, T] token IDs y_t.

        Returns:
            Scalar: mean auxiliary loss over positions where y_t ∈ V_num.
            Returns 0 if there are no numeric-target positions.

        Notes on numerical stability:
            In exact arithmetic, both r^T K r (K PSD) and r^T L r (L PSD) are non-negative.
            We still apply ReLU to guard against tiny negative values from floating-point error.
        """
        if self.kernel_K.numel() == 0 or self.vnum_token_ids.numel() == 0:
            raise RuntimeError("SMMDLoss is not initialized. Pass tokenizer in __init__ or call set_tokenizer().")

        if logits.dim() != 3 or targets.dim() != 2:
            raise ValueError("Expected logits [B,T,V] and targets [B,T].")

        B, T, V = logits.shape
        if targets.shape != (B, T):
            raise ValueError("targets shape must match logits [B,T].")

        logits_flat = logits.reshape(B * T, V)
        targets_flat = targets.reshape(B * T)

        mask = self._numeric_target_mask(targets_flat)
        if not mask.any():
            return self._zero.to(device=logits.device)

        # π(y): map target token IDs to indices in V_num
        y_idx = self.token_to_index[targets_flat[mask]]  # [N_pos]

        # p = softmax(ℓ[V_num])  
        digit_logits = logits_flat[mask][:, self.vnum_token_ids]  # [N_pos, N]
        p = F.softmax(digit_logits, dim=-1)

        # q = one-hot target     
        # r = p - q                (paper residual definition)
        # Compute r^T K r without explicitly forming r:
        #
        # r^T K r = p^T K p + q^T K q - 2 q^T K p
        # where q is one-hot, so:
        #   q^T K q = K_tt
        #   q^T K p = (K[t] · p)
        pK = p @ self.kernel_K
        term_pp = (pK * p).sum(dim=1)                # p^T K p
        K_t = self.kernel_K[y_idx]                   # row K[t, :]
        term_qp = (K_t * p).sum(dim=1)               # q^T K p
        term_qq = self.kernel_K.diag().gather(0, y_idx)  # K_tt

        L_mmd = term_pp + term_qq - 2.0 * term_qp    # r^T K r 

        # Smoothness term: r^T L r with L = D - K
        # Expand r^T D r similarly using q one-hot:
        #
        # r^T D r = sum_i deg_i r_i^2
        #        = sum_i deg_i p_i^2 + deg_t*(1 - 2 p_t)
        deg = self.kernel_deg.unsqueeze(0)           # [1, N]
        term_Dp = ((p * p) * deg).sum(dim=1)         # sum_i deg_i p_i^2
        p_t = p.gather(1, y_idx.unsqueeze(1)).squeeze(1)
        deg_t = self.kernel_deg.gather(0, y_idx)
        term_Dr = term_Dp + deg_t * (1.0 - 2.0 * p_t)

        L_smooth = term_Dr - L_mmd                   # r^T(D-K)r = r^T L r

        if self.loss_mode == LossMode.MMD:
            raw = L_mmd
        elif self.loss_mode == LossMode.smooth:
            raw = self.alpha * L_smooth
        elif self.loss_mode == LossMode.SMMD:
            raw = L_mmd + self.alpha * L_smooth   
        else:
            raise RuntimeError(f"Unknown loss_mode: {self.loss_mode}")

        return F.relu(raw).mean()

    def _numeric_target_mask(self, y: torch.Tensor) -> torch.Tensor:
        """
        Returns a boolean mask for positions where y ∈ V_num.

        This corresponds to the paper rule:
          - If y_t ∈ V_num, apply SMMD at position t
          - Otherwise, SMMD contribution is 0 (Algorithm 2)
        """
        vocab_size = int(self.token_is_numeric.numel())
        valid = (y >= 0) & (y < vocab_size)
        safe_y = torch.where(valid, y, torch.zeros_like(y))
        return valid & self.token_is_numeric[safe_y]


# Optional alias: paper name "Smooth MMD" / "SMMD"
SmoothMMDLoss = SMMDLoss
