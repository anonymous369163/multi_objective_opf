"""
Linearized VAE for Preference Flow Matching

This module implements a VAE that constrains the latent space to be approximately
one-dimensional along the preference direction. This makes the latent space 
suitable for simple Flow Matching learning.

Key features:
1. Preference-aware encoding with FiLM conditioning
2. Linearization constraints (L_1D and L_order)
3. Optional NGT physics loss for feasibility
4. Wrap-aware reconstruction loss for angles

Author: Auto-generated from VAE+Flow plan
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Dict, Tuple, Optional, List


class LinearizedVAE_Encoder(nn.Module):
    """
    VAE Encoder with preference conditioning for linearized latent space.
    
    The encoder maps (scene s, solution x, preference λ) to a latent distribution.
    The goal is to make z(s, λ) approximately linear in λ for fixed s.
    """
    
    def __init__(self, input_dim: int, output_dim: int, latent_dim: int, 
                 hidden_dim: int, num_layers: int, pref_dim: int = 1, act: str = 'relu'):
        super().__init__()
        
        act_list = {'relu': nn.ReLU(), 'silu': nn.SiLU(), 'gelu': nn.GELU(), 
                    'softplus': nn.Softplus(), 'tanh': nn.Tanh()}
        act_fn = act_list.get(act, nn.ReLU())
        
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.pref_dim = pref_dim
        
        # Scene (x) branch - encodes load/condition features
        self.scene_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            act_fn,
            nn.Linear(hidden_dim, hidden_dim),
            act_fn,
            nn.LayerNorm(hidden_dim)
        )
        
        # Solution (y) branch - encodes voltage solution
        self.solution_encoder = nn.Sequential(
            nn.Linear(output_dim, hidden_dim),
            act_fn,
            nn.Linear(hidden_dim, hidden_dim),
            act_fn,
            nn.LayerNorm(hidden_dim)
        )
        
        # Preference conditioning via FiLM
        # Maps preference λ to modulation parameters
        self.pref_film = nn.Sequential(
            nn.Linear(pref_dim, hidden_dim),
            act_fn,
            nn.Linear(hidden_dim, hidden_dim * 2)  # [gamma, beta]
        )
        
        # Cross-attention style fusion: solution modulates scene
        self.gamma_sol = nn.Linear(hidden_dim, hidden_dim)
        self.beta_sol = nn.Linear(hidden_dim, hidden_dim)
        
        # Deep fusion network
        fusion_layers = []
        for _ in range(max(1, num_layers)):
            fusion_layers.extend([
                nn.Linear(hidden_dim, hidden_dim),
                act_fn,
                nn.LayerNorm(hidden_dim)
            ])
        self.fusion_net = nn.Sequential(*fusion_layers)
        
        # Latent distribution parameters
        self.mean_layer = nn.Linear(hidden_dim, latent_dim)
        self.logvar_layer = nn.Linear(hidden_dim, latent_dim)
    
    def forward(self, scene: torch.Tensor, solution: torch.Tensor, 
                pref: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode (scene, solution, pref) to latent distribution.
        
        Args:
            scene: [B, input_dim] - Scene features (load data)
            solution: [B, output_dim] - Target solution (voltage)
            pref: [B, pref_dim] or [B] or None - Preference parameter (normalized λ)
        
        Returns:
            mean: [B, latent_dim] - Mean of latent distribution
            logvar: [B, latent_dim] - Log variance of latent distribution
        """
        # Encode scene and solution
        scene_feat = self.scene_encoder(scene)       # [B, H]
        sol_feat = self.solution_encoder(solution)   # [B, H]
        
        # Solution modulates scene (FiLM style)
        gamma = self.gamma_sol(sol_feat)             # [B, H]
        beta = self.beta_sol(sol_feat)               # [B, H]
        fused = gamma * scene_feat + beta            # [B, H]
        
        # Apply preference FiLM if provided
        if pref is not None:
            if pref.dim() == 1:
                pref = pref.unsqueeze(-1)
            
            pref_params = self.pref_film(pref)       # [B, 2H]
            pref_gamma = pref_params[:, :self.hidden_dim]
            pref_beta = pref_params[:, self.hidden_dim:]
            
            fused = (1.0 + pref_gamma) * fused + pref_beta
        
        # Deep fusion with skip connection
        deep = self.fusion_net(fused)
        deep = deep + fused
        
        # Output distribution parameters
        mean = self.mean_layer(deep)
        logvar = self.logvar_layer(deep).clamp(-20, 20)  # Numerical stability
        
        return mean, logvar
    
    def encode_from_condition(self, scene: torch.Tensor, 
                              pref: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode from scene and preference only (inference mode, no solution).
        
        Args:
            scene: [B, input_dim] - Scene features
            pref: [B, pref_dim] or None - Preference parameter
        
        Returns:
            mean, logvar: Latent distribution parameters
        """
        scene_feat = self.scene_encoder(scene)
        
        # Apply preference FiLM if provided
        if pref is not None:
            if pref.dim() == 1:
                pref = pref.unsqueeze(-1)
            
            pref_params = self.pref_film(pref)
            pref_gamma = pref_params[:, :self.hidden_dim]
            pref_beta = pref_params[:, self.hidden_dim:]
            
            scene_feat = (1.0 + pref_gamma) * scene_feat + pref_beta
        
        deep = self.fusion_net(scene_feat)
        deep = deep + scene_feat
        
        mean = self.mean_layer(deep)
        logvar = self.logvar_layer(deep).clamp(-20, 20)
        
        return mean, logvar


class LinearizedVAE_Decoder(nn.Module):
    """
    VAE Decoder that maps (scene, z) to solution.
    
    The decoder is preference-agnostic - it only sees the latent z.
    The preference information is encoded in z via the encoder.
    """
    
    def __init__(self, input_dim: int, output_dim: int, latent_dim: int,
                 hidden_dim: int, num_layers: int, act: str = 'relu'):
        super().__init__()
        
        act_list = {'relu': nn.ReLU(), 'silu': nn.SiLU(), 'gelu': nn.GELU(),
                    'softplus': nn.Softplus(), 'tanh': nn.Tanh()}
        act_fn = act_list.get(act, nn.ReLU())
        
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        
        # Scene conditioning via FiLM
        self.scene_film = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            act_fn,
            nn.Linear(hidden_dim, hidden_dim * 2)  # [gamma, beta]
        )
        
        # Latent embedding
        self.z_embed = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            act_fn
        )
        
        # Main decoder network
        decoder_layers = []
        for _ in range(num_layers):
            decoder_layers.extend([
                nn.Linear(hidden_dim, hidden_dim),
                act_fn
            ])
        decoder_layers.append(nn.Linear(hidden_dim, output_dim))
        self.decoder_net = nn.Sequential(*decoder_layers)
    
    def forward(self, scene: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """
        Decode from scene and latent z.
        
        Args:
            scene: [B, input_dim] - Scene features
            z: [B, latent_dim] - Latent variable
        
        Returns:
            y: [B, output_dim] - Reconstructed solution
        """
        # Embed latent
        z_feat = self.z_embed(z)  # [B, H]
        
        # Apply scene conditioning via FiLM
        scene_params = self.scene_film(scene)  # [B, 2H]
        gamma = scene_params[:, :self.hidden_dim]
        beta = scene_params[:, self.hidden_dim:]
        
        # Modulated features
        feat = (1.0 + gamma) * z_feat + beta
        
        # Decode to output
        y = self.decoder_net(feat)
        
        return y


class LinearizedVAE(nn.Module):
    """
    Complete Linearized VAE for preference flow learning.
    
    This VAE is trained with additional constraints to make the latent space
    approximately one-dimensional along the preference direction.
    
    Training losses:
    - L_rec: Reconstruction loss (with Va angle wrapping)
    - L_KL: KL divergence regularization
    - L_1D: Low-rank / one-dimensionality constraint (PCA style)
    - L_order: Monotonic ordering constraint (z projection vs λ)
    - L_NGT: Optional physics constraint (feasibility)
    """
    
    def __init__(self, input_dim: int, output_dim: int, latent_dim: int = 32,
                 hidden_dim: int = 256, num_layers: int = 3, pref_dim: int = 1,
                 NPred_Va: int = None, act: str = 'relu'):
        """
        Initialize the Linearized VAE.
        
        Args:
            input_dim: Dimension of scene features (load data)
            output_dim: Dimension of solution (Va + Vm)
            latent_dim: Dimension of latent space
            hidden_dim: Hidden layer dimension
            num_layers: Number of layers in encoder/decoder
            pref_dim: Dimension of preference parameter
            NPred_Va: Number of Va dimensions (for angle wrapping)
            act: Activation function
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.latent_dim = latent_dim
        self.pref_dim = pref_dim
        self.NPred_Va = NPred_Va if NPred_Va is not None else output_dim // 2
        
        # Encoder and Decoder
        self.encoder = LinearizedVAE_Encoder(
            input_dim, output_dim, latent_dim, hidden_dim, num_layers, pref_dim, act
        )
        self.decoder = LinearizedVAE_Decoder(
            input_dim, output_dim, latent_dim, hidden_dim, num_layers, act
        )
    
    def reparameterize(self, mean: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Reparameterization trick: z = mean + std * epsilon"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mean + std * eps
    
    def encode(self, scene: torch.Tensor, solution: torch.Tensor,
               pref: Optional[torch.Tensor] = None, 
               use_mean: bool = False) -> torch.Tensor:
        """
        Encode to latent space.
        
        Args:
            scene: Scene features
            solution: Target solution
            pref: Preference parameter
            use_mean: If True, return mean (deterministic). If False, sample.
        
        Returns:
            z: Latent variable
        """
        mean, logvar = self.encoder(scene, solution, pref)
        if use_mean:
            return mean
        return self.reparameterize(mean, logvar)
    
    def decode(self, scene: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """Decode from latent space."""
        return self.decoder(scene, z)
    
    def forward(self, scene: torch.Tensor, solution: torch.Tensor,
                pref: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Full forward pass: encode and decode.
        
        Returns:
            y_recon: Reconstructed solution
            mean: Latent mean
            logvar: Latent log-variance
        """
        mean, logvar = self.encoder(scene, solution, pref)
        z = self.reparameterize(mean, logvar)
        y_recon = self.decoder(scene, z)
        return y_recon, mean, logvar
    
    def inference(self, scene: torch.Tensor, 
                  pref: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Inference mode: generate solution from scene and preference.
        
        Args:
            scene: Scene features
            pref: Preference parameter
        
        Returns:
            y: Generated solution
        """
        mean, logvar = self.encoder.encode_from_condition(scene, pref)
        z = mean  # Use mean for deterministic inference
        return self.decoder(scene, z)


# ==================== Loss Functions ====================

def wrap_angle_difference(diff: torch.Tensor, NPred_Va: int) -> torch.Tensor:
    """
    Wrap angle differences to [-π, π] for Va dimensions.
    
    Args:
        diff: Difference tensor [B, output_dim]
        NPred_Va: Number of Va dimensions (assumed to be first NPred_Va dims)
    
    Returns:
        Wrapped difference tensor
    """
    if NPred_Va <= 0:
        return diff
    
    # Split Va and Vm dimensions (non-inplace operation)
    va_diff = diff[:, :NPred_Va]
    vm_diff = diff[:, NPred_Va:]
    
    # Wrap Va dimensions to [-π, π]
    wrapped_va = torch.atan2(torch.sin(va_diff), torch.cos(va_diff))
    
    # Concatenate back (non-inplace, gradient-safe)
    wrapped = torch.cat([wrapped_va, vm_diff], dim=1)
    return wrapped


def compute_reconstruction_loss(y_recon: torch.Tensor, y_target: torch.Tensor,
                                 NPred_Va: int = 0) -> torch.Tensor:
    """
    Compute reconstruction loss with angle wrapping for Va dimensions.
    
    Args:
        y_recon: Reconstructed solution [B, output_dim]
        y_target: Target solution [B, output_dim]
        NPred_Va: Number of Va dimensions
    
    Returns:
        Reconstruction loss (MSE)
    """
    diff = y_recon - y_target
    wrapped_diff = wrap_angle_difference(diff, NPred_Va)
    return (wrapped_diff ** 2).mean()


def compute_kl_loss(mean: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    """
    Compute KL divergence loss: KL(q(z|x,y) || N(0,I))
    
    Args:
        mean: [B, latent_dim]
        logvar: [B, latent_dim]
    
    Returns:
        KL loss (scalar)
    """
    # KL = -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    kl = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp(), dim=1)
    return kl.mean()


def compute_1d_loss(z_list: List[torch.Tensor]) -> torch.Tensor:
    """
    Compute one-dimensionality loss (PCA-style).
    
    Given latent codes from the same scene but different preferences,
    penalize variance in directions other than the first principal component.
    
    Args:
        z_list: List of [K, latent_dim] tensors, each for one scene
                with K different preference points
    
    Returns:
        L_1D loss (scalar)
    """
    device = z_list[0].device
    total_loss = torch.tensor(0.0, device=device)
    n_valid = 0
    
    for z_scene in z_list:
        K = z_scene.shape[0]
        if K < 3:  # Need at least 3 points for meaningful PCA
            continue
        
        # Ensure contiguous tensor for eigenvalue computation
        z_scene = z_scene.contiguous()
        
        # Center the data
        z_mean = z_scene.mean(dim=0, keepdim=True)
        z_centered = z_scene - z_mean  # [K, latent_dim]
        
        # Compute covariance matrix
        cov = z_centered.T @ z_centered / (K - 1)  # [latent_dim, latent_dim]
        
        # Make symmetric for numerical stability
        cov = (cov + cov.T) / 2
        
        try:
            # Get eigenvalues (sorted in ascending order)
            eigenvalues = torch.linalg.eigvalsh(cov)  # [latent_dim]
            
            # L_1D = sum of all eigenvalues except the largest
            # Normalized by the largest eigenvalue for scale invariance
            largest_eig = eigenvalues[-1].clamp(min=1e-8)
            other_eigs = eigenvalues[:-1].clamp(min=0)  # Eigenvalues should be non-negative
            
            # Ratio loss: we want other eigenvalues to be small relative to largest
            loss = other_eigs.sum() / largest_eig
            
            total_loss = total_loss + loss
            n_valid += 1
        except Exception:
            # Skip if eigenvalue computation fails
            continue
    
    if n_valid == 0:
        return torch.tensor(0.0, device=device, requires_grad=True)
    
    return total_loss / n_valid


def compute_order_loss(z_list: List[torch.Tensor], 
                       lambda_list: List[torch.Tensor]) -> torch.Tensor:
    """
    Compute monotonic ordering loss.
    
    Ensure that the projection of z onto the principal direction is monotonic
    with respect to λ.
    
    Args:
        z_list: List of [K, latent_dim] tensors, each for one scene
        lambda_list: List of [K] tensors with corresponding normalized λ values
    
    Returns:
        L_order loss (scalar)
    """
    device = z_list[0].device
    total_loss = torch.tensor(0.0, device=device)
    n_valid = 0
    
    for z_scene, lambdas in zip(z_list, lambda_list):
        K = z_scene.shape[0]
        if K < 3:
            continue
        
        # Ensure contiguous tensor
        z_scene = z_scene.contiguous()
        
        # Compute principal direction via SVD
        z_mean = z_scene.mean(dim=0, keepdim=True)
        z_centered = z_scene - z_mean
        
        try:
            _, _, Vh = torch.linalg.svd(z_centered, full_matrices=False)
            principal_dir = Vh[0].clone()  # First right singular vector, clone to avoid view issues
        except Exception:
            continue
        
        # Project onto principal direction
        projections = z_scene @ principal_dir  # [K]
        
        # Compute differences
        dlambda = lambdas[1:] - lambdas[:-1]  # [K-1]
        dproj = projections[1:] - projections[:-1]  # [K-1]
        
        # Hinge loss: penalize when signs don't match
        # We want sign(dlambda) == sign(dproj)
        # Loss = max(0, -dlambda * dproj) encourages same sign
        # Normalize by |dlambda| to handle varying step sizes
        loss = F.relu(-dlambda * dproj / (dlambda.abs() + 1e-8)).mean()
        
        total_loss = total_loss + loss
        n_valid += 1
    
    if n_valid == 0:
        return torch.tensor(0.0, device=device, requires_grad=True)
    
    return total_loss / n_valid


def compute_linearized_vae_loss(
    vae: LinearizedVAE,
    scene: torch.Tensor,
    solutions_by_pref: Dict[float, torch.Tensor],
    lambda_values: List[float],
    sample_indices: torch.Tensor,
    ngt_loss_fn=None,
    config=None
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Compute the complete linearized VAE loss.
    
    Optimized version:
    - Reuses `mean` from VAE forward as z (avoids duplicate encoder calls)
    - Uses tensor operations instead of per-sample Python loops
    
    Args:
        vae: LinearizedVAE model
        scene: [B, input_dim] - Scene features for the batch
        solutions_by_pref: Dict mapping λ -> [N, output_dim] solutions
        lambda_values: List of λ values (sorted)
        sample_indices: [B] - Indices into solutions_by_pref tensors
        ngt_loss_fn: Optional NGT loss function for physics constraints
        config: Configuration object with loss weights
    
    Returns:
        total_loss: Combined loss
        loss_dict: Dictionary of individual loss components
    """
    device = scene.device
    B = scene.shape[0]
    
    # Get loss weights from config
    alpha_kl = getattr(config, 'linearized_vae_alpha_kl', 0.001) if config else 0.001
    beta_1d = getattr(config, 'linearized_vae_beta_1d', 0.1) if config else 0.1
    gamma_order = getattr(config, 'linearized_vae_gamma_order', 0.01) if config else 0.01
    delta_ngt = getattr(config, 'linearized_vae_delta_ngt', 0.001) if config else 0.001
    
    # Normalize lambda values
    lambda_max = max(lambda_values)
    
    # Sample multiple preferences per scene for linearization constraint
    n_pref_samples = min(len(lambda_values), getattr(config, 'linearized_vae_n_pref_samples', 8) if config else 8)
    sampled_lambdas = sorted(lambda_values[::max(1, len(lambda_values) // n_pref_samples)])[:n_pref_samples]
    K = len(sampled_lambdas)
    
    # Pre-allocate tensor to collect z values: [B, K, latent_dim]
    # This avoids per-sample Python loops
    z_all = []
    lambda_normalized_list = []
    
    # Accumulate losses
    total_rec_loss = 0.0
    total_kl_loss = 0.0
    total_ngt_loss = 0.0
    n_samples = 0
    
    for k, lc in enumerate(sampled_lambdas):
        # Normalized lambda value
        lc_norm = lc / lambda_max
        lambda_normalized_list.append(lc_norm)
        
        # Get solutions for this preference
        solutions = solutions_by_pref[lc].to(device)  # [N, output_dim]
        batch_solutions = solutions[sample_indices].clone()  # [B, output_dim]
        
        # Create preference tensor
        pref = torch.full((B, 1), lc_norm, device=device, dtype=torch.float32)
        
        # Forward pass - mean is already the latent code z when use_mean=True
        y_recon, mean, logvar = vae(scene, batch_solutions, pref)
        
        # Reconstruction loss
        rec_loss = compute_reconstruction_loss(y_recon, batch_solutions, vae.NPred_Va)
        total_rec_loss = total_rec_loss + rec_loss
        
        # KL loss
        kl_loss = compute_kl_loss(mean, logvar)
        total_kl_loss = total_kl_loss + kl_loss
        
        # NGT loss (optional)
        if ngt_loss_fn is not None and delta_ngt > 0:
            ngt_loss, _ = ngt_loss_fn(y_recon, scene, pref)
            total_ngt_loss = total_ngt_loss + ngt_loss
        
        # OPTIMIZATION: Use mean directly as z (no separate encode call!)
        # mean is [B, latent_dim], clone to avoid gradient issues
        z_all.append(mean.clone())
        
        n_samples += 1
    
    # Average per-preference losses
    L_rec = total_rec_loss / n_samples
    L_kl = total_kl_loss / n_samples
    L_ngt = total_ngt_loss / n_samples if isinstance(total_ngt_loss, torch.Tensor) and total_ngt_loss > 0 else torch.tensor(0.0, device=device)
    
    # Stack z values: [K, B, latent_dim] -> transpose to [B, K, latent_dim]
    z_stacked = torch.stack(z_all, dim=0)  # [K, B, latent_dim]
    z_batched = z_stacked.permute(1, 0, 2)  # [B, K, latent_dim]
    
    # Lambda tensor for all samples: [K]
    lambda_tensor = torch.tensor(lambda_normalized_list, device=device, dtype=torch.float32)
    
    # Compute linearization losses using batched operations
    if K >= 3:
        L_1d = compute_1d_loss_batched(z_batched)
        L_order = compute_order_loss_batched(z_batched, lambda_tensor)
    else:
        L_1d = torch.tensor(0.0, device=device, requires_grad=True)
        L_order = torch.tensor(0.0, device=device, requires_grad=True)
    
    # Total loss
    total_loss = L_rec + alpha_kl * L_kl + beta_1d * L_1d + gamma_order * L_order + delta_ngt * L_ngt
    
    # Loss dictionary for logging
    loss_dict = {
        'L_rec': L_rec.item(),
        'L_kl': L_kl.item(),
        'L_1d': L_1d.item() if torch.is_tensor(L_1d) else L_1d,
        'L_order': L_order.item() if torch.is_tensor(L_order) else L_order,
        'L_ngt': L_ngt.item() if torch.is_tensor(L_ngt) else L_ngt,
        'total': total_loss.item()
    }
    
    return total_loss, loss_dict


def compute_1d_loss_batched(z_batched: torch.Tensor) -> torch.Tensor:
    """
    Compute one-dimensionality loss (PCA-style) for batched input.
    
    Optimized version that processes all samples in parallel.
    
    Args:
        z_batched: [B, K, latent_dim] - Latent codes for B samples, K preferences each
    
    Returns:
        L_1D loss (scalar)
    """
    B, K, D = z_batched.shape
    device = z_batched.device
    
    if K < 3:
        return torch.tensor(0.0, device=device, requires_grad=True)
    
    # Center the data: [B, K, D]
    z_mean = z_batched.mean(dim=1, keepdim=True)  # [B, 1, D]
    z_centered = z_batched - z_mean  # [B, K, D]
    
    # Compute covariance matrix for each sample: [B, D, D]
    # cov = z_centered^T @ z_centered / (K-1)
    cov = torch.bmm(z_centered.transpose(1, 2), z_centered) / (K - 1)  # [B, D, D]
    
    # Make symmetric for numerical stability
    cov = (cov + cov.transpose(1, 2)) / 2
    
    # Compute eigenvalues for each sample
    # Use try-except to handle potential numerical issues
    try:
        eigenvalues = torch.linalg.eigvalsh(cov)  # [B, D], sorted ascending
        
        # L_1D = sum of all eigenvalues except the largest, normalized by largest
        largest_eig = eigenvalues[:, -1].clamp(min=1e-8)  # [B]
        other_eigs = eigenvalues[:, :-1].clamp(min=0)  # [B, D-1]
        
        # Ratio loss per sample, then average
        loss_per_sample = other_eigs.sum(dim=1) / largest_eig  # [B]
        
        return loss_per_sample.mean()
    except Exception:
        # Fallback to per-sample computation if batched fails
        return compute_1d_loss([z_batched[i] for i in range(B)])


def compute_order_loss_batched(z_batched: torch.Tensor, 
                               lambda_tensor: torch.Tensor) -> torch.Tensor:
    """
    Compute monotonic ordering loss for batched input.
    
    Optimized version that processes all samples in parallel.
    
    Args:
        z_batched: [B, K, latent_dim] - Latent codes for B samples, K preferences each
        lambda_tensor: [K] - Normalized lambda values (same for all samples)
    
    Returns:
        L_order loss (scalar)
    """
    B, K, D = z_batched.shape
    device = z_batched.device
    
    if K < 3:
        return torch.tensor(0.0, device=device, requires_grad=True)
    
    # Center the data: [B, K, D]
    z_mean = z_batched.mean(dim=1, keepdim=True)  # [B, 1, D]
    z_centered = z_batched - z_mean  # [B, K, D]
    
    # Compute principal direction via SVD for each sample
    try:
        # SVD: z_centered = U @ S @ Vh
        # We want the first right singular vector (principal direction)
        _, _, Vh = torch.linalg.svd(z_centered, full_matrices=False)  # Vh: [B, K, D]
        principal_dir = Vh[:, 0, :]  # [B, D] - first right singular vector for each sample
        
        # Project onto principal direction: [B, K]
        projections = torch.bmm(z_batched, principal_dir.unsqueeze(-1)).squeeze(-1)  # [B, K]
        
        # Compute differences
        dlambda = lambda_tensor[1:] - lambda_tensor[:-1]  # [K-1]
        dproj = projections[:, 1:] - projections[:, :-1]  # [B, K-1]
        
        # Hinge loss: penalize when signs don't match
        # We want sign(dlambda) == sign(dproj)
        # dlambda is [K-1], broadcast to [B, K-1]
        dlambda_expanded = dlambda.unsqueeze(0).expand(B, -1)  # [B, K-1]
        
        # Loss = max(0, -dlambda * dproj) / |dlambda|
        loss_per_pair = F.relu(-dlambda_expanded * dproj / (dlambda_expanded.abs() + 1e-8))  # [B, K-1]
        
        # Average over pairs and samples
        return loss_per_pair.mean()
    except Exception:
        # Fallback to per-sample computation if batched fails
        z_list = [z_batched[i] for i in range(B)]
        lambda_list = [lambda_tensor for _ in range(B)]
        return compute_order_loss(z_list, lambda_list)


# ==================== Visualization Utilities ====================

def visualize_latent_linearity(vae: LinearizedVAE, 
                               scene: torch.Tensor,
                               solutions_by_pref: Dict[float, torch.Tensor],
                               sample_idx: int = 0,
                               save_path: str = None):
    """
    Visualize the latent space to check if it's approximately one-dimensional.
    
    Args:
        vae: Trained LinearizedVAE
        scene: [N, input_dim] - Scene features
        solutions_by_pref: Dict mapping λ -> [N, output_dim]
        sample_idx: Which sample to visualize
        save_path: Path to save the figure
    """
    import matplotlib.pyplot as plt
    from sklearn.decomposition import PCA
    
    vae.eval()
    device = next(vae.parameters()).device
    
    # Collect latent codes for this scene across all preferences
    lambda_values = sorted(solutions_by_pref.keys())
    z_list = []
    
    with torch.no_grad():
        for lc in lambda_values:
            sol = solutions_by_pref[lc][sample_idx:sample_idx+1].to(device)
            s = scene[sample_idx:sample_idx+1].to(device)
            pref = torch.tensor([[lc / max(lambda_values)]], device=device)
            
            z = vae.encode(s, sol, pref, use_mean=True)
            z_list.append(z.cpu().numpy().flatten())
    
    z_array = np.array(z_list)  # [K, latent_dim]
    
    # PCA to 2D for visualization
    if z_array.shape[1] > 2:
        pca = PCA(n_components=2)
        z_2d = pca.fit_transform(z_array)
        explained_var = pca.explained_variance_ratio_
    else:
        z_2d = z_array
        explained_var = [1.0, 0.0]
    
    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot 1: 2D projection colored by lambda
    colors = plt.cm.viridis(np.linspace(0, 1, len(lambda_values)))
    ax1 = axes[0]
    for i, (lc, z) in enumerate(zip(lambda_values, z_2d)):
        ax1.scatter(z[0], z[1], c=[colors[i]], s=50, label=f'λ={lc:.1f}')
    ax1.plot(z_2d[:, 0], z_2d[:, 1], 'k--', alpha=0.3)
    ax1.set_xlabel(f'PC1 ({explained_var[0]:.1%} var)')
    ax1.set_ylabel(f'PC2 ({explained_var[1]:.1%} var)')
    ax1.set_title(f'Latent Space (Sample {sample_idx})')
    ax1.legend(loc='best', fontsize=6, ncol=2)
    
    # Plot 2: First PC vs lambda (should be monotonic and linear)
    ax2 = axes[1]
    ax2.scatter(lambda_values, z_2d[:, 0], c=colors, s=50)
    ax2.plot(lambda_values, z_2d[:, 0], 'b-', alpha=0.5)
    ax2.set_xlabel('Lambda (preference)')
    ax2.set_ylabel('First Principal Component')
    ax2.set_title('Linearity Check: PC1 vs Lambda')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved visualization to {save_path}")
    
    plt.close()
    
    return explained_var


# For numpy operations in visualization
import numpy as np
import random
