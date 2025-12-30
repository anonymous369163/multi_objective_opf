"""
Latent Flow Matching for Preference Flow Learning

This module implements Flow Matching in the latent space of the Linearized VAE.
The flow model learns the velocity field dz/dr where r is the normalized preference
coordinate and z is the latent variable.

Stage 2 of the VAE+Flow approach:
1. Freeze the trained Linearized VAE
2. Encode training data to latent space
3. Learn velocity field v(s, z, r) = dz/dr in latent space
4. Inference: integrate in latent space, then decode

Author: Auto-generated from VAE+Flow plan
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional, List
import math


class LatentVelocityMLP(nn.Module):
    """
    MLP that predicts velocity in the latent space.
    
    Input: [scene_features, z, r] where:
        - scene_features: Condition features (load data)
        - z: Current latent position
        - r: Normalized preference coordinate (0 to 1)
    
    Output: v = dz/dr (velocity in latent space)
    """
    
    def __init__(self, scene_dim: int, latent_dim: int, hidden_dim: int = 256,
                 num_layers: int = 4, act: str = 'silu'):
        """
        Initialize the velocity MLP.
        
        Args:
            scene_dim: Dimension of scene features
            latent_dim: Dimension of latent space (both input z and output v)
            hidden_dim: Hidden layer dimension
            num_layers: Number of hidden layers
            act: Activation function
        """
        super().__init__()
        
        act_list = {'relu': nn.ReLU(), 'silu': nn.SiLU(), 'gelu': nn.GELU(),
                    'tanh': nn.Tanh(), 'softplus': nn.Softplus()}
        act_fn = act_list.get(act, nn.SiLU())
        
        self.scene_dim = scene_dim
        self.latent_dim = latent_dim
        
        # Scene embedding
        self.scene_embed = nn.Sequential(
            nn.Linear(scene_dim, hidden_dim),
            act_fn,
            nn.LayerNorm(hidden_dim)
        )
        
        # Latent embedding
        self.z_embed = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            act_fn
        )
        
        # Time/preference embedding (sinusoidal + MLP)
        self.r_embed = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            act_fn
        )
        
        # Main velocity network
        layers = []
        layers.append(nn.Linear(hidden_dim * 3, hidden_dim))
        layers.append(act_fn)
        
        for _ in range(num_layers - 2):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(act_fn)
            layers.append(nn.LayerNorm(hidden_dim))
        
        layers.append(nn.Linear(hidden_dim, latent_dim))
        
        self.velocity_net = nn.Sequential(*layers)
    
    def get_timestep_embedding(self, r: torch.Tensor, dim: int) -> torch.Tensor:
        """
        Create sinusoidal timestep embeddings (from DDPM).
        
        Args:
            r: [B, 1] normalized preference coordinate
            dim: Embedding dimension
        
        Returns:
            emb: [B, dim] sinusoidal embedding
        """
        half_dim = dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=r.device, dtype=r.dtype) * -emb)
        emb = r * emb.unsqueeze(0)  # [B, half_dim]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)  # [B, dim]
        return emb
    
    def forward(self, scene: torch.Tensor, z: torch.Tensor, 
                r: torch.Tensor) -> torch.Tensor:
        """
        Predict velocity at given (scene, z, r).
        
        Args:
            scene: [B, scene_dim] - Scene features
            z: [B, latent_dim] - Current latent position
            r: [B, 1] or [B] - Normalized preference coordinate
        
        Returns:
            v: [B, latent_dim] - Predicted velocity dz/dr
        """
        if r.dim() == 1:
            r = r.unsqueeze(-1)
        
        # Embed scene
        scene_feat = self.scene_embed(scene)  # [B, H]
        
        # Embed latent
        z_feat = self.z_embed(z)  # [B, H]
        
        # Embed preference coordinate (timestep-style)
        r_emb = self.get_timestep_embedding(r, scene_feat.shape[-1])  # [B, H]
        r_feat = self.r_embed(r_emb)  # [B, H]
        
        # Concatenate and predict velocity
        combined = torch.cat([scene_feat, z_feat, r_feat], dim=-1)  # [B, 3H]
        v = self.velocity_net(combined)  # [B, latent_dim]
        
        return v


class LatentFlowModel(nn.Module):
    """
    Complete Latent Flow Matching model.
    
    This model operates in the latent space of a trained Linearized VAE.
    It learns the velocity field v(s, z, r) = dz/dr and can integrate
    from z(r=0) to z(r=r_target) using various ODE solvers.
    """
    
    def __init__(self, scene_dim: int, latent_dim: int, hidden_dim: int = 256,
                 num_layers: int = 4, act: str = 'silu'):
        """
        Initialize the Latent Flow Model.
        
        Args:
            scene_dim: Dimension of scene features (input_dim of VAE)
            latent_dim: Dimension of latent space
            hidden_dim: Hidden dimension for velocity MLP
            num_layers: Number of layers in velocity MLP
            act: Activation function
        """
        super().__init__()
        
        self.scene_dim = scene_dim
        self.latent_dim = latent_dim
        
        # Velocity network
        self.velocity_net = LatentVelocityMLP(
            scene_dim=scene_dim,
            latent_dim=latent_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            act=act
        )
    
    def predict_velocity(self, scene: torch.Tensor, z: torch.Tensor,
                         r: torch.Tensor) -> torch.Tensor:
        """Predict velocity at (scene, z, r)."""
        return self.velocity_net(scene, z, r)
    
    def forward(self, scene: torch.Tensor, z: torch.Tensor,
                r: torch.Tensor) -> torch.Tensor:
        """Forward pass: predict velocity."""
        return self.predict_velocity(scene, z, r)
    
    def integrate(self, scene: torch.Tensor, z_start: torch.Tensor,
                  r_start: float, r_end: float, num_steps: int = 20,
                  method: str = 'heun') -> torch.Tensor:
        """
        Integrate the ODE from r_start to r_end.
        
        dz/dr = v(s, z, r)
        z(r_end) = z(r_start) + ∫_{r_start}^{r_end} v(s, z(r), r) dr
        
        Uses Heun method (RK2) by default for consistency between training and inference.
        
        Args:
            scene: [B, scene_dim] - Scene features
            z_start: [B, latent_dim] - Starting latent position
            r_start: Starting preference coordinate
            r_end: Ending preference coordinate
            num_steps: Number of integration steps
            method: ODE solver ('euler', 'heun', 'rk4'), default='heun'
        
        Returns:
            z_end: [B, latent_dim] - Final latent position
        """
        self.eval()
        device = scene.device
        batch_size = scene.shape[0]
        
        dr = (r_end - r_start) / num_steps
        z = z_start.clone()
        r = r_start
        
        with torch.no_grad():
            for step in range(num_steps):
                r_tensor = torch.full((batch_size, 1), r, device=device, dtype=scene.dtype)
                
                if method.lower() == 'euler':
                    v = self.predict_velocity(scene, z, r_tensor)
                    z = z + dr * v
                
                elif method.lower() == 'heun':
                    # Predictor step
                    v1 = self.predict_velocity(scene, z, r_tensor)
                    z_pred = z + dr * v1
                    
                    # Corrector step
                    r_next = torch.full((batch_size, 1), r + dr, device=device, dtype=scene.dtype)
                    v2 = self.predict_velocity(scene, z_pred, r_next)
                    z = z + dr * 0.5 * (v1 + v2)
                
                elif method.lower() == 'rk4':
                    # RK4 method
                    r_half = torch.full((batch_size, 1), r + dr/2, device=device, dtype=scene.dtype)
                    r_next = torch.full((batch_size, 1), r + dr, device=device, dtype=scene.dtype)
                    
                    k1 = self.predict_velocity(scene, z, r_tensor)
                    k2 = self.predict_velocity(scene, z + dr/2 * k1, r_half)
                    k3 = self.predict_velocity(scene, z + dr/2 * k2, r_half)
                    k4 = self.predict_velocity(scene, z + dr * k3, r_next)
                    
                    z = z + dr / 6 * (k1 + 2*k2 + 2*k3 + k4)
                
                else:
                    raise ValueError(f"Unknown integration method: {method}")
                
                r += dr
        
        return z
    
    def integrate_trajectory(self, scene: torch.Tensor, z_start: torch.Tensor,
                             r_values: List[float], method: str = 'heun') -> torch.Tensor:
        """
        Integrate through a sequence of r values and return trajectory.
        
        Uses Heun method (RK2) by default for consistency with training.
        
        Args:
            scene: [B, scene_dim]
            z_start: [B, latent_dim]
            r_values: List of r values (sorted ascending)
            method: ODE solver ('euler', 'heun'), default='heun'
        
        Returns:
            z_trajectory: [B, len(r_values), latent_dim] - Latent trajectory
        """
        self.eval()
        device = scene.device
        batch_size = scene.shape[0]
        
        trajectory = [z_start.clone()]
        z = z_start.clone()
        
        with torch.no_grad():
            for i in range(len(r_values) - 1):
                r_current = r_values[i]
                r_next = r_values[i + 1]
                dr = r_next - r_current
                
                r_tensor = torch.full((batch_size, 1), r_current, device=device, dtype=scene.dtype)
                
                if method.lower() == 'euler':
                    v = self.predict_velocity(scene, z, r_tensor)
                    z = z + dr * v
                
                elif method.lower() == 'heun':
                    v1 = self.predict_velocity(scene, z, r_tensor)
                    z_pred = z + dr * v1
                    r_next_tensor = torch.full((batch_size, 1), r_next, device=device, dtype=scene.dtype)
                    v2 = self.predict_velocity(scene, z_pred, r_next_tensor)
                    z = z + dr * 0.5 * (v1 + v2)
                
                else:
                    raise ValueError(f"Unknown method: {method}")
                
                trajectory.append(z.clone())
        
        return torch.stack(trajectory, dim=1)  # [B, K, latent_dim]


class LatentFlowWithVAE(nn.Module):
    """
    Complete VAE+Flow model for preference flow learning.
    
    This class combines the trained Linearized VAE and Latent Flow Model
    for end-to-end inference.
    """
    
    def __init__(self, vae, flow_model):
        """
        Initialize with pre-trained components.
        
        Args:
            vae: Trained LinearizedVAE
            flow_model: Trained LatentFlowModel
        """
        super().__init__()
        self.vae = vae
        self.flow_model = flow_model
        
        # Freeze VAE by default
        for param in self.vae.parameters():
            param.requires_grad = False
    
    def encode(self, scene: torch.Tensor, solution: torch.Tensor,
               pref: torch.Tensor) -> torch.Tensor:
        """Encode to latent space using VAE."""
        return self.vae.encode(scene, solution, pref, use_mean=True)
    
    def decode(self, scene: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """Decode from latent space using VAE."""
        return self.vae.decode(scene, z)
    
    def predict(self, scene: torch.Tensor, z_start: torch.Tensor,
                r_start: float, r_target: float, 
                num_steps: int = 20, method: str = 'heun') -> torch.Tensor:
        """
        Predict solution at target preference.
        
        Args:
            scene: Scene features
            z_start: Starting latent (from lambda=0 solution)
            r_start: Starting preference coordinate (normalized)
            r_target: Target preference coordinate (normalized)
            num_steps: Number of integration steps
            method: ODE solver
        
        Returns:
            solution: Predicted solution at target preference
        """
        # Integrate in latent space
        z_target = self.flow_model.integrate(
            scene, z_start, r_start, r_target, num_steps, method
        )
        
        # Decode to solution space
        solution = self.decode(scene, z_target)
        
        return solution
    
    def predict_trajectory(self, scene: torch.Tensor, z_start: torch.Tensor,
                           r_values: List[float], method: str = 'heun') -> torch.Tensor:
        """
        Predict solutions along a preference trajectory.
        
        Args:
            scene: Scene features
            z_start: Starting latent
            r_values: List of preference coordinates
            method: ODE solver
        
        Returns:
            solutions: [B, K, output_dim] - Solutions at each preference
        """
        # Get latent trajectory
        z_trajectory = self.flow_model.integrate_trajectory(
            scene, z_start, r_values, method
        )
        
        # Decode each point
        B, K, D = z_trajectory.shape
        z_flat = z_trajectory.view(B * K, D)
        scene_expanded = scene.unsqueeze(1).expand(-1, K, -1).contiguous().view(B * K, -1)
        
        solutions_flat = self.decode(scene_expanded, z_flat)
        solutions = solutions_flat.view(B, K, -1)
        
        return solutions


# ==================== Training Loss Functions ====================

def precompute_latent_cache(
    vae,
    scene: torch.Tensor,
    solutions_by_pref: Dict[float, torch.Tensor],
    lambda_values: List[float],
    sample_indices: torch.Tensor,
    device: torch.device
) -> Tuple[Dict[float, torch.Tensor], List[float]]:
    """
    Pre-compute and cache all latent codes for the given samples.
    
    This avoids redundant encoding during training.
    
    Args:
        vae: Trained LinearizedVAE (frozen)
        scene: [B, input_dim] - Scene features
        solutions_by_pref: Dict mapping λ -> [N, output_dim]
        lambda_values: Sorted list of λ values
        sample_indices: [B] - Indices into solutions
        device: Device
    
    Returns:
        z_cache: Dict mapping λ -> [B, latent_dim] cached latent codes
        r_values: List of normalized preference coordinates
    """
    B = scene.shape[0]
    
    # Normalize lambda values to [0, 1]
    lambda_min = min(lambda_values)
    lambda_max = max(lambda_values)
    r_values = [(lc - lambda_min) / (lambda_max - lambda_min) if lambda_max > lambda_min else 0.0
                for lc in lambda_values]
    
    # Encode all solutions to latent space (with frozen VAE)
    vae.eval()
    z_cache = {}
    with torch.no_grad():
        for lc, r in zip(lambda_values, r_values):
            sol = solutions_by_pref[lc].to(device)[sample_indices].clone()
            pref = torch.full((B, 1), r, device=device, dtype=scene.dtype)
            z = vae.encode(scene, sol, pref, use_mean=True)
            z_cache[lc] = z.clone()  # Clone to avoid any view issues
    
    return z_cache, r_values


def compute_latent_flow_loss(
    flow_model: LatentFlowModel,
    vae,  # LinearizedVAE
    scene: torch.Tensor,
    solutions_by_pref: Dict[float, torch.Tensor],
    lambda_values: List[float],
    sample_indices: torch.Tensor,
    config=None,
    z_cache: Dict[float, torch.Tensor] = None,
    r_values: List[float] = None
) -> Tuple[torch.Tensor, Dict[str, float], Dict[float, torch.Tensor], List[float]]:
    """
    Compute training loss for the latent flow model.
    
    Loss components:
    - L_v: Velocity MSE at endpoints ||v_pred(z_k, r_k) - v_target||^2
    - L_z1: One-step state consistency ||(z_k + dr * v_pred) - z_{k+1}||^2
    - L_fm: Flow-Matching loss for non-adjacent pairs with interpolated points
            Sample (k, k+m), t~U(0,1), z_t = (1-t)*z_k + t*z_{k+m}
            ||v_pred(z_t, r_t) - (z_{k+m} - z_k) / (r_{k+m} - r_k)||^2
    
    Args:
        flow_model: LatentFlowModel to train
        vae: Trained LinearizedVAE (frozen, used to encode solutions)
        scene: [B, input_dim] - Scene features
        solutions_by_pref: Dict mapping λ -> [N, output_dim]
        lambda_values: Sorted list of λ values
        sample_indices: [B] - Indices into solutions
        config: Configuration object
        z_cache: Pre-computed latent cache (optional, avoids redundant encoding)
        r_values: Pre-computed normalized r values (optional)
    
    Returns:
        loss: Total loss
        loss_dict: Dictionary with loss components
        z_cache: Computed/reused latent cache (for reuse in rollout)
        r_values: Computed/reused r values
    """
    device = scene.device
    B = scene.shape[0]
    
    # Use cache or compute fresh
    if z_cache is None or r_values is None:
        z_cache, r_values = precompute_latent_cache(
            vae, scene, solutions_by_pref, lambda_values, sample_indices, device
        )
    
    # Loss weights
    beta_z1 = getattr(config, 'latent_flow_beta_z1', 1.0) if config else 1.0
    alpha_fm = getattr(config, 'latent_flow_alpha_fm', 1.0) if config else 1.0
    
    # Number of samples
    n_pair_samples = getattr(config, 'latent_flow_n_pair_samples', 4) if config else 4
    n_fm_samples = getattr(config, 'latent_flow_n_fm_samples', 4) if config else 4
    
    n_lambdas = len(lambda_values)
    
    # Build z tensor for efficient indexing: [K, B, latent_dim]
    z_tensor = torch.stack([z_cache[lc] for lc in lambda_values], dim=0)  # [K, B, D]
    r_tensor_all = torch.tensor(r_values, device=device, dtype=scene.dtype)  # [K]
    
    # =====================================================================
    # Part 1: Adjacent pair loss (L_v + L_z1) - for endpoint supervision
    # =====================================================================
    total_loss_v = torch.tensor(0.0, device=device)
    total_loss_z1 = torch.tensor(0.0, device=device)
    n_pairs = 0
    
    # Randomly sample adjacent pair indices
    pair_indices = torch.randint(0, n_lambdas - 1, (n_pair_samples,))
    
    for idx in pair_indices:
        k = idx.item()
        r_k = r_values[k]
        r_k1 = r_values[k + 1]
        
        z_k = z_tensor[k]        # [B, latent_dim]
        z_k1 = z_tensor[k + 1]   # [B, latent_dim]
        
        # Ground truth velocity
        dr = r_k1 - r_k
        if abs(dr) < 1e-8:
            continue
        v_target = (z_k1 - z_k) / dr  # [B, latent_dim]
        
        # Predicted velocity at endpoint
        r_k_input = torch.full((B, 1), r_k, device=device, dtype=scene.dtype)
        v_pred = flow_model.predict_velocity(scene, z_k, r_k_input)
        
        # L_v: Velocity MSE loss
        loss_v = F.mse_loss(v_pred, v_target)
        total_loss_v = total_loss_v + loss_v
        
        # L_z1: One-step state consistency loss
        z_pred_one_step = z_k + dr * v_pred
        loss_z1 = F.mse_loss(z_pred_one_step, z_k1)
        total_loss_z1 = total_loss_z1 + loss_z1
        
        n_pairs += 1
    
    # =====================================================================
    # Part 2: Flow-Matching loss (L_fm) - for bridge interpolation
    # Sample non-adjacent pairs (k, k+m) where m >= 1, then sample t~U(0,1)
    # This trains the model to predict correct velocity at any interpolated point
    # =====================================================================
    total_loss_fm = torch.tensor(0.0, device=device)
    n_fm = 0
    
    # Minimum gap between sampled indices (set to 1 means allow adjacent too)
    min_gap = getattr(config, 'latent_flow_fm_min_gap', 1) if config else 1
    # Maximum gap (set to n_lambdas-1 to allow full range)
    max_gap = getattr(config, 'latent_flow_fm_max_gap', n_lambdas - 1) if config else n_lambdas - 1
    max_gap = min(max_gap, n_lambdas - 1)
    
    for _ in range(n_fm_samples):
        # Sample start index k and gap m
        # k can be from 0 to n_lambdas - 1 - min_gap
        max_k = n_lambdas - 1 - min_gap
        if max_k < 0:
            continue
        k = torch.randint(0, max_k + 1, (1,)).item()
        
        # Sample gap m from [min_gap, min(max_gap, n_lambdas - 1 - k)]
        max_m = min(max_gap, n_lambdas - 1 - k)
        if max_m < min_gap:
            continue
        m = torch.randint(min_gap, max_m + 1, (1,)).item()
        
        k_end = k + m
        
        z_k = z_tensor[k]          # [B, latent_dim]
        z_km = z_tensor[k_end]     # [B, latent_dim]
        r_k = r_values[k]
        r_km = r_values[k_end]
        
        dr_total = r_km - r_k
        if abs(dr_total) < 1e-8:
            continue
        
        # Target velocity (constant along the bridge in ideal linear latent space)
        v_target_fm = (z_km - z_k) / dr_total  # [B, latent_dim]
        
        # Sample t ~ U(0, 1) for each sample in batch (more diversity)
        t = torch.rand(B, 1, device=device, dtype=scene.dtype)  # [B, 1]
        
        # Interpolated latent and r
        z_t = (1 - t) * z_k + t * z_km  # [B, latent_dim]
        r_t = (1 - t.squeeze(-1)) * r_k + t.squeeze(-1) * r_km  # [B]
        r_t_input = r_t.unsqueeze(-1)  # [B, 1]
        
        # Predicted velocity at interpolated point
        v_pred_fm = flow_model.predict_velocity(scene, z_t, r_t_input)
        
        # L_fm: Flow-Matching loss
        loss_fm = F.mse_loss(v_pred_fm, v_target_fm)
        total_loss_fm = total_loss_fm + loss_fm
        n_fm += 1
    
    # =====================================================================
    # Combine losses
    # =====================================================================
    loss_dict = {}
    total_loss = torch.tensor(0.0, device=device, requires_grad=True)
    
    if n_pairs > 0:
        avg_loss_v = total_loss_v / n_pairs
        avg_loss_z1 = total_loss_z1 / n_pairs
        total_loss = avg_loss_v + beta_z1 * avg_loss_z1
        loss_dict['L_v'] = avg_loss_v.item()
        loss_dict['L_z1'] = avg_loss_z1.item()
    else:
        loss_dict['L_v'] = 0.0
        loss_dict['L_z1'] = 0.0
    
    if n_fm > 0:
        avg_loss_fm = total_loss_fm / n_fm
        total_loss = total_loss + alpha_fm * avg_loss_fm
        loss_dict['L_fm'] = avg_loss_fm.item()
    else:
        loss_dict['L_fm'] = 0.0
    
    loss_dict['total'] = total_loss.item() if torch.is_tensor(total_loss) else total_loss
    
    return total_loss, loss_dict, z_cache, r_values


def heun_step(flow_model: LatentFlowModel, scene: torch.Tensor, 
              z: torch.Tensor, r: float, dr: float, device: torch.device) -> torch.Tensor:
    """
    Heun's method (RK2) for one integration step.
    
    More accurate than Euler, consistent between training and inference.
    
    Args:
        flow_model: LatentFlowModel
        scene: [B, input_dim]
        z: [B, latent_dim] current state
        r: Current normalized preference
        dr: Step size
        device: Device
    
    Returns:
        z_next: [B, latent_dim] next state
    """
    B = scene.shape[0]
    
    # Predictor step (Euler)
    r_tensor = torch.full((B, 1), r, device=device, dtype=scene.dtype)
    v1 = flow_model.predict_velocity(scene, z, r_tensor)
    z_euler = z + dr * v1
    
    # Corrector step
    r_next_tensor = torch.full((B, 1), r + dr, device=device, dtype=scene.dtype)
    v2 = flow_model.predict_velocity(scene, z_euler, r_next_tensor)
    
    # Average velocities (Heun)
    z_next = z + dr * 0.5 * (v1 + v2)
    
    return z_next


def compute_latent_flow_loss_with_rollout(
    flow_model: LatentFlowModel,
    vae,
    scene: torch.Tensor,
    solutions_by_pref: Dict[float, torch.Tensor],
    lambda_values: List[float],
    sample_indices: torch.Tensor,
    config=None
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Compute training loss with multi-step rollout for regularization.
    
    L = L_v + β * L_z1 + γ * L_rollout
    
    Key improvements:
    1. Uses Heun method (RK2) for rollout - consistent with inference
    2. Includes one-step state consistency loss (L_z1)
    3. Reuses z_cache from base loss computation (no redundant encoding)
    
    Args:
        flow_model: LatentFlowModel to train
        vae: Trained LinearizedVAE (frozen)
        scene: [B, input_dim]
        solutions_by_pref: Dict mapping λ -> [N, output_dim]
        lambda_values: Sorted list of λ values
        sample_indices: [B] - Indices
        config: Configuration object
    
    Returns:
        total_loss: Combined loss
        loss_dict: Dictionary with all loss components
    """
    device = scene.device
    B = scene.shape[0]
    
    # First compute base loss (L_v + L_z1) and get z_cache
    loss_base, loss_dict, z_cache, r_values = compute_latent_flow_loss(
        flow_model, vae, scene, solutions_by_pref, 
        lambda_values, sample_indices, config
    )
    
    # Rollout weight
    gamma_rollout = getattr(config, 'latent_flow_gamma_rollout', 0.1) if config else 0.1
    
    if gamma_rollout <= 0:
        return loss_base, loss_dict
    
    # Use Heun method for rollout (consistent with inference)
    use_heun = getattr(config, 'latent_flow_rollout_use_heun', True) if config else True
    
    # Build z_gt tensor from cache (no redundant encoding!)
    z_gt_list = [z_cache[lc] for lc in lambda_values]
    z_gt = torch.stack(z_gt_list, dim=1)  # [B, K, latent_dim]
    
    # Rollout from z_0
    rollout_horizon = getattr(config, 'latent_flow_rollout_horizon', 5) if config else 5
    rollout_horizon = min(rollout_horizon, len(lambda_values))
    
    z_start = z_gt[:, 0, :]  # [B, latent_dim]
    r_rollout = r_values[:rollout_horizon]
    
    # Need gradients for rollout
    flow_model.train()
    z_rolled = z_start.clone()
    rollout_loss = torch.tensor(0.0, device=device)
    
    for i in range(rollout_horizon - 1):
        dr = r_rollout[i + 1] - r_rollout[i]
        
        if use_heun:
            # Heun step (RK2) - same as inference
            z_rolled = heun_step(flow_model, scene, z_rolled, r_rollout[i], dr, device)
        else:
            # Euler step (for comparison/ablation)
            r_tensor = torch.full((B, 1), r_rollout[i], device=device, dtype=scene.dtype)
            v = flow_model.predict_velocity(scene, z_rolled, r_tensor)
            z_rolled = z_rolled + dr * v
        
        # Compare with ground truth
        z_gt_i = z_gt[:, i + 1, :]
        rollout_loss = rollout_loss + F.mse_loss(z_rolled, z_gt_i)
    
    rollout_loss = rollout_loss / (rollout_horizon - 1) if rollout_horizon > 1 else rollout_loss
    
    # Total loss: L = L_base + γ * L_rollout
    total_loss = loss_base + gamma_rollout * rollout_loss
    
    loss_dict['L_rollout'] = rollout_loss.item() if torch.is_tensor(rollout_loss) else rollout_loss
    loss_dict['total'] = total_loss.item()
    
    return total_loss, loss_dict
