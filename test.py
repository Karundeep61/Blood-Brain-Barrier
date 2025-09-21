#!/usr/bin/env python3
import os, time, math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import imageio
from tqdm import trange

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

# Problem domain (meters and seconds are not enforced here; this is scaled non-dimensional)
Lx, Ly, Lz = 1.0, 0.6, 0.6
T_final = 0.03

# PDE parameters (tune for realism)
u_blood = 1e-3  # advection velocity in blood region (m/s) (used in loss)
D_water = 6e-10
D_brain = 0.6 * D_water
D_barrier_internal = 1e-12
vessel_frac = 0.28
barrier_frac = 0.12

# Barrier geometry in x (fractions)
vessel_x = vessel_frac * Lx
barrier_x = barrier_frac * Lx
barrier_start_x = vessel_x
barrier_end_x = vessel_x + barrier_x

# Drug parameters (example: fentanyl-like)
drug = {
    "name": "Fentanyl",
    "C0": 1.0,
    "fu": 0.15,
    "Vmax": 1e-6,
    "Km": 1e-3,
    "P": 1e-4  # permeability-like scalar for interface terms (tune)
}

# PINN hyperparams
layers = [4, 128, 128, 128, 128, 1]
lr = 1e-3
epochs_adam = 6000
batch_size = 20000  # collocation batch size per iteration
use_amp = True  # mixed precision
save_dir = "pinn_outputs"
os.makedirs(save_dir, exist_ok=True)

# neural network
class MLP(nn.Module):
    def __init__(self, layers, act=nn.Tanh()):
        super().__init__()
        seq = []
        for i in range(len(layers)-1):
            seq.append(nn.Linear(layers[i], layers[i+1]))
            if i < len(layers)-2:
                seq.append(act)
        self.net = nn.Sequential(*seq)
        self._init_weights()

    def _init_weights(self):
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.net(x)

model = MLP(layers).to(device)

# utility to sample points in domain
def sample_collocation(n):
    x = np.random.rand(n) * Lx
    y = (np.random.rand(n) - 0.5) * Ly
    z = (np.random.rand(n) - 0.5) * Lz
    t = np.random.rand(n) * T_final
    pts = np.stack([x,y,z,t], axis=1).astype(np.float32)
    return pts

def sample_boundary_inlet(n):
    x = np.random.rand(n) * (vessel_x*0.9)
    y = (np.random.rand(n) - 0.5) * Ly
    z = (np.random.rand(n) - 0.5) * Lz
    t = np.random.rand(n) * T_final
    pts = np.stack([x,y,z,t], axis=1).astype(np.float32)
    vals = np.ones((n,1), dtype=np.float32) * drug["C0"]
    return pts, vals

def sample_ic(n):
    x = np.random.rand(n) * Lx
    y = (np.random.rand(n) - 0.5) * Ly
    z = (np.random.rand(n) - 0.5) * Lz
    t = np.zeros(n, dtype=np.float32)
    pts = np.stack([x,y,z,t], axis=1).astype(np.float32)
    vals = np.zeros((n,1), dtype=np.float32)
    return pts, vals

# PDE residual computation (autograd)
def pde_residual(model, xytz):
    xytz = xytz.requires_grad_(True)
    C = model(xytz)  # [N,1]
    grads = torch.autograd.grad(C, xytz, grad_outputs=torch.ones_like(C), create_graph=True)[0]
    C_t = grads[:,3:4]
    C_x = grads[:,0:1]
    C_y = grads[:,1:2]
    C_z = grads[:,2:3]
    C_xx = torch.autograd.grad(C_x, xytz, grad_outputs=torch.ones_like(C_x), create_graph=True)[0][:,0:1]
    C_yy = torch.autograd.grad(C_y, xytz, grad_outputs=torch.ones_like(C_y), create_graph=True)[0][:,1:2]
    C_zz = torch.autograd.grad(C_z, xytz, grad_outputs=torch.ones_like(C_z), create_graph=True)[0][:,2:3]

    x = xytz[:,0:1]
    # spatially varying diffusion map: simple piecewise based on x coordinate
    D = torch.where(x < barrier_start_x, torch.tensor(D_water, device=device, dtype=x.dtype),
        torch.where(x < barrier_end_x, torch.tensor(D_barrier_internal, device=device, dtype=x.dtype),
                    torch.tensor(D_brain, device=device, dtype=x.dtype)))
    # simple advection only in vessel region (x < vessel_x)
    u = torch.where(x < vessel_x, torch.tensor(u_blood, device=device, dtype=x.dtype), torch.tensor(0.0, device=device, dtype=x.dtype))
    # efflux inside barrier region only (modeled volumetrically here)
    in_barrier = ((x >= barrier_start_x) & (x <= barrier_end_x)).float()
    Vmax = torch.tensor(drug["Vmax"], device=device, dtype=x.dtype)
    Km = torch.tensor(drug["Km"], device=device, dtype=x.dtype)
    C_free = C * drug["fu"]
    efflux = in_barrier * (Vmax * C_free / (Km + C_free + 1e-12))

    residual = C_t + u * C_x - D * (C_xx + C_yy + C_zz) + efflux
    return residual, C

# losses and training data
n_collocation = 120000
n_inlet = 20000
n_ic = 20000

collocation_pts = torch.tensor(sample_collocation(n_collocation), device=device)
inlet_pts_np, inlet_vals_np = sample_boundary_inlet(n_inlet)
inlet_pts = torch.tensor(inlet_pts_np, device=device)
inlet_vals = torch.tensor(inlet_vals_np, device=device)
ic_pts_np, ic_vals_np = sample_ic(n_ic)
ic_pts = torch.tensor(ic_pts_np, device=device)
ic_vals = torch.tensor(ic_vals_np, device=device)

optimizer = optim.Adam(model.parameters(), lr=lr)
scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
mse = nn.MSELoss()

# training loop (Adam)
pbar = trange(epochs_adam, desc="Train")
for ep in pbar:
    model.train()
    perm = torch.randperm(collocation_pts.shape[0], device=device)[:batch_size]
    batch_coll = collocation_pts[perm]
    optimizer.zero_grad()
    with torch.cuda.amp.autocast(enabled=use_amp):
        res, _ = pde_residual(model, batch_coll)
        loss_pde = mse(res, torch.zeros_like(res))
        # inlet BC
        pred_inlet = model(inlet_pts)
        loss_inlet = mse(pred_inlet, inlet_vals)
        # initial condition loss
        pred_ic = model(ic_pts)
        loss_ic = mse(pred_ic, ic_vals)
        loss = loss_pde + 10.0 * (loss_inlet + loss_ic)
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
    if ep % 200 == 0 or ep == epochs_adam-1:
        with torch.no_grad():
            total_loss = float(loss.detach().cpu().item())
            pbar.set_postfix({"loss": total_loss})

# optional L-BFGS fine tuning (commented by default)
use_lbfgs = False
if use_lbfgs:
    lbfgs_opt = optim.LBFGS(model.parameters(), lr=1.0, max_iter=500, tolerance_grad=1e-8, tolerance_change=1e-9)
    def closure():
        lbfgs_opt.zero_grad()
        res_all, _ = pde_residual(model, collocation_pts)
        loss_all = mse(res_all, torch.zeros_like(res_all))
        loss_bc = mse(model(inlet_pts), inlet_vals) + mse(model(ic_pts), ic_vals)
        loss_tot = loss_all + 10.0 * loss_bc
        loss_tot.backward()
        return loss_tot
    lbfgs_opt.step(closure)

torch.save(model.state_dict(), os.path.join(save_dir, "pinn_bbb_model.pt"))
print("Model saved.")

# Visualization: evaluate model on a grid and save mid-z time evolution frames
nx_vis, ny_vis, nt_vis = 120, 72, 40
xs = np.linspace(0, Lx, nx_vis)
ys = np.linspace(-Ly/2, Ly/2, ny_vis)
zs = [0.0]  # mid z plane at z=0
ts = np.linspace(0, T_final, nt_vis)
frames = []
frame_files = []
for ti, tval in enumerate(ts):
    X, Y = np.meshgrid(xs, ys, indexing="xy")
    x_flat = X.ravel().astype(np.float32)
    y_flat = Y.ravel().astype(np.float32)
    z_flat = np.zeros_like(x_flat, dtype=np.float32)
    t_flat = np.ones_like(x_flat, dtype=np.float32) * float(tval)
    pts = np.stack([x_flat, y_flat, z_flat, t_flat], axis=1)
    pts_t = torch.tensor(pts, device=device)
    model.eval()
    with torch.no_grad():
        pred = model(pts_t).cpu().numpy().reshape((ny_vis, nx_vis)).T
    vmin, vmax = 0.0, max(1e-6, pred.max())
    fig, ax = plt.subplots(1,1, figsize=(6,4))
    im = ax.imshow(pred.T, origin='lower', extent=[0,Lx,-Ly/2,Ly/2], vmin=vmin, vmax=vmax, cmap='viridis', aspect='auto')
    ax.set_title(f"{drug['name']} t={tval:.4f}s")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046)
    cbar.set_label("C")
    fname = os.path.join(save_dir, f"frame_{ti:04d}.png")
    fig.savefig(fname, dpi=120, bbox_inches='tight')
    plt.close(fig)
    frame_files.append(fname)
    print(f"Saved frame {ti+1}/{len(ts)}")

gif_path = os.path.join(save_dir, f"{drug['name']}_pinn_midz.gif")
images = [imageio.imread(f) for f in frame_files]
imageio.mimsave(gif_path, images, fps=12)
print("Saved GIF:", gif_path)
