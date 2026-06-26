import numpy as np
import pandas as pd
from sys import stderr
from numpy import zeros, arange, isscalar,diag, dot,eye, ix_, ones, r_, pi, flatnonzero as find
from scipy.sparse import csr_matrix
from numpy.linalg import solve
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# === Initialization ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.float32

case_name = 'pglib_opf_case3_lmbd'
case_path = f'../excel_outputs/{case_name}.xlsx'
case = pd.read_excel(case_path, sheet_name=['baseMVA','bus','gen','gencost','branch'])

bus_to_idx = {bus: i+1 for i, bus in enumerate(case['bus'].bus_i.values)}
bus_idx = [bus_to_idx[bus] for bus in case['bus'].bus_i.values]
case['bus'].bus_i = case['bus'].bus_i.replace(bus_to_idx) # rename the bus for making PTDF
case['gen'].bus_i = case['gen'].bus_i.replace(bus_to_idx)
case['gencost'].bus_i = case['gencost'].bus_i.replace(bus_to_idx)
case['branch'].bus_i = case['branch'].bus_i.replace(bus_to_idx)
case['branch'].bus_j = case['branch'].bus_j.replace(bus_to_idx)
nbus = case['bus'].shape[0]
ngen = case['gen'].shape[0]
nbranch = case['branch'].shape[0]

# per unit p.u. conversion for cost coefficients
baseMVA = case['baseMVA'].values[0][0]
c2 = case['gencost'].c2.values * baseMVA**2
c1 = case['gencost'].c1.values * baseMVA
c0 = case['gencost'].c0.values

# calculate susceptance, conductance, admittance-square y_sq
# $Z = r + ix$ $Y = g + ib$ $Y = \frac{1}{Z} = \frac{r}{r^2 + x^2} - i\frac{x}{r^2 + x^2}$
# 1. Physics: Admittance Y = g + i*b
r = case['branch']['r'].values
x = case['branch']['x'].values
Z_sq = r**2 + x**2
g = r / Z_sq
b = -x / Z_sq
y_sq = 1 / Z_sq

# 2. Extract Line Charging, Taps, and Phase Shifts
bc = case['branch']['b'].values # MATPOWER branch 'b' is total line charging susceptance
tau = np.where(case['branch']['ratio'].values == 0, 1.0, case['branch']['ratio'].values)
theta_shift = np.radians(case['branch']['angle'].values)

# 3. Data Extraction
Gs = case['bus']['Gs'].values / baseMVA
Bs = case['bus']['Bs'].values / baseMVA
Pd = case['bus'].Pd.values / baseMVA
Qd = case['bus'].Qd.values / baseMVA

# State vector dimension D = 2 * |B|
D = 2 * nbus

# Initialize lists to store matrices for all branches
M_pf = []; M_qf = []; M_pt = []; M_qt = []

# Pre-calculate derived branch elements
g11 = g / (tau**2)
g12 = g * np.cos(theta_shift) / tau
g21 = g * np.sin(theta_shift) / tau
g22 = g

b11 = (b + bc/2) / (tau**2)
b12 = b * np.cos(theta_shift) / tau
b21 = b * np.sin(theta_shift) / tau
b22 = b + bc/2

for l in range(nbranch):
    # Python uses 0-based indexing; your dictionary offset to 1-based, so we subtract 1
    i = int(case['branch']['bus_i'].values[l]) - 1
    j = int(case['branch']['bus_j'].values[l]) - 1
    
    # Real and Imaginary indices
    i_B = i + nbus
    j_B = j + nbus

    # --- FROM END ---
    A_pf = np.zeros((D, D))
    A_pf[i, i] = g11[l]
    A_pf[i_B, i_B] = g11[l]
    A_pf[i, j] = -(g12[l] - b21[l])
    A_pf[i_B, j_B] = -(g12[l] - b21[l])
    A_pf[i, j_B] = (g21[l] + b12[l])
    A_pf[i_B, j] = -(g21[l] + b12[l])
    M_pf.append(0.5 * (A_pf + A_pf.T))

    A_qf = np.zeros((D, D))
    A_qf[i, i] = -b11[l]
    A_qf[i_B, i_B] = -b11[l]
    A_qf[i, j] = (b12[l] + g21[l])
    A_qf[i_B, j_B] = (b12[l] + g21[l])
    A_qf[i, j_B] = -(b21[l] - g12[l])
    A_qf[i_B, j] = (b21[l] - g12[l])
    M_qf.append(0.5 * (A_qf + A_qf.T))

    # --- TO END ---
    A_pt = np.zeros((D, D))
    A_pt[j, j] = g22[l]
    A_pt[j_B, j_B] = g22[l]
    A_pt[j, i] = -(g12[l] + b21[l])
    A_pt[j_B, i_B] = -(g12[l] + b21[l])
    A_pt[j, i_B] = -(g21[l] - b12[l])
    A_pt[j_B, i] = (g21[l] - b12[l])
    M_pt.append(0.5 * (A_pt + A_pt.T))

    A_qt = np.zeros((D, D))
    A_qt[j, j] = -b22[l]
    A_qt[j_B, j_B] = -b22[l]
    A_qt[j, i] = (b12[l] - g21[l])
    A_qt[j_B, i_B] = (b12[l] - g21[l])
    A_qt[j, i_B] = (b21[l] + g12[l])
    A_qt[j_B, i] = -(b21[l] + g12[l])
    M_qt.append(0.5 * (A_qt + A_qt.T))
    
# ------------------------------------------------------------
# 2. Nodal Power Injection Matrices (Guaranteed Consistency)
# ------------------------------------------------------------
# Initialize empty matrices
M_p = [np.zeros((D, D)) for _ in range(nbus)]
M_q = [np.zeros((D, D)) for _ in range(nbus)]

# Add Nodal Shunts
for i in range(nbus):
    # Active power shunt (V^2 * Gs)
    M_p[i][i, i] = Gs[i]
    M_p[i][i+nbus, i+nbus] = Gs[i]
    # Reactive power shunt (V^2 * -Bs)
    M_q[i][i, i] = -Bs[i]
    M_q[i][i+nbus, i+nbus] = -Bs[i]

# Add Branch Flows
# Nodal injection must exactly equal the sum of outgoing/incoming flows
for l in range(nbranch):
    from_bus = int(case['branch']['bus_i'].values[l]) - 1
    to_bus = int(case['branch']['bus_j'].values[l]) - 1
    
    # Add flow leaving the 'from' bus
    M_p[from_bus] += M_pf[l]
    M_q[from_bus] += M_qf[l]
    
    # Add flow leaving the 'to' bus
    M_p[to_bus] += M_pt[l]
    M_q[to_bus] += M_qt[l]

M_c = []; M_s = []

for l in range(nbranch):
    i = int(case['branch']['bus_i'].values[l]) - 1
    j = int(case['branch']['bus_j'].values[l]) - 1
    i_B = i + nbus
    j_B = j + nbus

    # Angle Cosine Extraction (Eq 29 & 30)
    A_c = np.zeros((D, D))
    A_c[i, j] = 1
    A_c[i_B, j_B] = 1
    M_c.append(0.5 * (A_c + A_c.T))

    # Angle Sine Extraction (Eq 31 & 32)
    A_s = np.zeros((D, D))
    A_s[i_B, j] = 1
    A_s[i, j_B] = -1
    M_s.append(0.5 * (A_s + A_s.T))

M_V = []
for i in range(nbus):
    # Voltage Magnitude Extraction (Eq 33 & 34)
    A_V = np.zeros((D, D))
    A_V[i, i] = 1
    A_V[i + nbus, i + nbus] = 1
    M_V.append(A_V) # Already symmetric

# Identify the slack bus (MATPOWER sets bus type to 3 for slack)
slack_bus_idx = case['bus'][case['bus']['type'] == 3].index[0]

a_ref = np.zeros(D)
# Force the imaginary voltage component of the slack bus to 0
a_ref[slack_bus_idx + nbus] = 1

# ------------------------------------------------------------
# 1) Dimensions
# ------------------------------------------------------------
# These match the sizes from your MATPOWER data
nbus = nbus
ngen = ngen
nbranch = nbranch
d = D

# ------------------------------------------------------------
# 2) Stack quadratic matrices (For ALL buses and branches)
# ------------------------------------------------------------
# Nodal power and voltage matrices [nbus, d, d]
M_p_stack = torch.stack([torch.as_tensor(M_p[i], dtype=dtype, device=device) for i in range(nbus)])
M_q_stack = torch.stack([torch.as_tensor(M_q[i], dtype=dtype, device=device) for i in range(nbus)])
M_v_stack = torch.stack([torch.as_tensor(M_V[i], dtype=dtype, device=device) for i in range(nbus)])

# Branch flow matrices [nbranch, d, d]
M_pf_stack = torch.stack([torch.as_tensor(M_pf[l], dtype=dtype, device=device) for l in range(nbranch)])
M_qf_stack = torch.stack([torch.as_tensor(M_qf[l], dtype=dtype, device=device) for l in range(nbranch)])
M_pt_stack = torch.stack([torch.as_tensor(M_pt[l], dtype=dtype, device=device) for l in range(nbranch)])
M_qt_stack = torch.stack([torch.as_tensor(M_qt[l], dtype=dtype, device=device) for l in range(nbranch)])

# Angle difference matrices [nbranch, d, d]
M_c_stack = torch.stack([torch.as_tensor(M_c[l], dtype=dtype, device=device) for l in range(nbranch)])
M_s_stack = torch.stack([torch.as_tensor(M_s[l], dtype=dtype, device=device) for l in range(nbranch)])

# ------------------------------------------------------------
# 3) Symmetrize matrices (Required for stable Autograd)
# ------------------------------------------------------------
M_p_stack = 0.5 * (M_p_stack + M_p_stack.transpose(-1, -2))
M_q_stack = 0.5 * (M_q_stack + M_q_stack.transpose(-1, -2))
M_v_stack = 0.5 * (M_v_stack + M_v_stack.transpose(-1, -2))

M_pf_stack = 0.5 * (M_pf_stack + M_pf_stack.transpose(-1, -2))
M_qf_stack = 0.5 * (M_qf_stack + M_qf_stack.transpose(-1, -2))
M_pt_stack = 0.5 * (M_pt_stack + M_pt_stack.transpose(-1, -2))
M_qt_stack = 0.5 * (M_qt_stack + M_qt_stack.transpose(-1, -2))
M_c_stack = 0.5 * (M_c_stack + M_c_stack.transpose(-1, -2))
M_s_stack = 0.5 * (M_s_stack + M_s_stack.transpose(-1, -2))

# ------------------------------------------------------------
# 4) The C_g Matrix (Mapping Generators to Buses)
# ------------------------------------------------------------
# Shape: [nbus, ngen]
C_g = torch.zeros((nbus, ngen), dtype=dtype, device=device)
for gen_idx, bus_i in enumerate(case['gen']['bus_i'].values):
    bus_idx = int(bus_i) - 1 # convert to 0-based index
    C_g[bus_idx, gen_idx] = 1.0

# ------------------------------------------------------------
# 5) Vectors: Demands, Limits, and Reference
# ------------------------------------------------------------
Pd_bus = np.asarray(case['bus'].Pd.values, dtype=np.float32) / baseMVA
Qd_bus = np.asarray(case['bus'].Qd.values, dtype=np.float32) / baseMVA

pmax = np.asarray(case['gen'].Pmax.values, dtype=np.float32) / baseMVA
pmin = np.asarray(case['gen'].Pmin.values, dtype=np.float32) / baseMVA
qmax = np.asarray(case['gen'].Qmax.values, dtype=np.float32) / baseMVA
qmin = np.asarray(case['gen'].Qmin.values, dtype=np.float32) / baseMVA

# Apparent power branch limits (s_max)
smax = np.asarray(case['branch'].rateA.values, dtype=np.float32) / baseMVA
smax[smax == 0] = 9999.0  # Handle unconstrained lines gracefully

# Branch angle limits (converted to radians)
angmax = np.radians(np.asarray(case['branch'].angmax.values, dtype=np.float32))
angmin = np.radians(np.asarray(case['branch'].angmin.values, dtype=np.float32))

Vmax_arr = np.asarray(case['bus'].Vmax.values, dtype=np.float32)
Vmin_arr = np.asarray(case['bus'].Vmin.values, dtype=np.float32)

# ------------------------------------------------------------
# 6) Final problem dictionary for the PINN loss
# ------------------------------------------------------------
problem = {
    # Quadratic Matrices
    "M_p": M_p_stack,
    "M_q": M_q_stack,
    "M_v": M_v_stack,
    "M_pf": M_pf_stack,
    "M_qf": M_qf_stack,
    "M_pt": M_pt_stack,
    "M_qt": M_qt_stack,
    "M_c": M_c_stack,
    "M_s": M_s_stack,

    # Incidence Matrix
    "C_g": C_g,

    # Base Vectors
    "Pd": torch.as_tensor(Pd_bus, dtype=dtype, device=device),
    "Qd": torch.as_tensor(Qd_bus, dtype=dtype, device=device),
    
    "pmax": torch.as_tensor(pmax, dtype=dtype, device=device),
    "pmin": torch.as_tensor(pmin, dtype=dtype, device=device),
    "qmax": torch.as_tensor(qmax, dtype=dtype, device=device),
    "qmin": torch.as_tensor(qmin, dtype=dtype, device=device),
    
    "smax": torch.as_tensor(smax, dtype=dtype, device=device),
    "angmax": torch.as_tensor(angmax, dtype=dtype, device=device),
    "angmin": torch.as_tensor(angmin, dtype=dtype, device=device),
    
    "Vmax": torch.as_tensor(Vmax_arr, dtype=dtype, device=device),
    "Vmin": torch.as_tensor(Vmin_arr, dtype=dtype, device=device),
    
    # Add the cost coefficients
    "c2": torch.tensor(c2, dtype=dtype, device=device),
    "c1": torch.tensor(c1, dtype=dtype, device=device),
    "c0": torch.tensor(c0, dtype=dtype, device=device),
        
    # Anchor vector (Ensure a_ref from our earlier discussion is defined)
    "a_ref": torch.as_tensor(a_ref, dtype=dtype, device=device),

    # Metadata
    "nbus": nbus,
    "ngen": ngen,
    "nbranch": nbranch
}

print("Constructed PINN problem data for QCQP:")
print(f"  nbus    = {nbus}")
print(f"  ngen    = {ngen}")
print(f"  nbranch = {nbranch}")
print(f"  M_pf, M_qf shape = {tuple(problem['M_pf'].shape)}, {tuple(problem['M_qf'].shape)}")
print(f"  M_pt, M_qt shape = {tuple(problem['M_pt'].shape)}, {tuple(problem['M_qt'].shape)}")
print(f"  C_g shape  = {tuple(problem['C_g'].shape)}")
print(f"  M_p, M_q shape  = {tuple(problem['M_p'].shape)}, {tuple(problem['M_q'].shape)}")
print(f"  M_s, M_c shape  = {tuple(problem['M_s'].shape)}, {tuple(problem['M_c'].shape)}")
print(f"  M_V shape  = {tuple(problem['M_v'].shape)}")

total_samples=10000
 
def gaussian_batch(base_tensor, batch_size, variation_std=0.05, clamp_min=None):
    """
    Create a batch of tensors with Gaussian random variations.
    """
    base_batch = base_tensor.unsqueeze(0).repeat(batch_size, 1)
    
    # Use torch.abs() to ensure variation is calculated correctly on negative base loads
    noise = variation_std * torch.abs(base_tensor.unsqueeze(0)) * torch.randn_like(base_batch)
    batch = base_batch + noise
    
    if clamp_min is not None:
        batch = torch.clamp(batch, min=clamp_min)
    return batch

def generate_and_save_dataset(problem, total_samples=10000, save_path="acopf_problem_with_data.pt"):
    """
    Generates static samples and saves them INSIDE the problem dictionary.
    """
    print(f"Generating {total_samples} static samples...")
    
    # Generate the full batch of demands (clamping Pd to 0, leaving Qd unclamped)
    Pd_all = gaussian_batch(problem["Pd"], batch_size=total_samples, variation_std=0.05, clamp_min=0.0)
    Qd_all = gaussian_batch(problem["Qd"], batch_size=total_samples, variation_std=0.05, clamp_min=None)
    
    # Attach the full generated datasets directly to the problem dictionary
    problem["Pd_all"] = Pd_all
    problem["Qd_all"] = Qd_all
    
    # Save the entire problem dictionary (physics + data) to disk
    torch.save(problem, save_path)
    print(f"Problem dictionary with {total_samples} samples successfully saved to {save_path}")

# --- Execute Generation ---
generate_and_save_dataset(problem, total_samples=total_samples, save_path=f"./dataset/{case_name}_{total_samples}.pt")    