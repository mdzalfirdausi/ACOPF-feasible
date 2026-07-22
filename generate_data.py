import numpy as np
import pandas as pd
import argparse
import torch

# === Setup Command Line Arguments ===
parser = argparse.ArgumentParser(description="Generate dataset for ACOPF PINN.")
parser.add_argument('--case_name', type=str, required=True, help="Name of the grid case (e.g., pglib_opf_case3_lmbd)")
parser.add_argument('--samples', type=int, default=10000, help="Number of samples to generate (default: 10000)")
args = parser.parse_args()

# === Initialization ===
device = torch.device("cpu")
dtype = torch.float32 
torch.set_default_dtype(torch.float32)
torch.manual_seed(42)
np.random.seed(42)
# Use the parsed argument instead of a hardcoded string
case_name = args.case_name 
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

# --- ADD THIS: Extract nominal voltage magnitude and angle (in radians) ---
Vm = case['bus']['Vm'].values
Va = np.radians(case['bus']['Va'].values)
# --- ADD THIS: Extract 0-based From-Bus and To-Bus indexing vectors ---
fbus = case['branch']['bus_i'].values.astype(np.int64) - 1
tbus = case['branch']['bus_j'].values.astype(np.int64) - 1
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

# Identify the slack bus (MATPOWER sets bus type to 3 for slack)
slack_bus_idx = case['bus'][case['bus']['type'] == 3].index[0]

a_ref = np.zeros(D)
# Force the imaginary voltage component of the slack bus to 0
a_ref[slack_bus_idx + nbus] = 1


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
# --- ADD THIS: Format Vm and Va arrays ---
Vm_arr = np.asarray(Vm, dtype=np.float32)
Va_arr = np.asarray(Va, dtype=np.float32)
# ------------------------------------------------------------
# 6) Final problem dictionary for Graph / Branch-Incidence PINN
# ------------------------------------------------------------
problem = {
    # Nodal Shunts & Anchor
    "Gs": torch.as_tensor(Gs, dtype=dtype, device=device),
    "Bs": torch.as_tensor(Bs, dtype=dtype, device=device),
    "a_ref": torch.as_tensor(a_ref, dtype=dtype, device=device),

    # Graph Topology / Line Indexing (Long tensors for indexing)
    "fbus": torch.as_tensor(fbus, dtype=torch.long, device=device),
    "tbus": torch.as_tensor(tbus, dtype=torch.long, device=device),

    # 1D Branch Conductance & Susceptance Parameters
    "g11": torch.as_tensor(g11, dtype=dtype, device=device),
    "g12": torch.as_tensor(g12, dtype=dtype, device=device),
    "g21": torch.as_tensor(g21, dtype=dtype, device=device),
    "g22": torch.as_tensor(g22, dtype=dtype, device=device),
    "b11": torch.as_tensor(b11, dtype=dtype, device=device),
    "b12": torch.as_tensor(b12, dtype=dtype, device=device),
    "b21": torch.as_tensor(b21, dtype=dtype, device=device),
    "b22": torch.as_tensor(b22, dtype=dtype, device=device),

    # Incidence Matrix (Keep for Nodal Generation mapping)
    "C_g": C_g,

    # Nodal Demands
    "Pd": torch.as_tensor(Pd_bus, dtype=dtype, device=device),
    "Qd": torch.as_tensor(Qd_bus, dtype=dtype, device=device),
    
    # Generator Limits
    "pmax": torch.as_tensor(pmax, dtype=dtype, device=device),
    "pmin": torch.as_tensor(pmin, dtype=dtype, device=device),
    "qmax": torch.as_tensor(qmax, dtype=dtype, device=device),
    "qmin": torch.as_tensor(qmin, dtype=dtype, device=device),
    
    # Line Thermal & Angle Limits
    "smax": torch.as_tensor(smax, dtype=dtype, device=device),
    "angmax": torch.as_tensor(angmax, dtype=dtype, device=device),
    "angmin": torch.as_tensor(angmin, dtype=dtype, device=device),
    
    # Voltage Limits & Nominal Setpoints
    "Vmax": torch.as_tensor(Vmax_arr, dtype=dtype, device=device),
    "Vmin": torch.as_tensor(Vmin_arr, dtype=dtype, device=device),
    "Vm": torch.as_tensor(Vm_arr, dtype=dtype, device=device),
    "Va": torch.as_tensor(Va_arr, dtype=dtype, device=device),
    
    # Cost Coefficients
    "c2": torch.tensor(c2, dtype=dtype, device=device),
    "c1": torch.tensor(c1, dtype=dtype, device=device),
    "c0": torch.tensor(c0, dtype=dtype, device=device),

    # Metadata
    "nbus": nbus,
    "ngen": ngen,
    "nbranch": nbranch
}

print("Constructed PINN problem data for QCQP:")
print(f"  nbus    = {nbus}")
print(f"  ngen    = {ngen}")
print(f"  nbranch = {nbranch}")

total_samples = args.samples
 
def gaussian_batch(base_tensor, batch_size, variation_std=0.05):
    """
    Create a batch of tensors with Gaussian random variations.
    """
    base_batch = base_tensor.unsqueeze(0).repeat(batch_size, 1)
    
    # Use torch.abs() to ensure variation is calculated correctly on negative base loads
    noise = variation_std * torch.abs(base_tensor.unsqueeze(0)) * torch.randn_like(base_batch)
    batch = base_batch + noise
    # batch = base_batch
    
    return batch

def generate_and_save_dataset(problem, total_samples=10000, save_path="acopf_problem_with_data.pt"):
    print(f"Generating {total_samples} static samples...")
    
    # Generate the full batch of demands (clamping Pd to 0, leaving Qd unclamped)
    Pd_all = gaussian_batch(problem["Pd"], batch_size=total_samples, variation_std=0.05)
    Qd_all = gaussian_batch(problem["Qd"], batch_size=total_samples, variation_std=0.05)
    
    # Attach the full generated datasets directly to the problem dictionary
    problem["Pd_all"] = Pd_all
    problem["Qd_all"] = Qd_all
    
    # Save the entire problem dictionary (physics + data) to disk
    torch.save(problem, save_path)
    print(f"Problem dictionary with {total_samples} samples successfully saved to {save_path}")

# --- Execute Generation ---
generate_and_save_dataset(problem, total_samples=total_samples, save_path=f"./dataset/{case_name}_{total_samples}.pt")