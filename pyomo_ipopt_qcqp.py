import pyomo.environ as pyo
import numpy as np
import scipy.sparse as sp
import time
import torch
import argparse
def create_and_solve_acopf_ipopt(problem_dict, Pd_instance, Qd_instance, slack_imag_idx):
    """
    Formulates and solves the ACOPF QCQP problem for a single load instance using Pyomo and Ipopt.
    
    Inputs:
        problem_dict: The dictionary of system matrices (converted to numpy/scipy from torch)
        Pd_instance: 1D numpy array of active power demand for a single snapshot [nbus]
        Qd_instance: 1D numpy array of reactive power demand for a single snapshot [nbus]
        slack_imag_idx: Integer index of the slack bus imaginary voltage component
    """
    # 1. Initialize Model and Dimensions
    m = pyo.ConcreteModel()
    
    nbus = problem_dict["nbus"]
    ngen = problem_dict["ngen"]
    
    # 2. Extract and format limits (assumes scalar or 1D arrays)
    pmax = problem_dict["pmax"]
    pmin = problem_dict["pmin"]
    qmax = problem_dict["qmax"]
    qmin = problem_dict["qmin"]
    Vmax2 = problem_dict["Vmax"]**2
    Vmin2 = problem_dict["Vmin"]**2
    smax2 = problem_dict["smax"]**2
    
    # 3. Define Variables with Bounds
    m.GEN = pyo.RangeSet(0, ngen - 1)
    m.pg = pyo.Var(m.GEN, initialize=lambda m, i: float((pmin[i] + pmax[i]) / 2.0), bounds=lambda m, i: (float(pmin[i]), float(pmax[i])))
    m.qg = pyo.Var(m.GEN, initialize=lambda m, i: float((qmin[i] + qmax[i]) / 2.0), bounds=lambda m, i: (float(qmin[i]), float(qmax[i])))
    
    m.BUS2 = pyo.RangeSet(0, 2 * nbus - 1)
    max_v_rect = float(np.max(problem_dict["Vmax"])) 
    
    def v_init_rule(m, i):
        return 1.0 if i < nbus else 0.0
    m.v = pyo.Var(m.BUS2, initialize=v_init_rule, bounds=(-max_v_rect, max_v_rect))
    
    m.v[slack_imag_idx].fix(0.0)

    # Bulletproof quadratic form: strictly casts numpy data to Python floats
    def quad_form(matrix):
        M_coo = sp.coo_matrix(matrix)
        expr = 0.0
        for k in range(M_coo.nnz):
            expr += float(M_coo.data[k]) * m.v[int(M_coo.row[k])] * m.v[int(M_coo.col[k])]
        return expr

    # 4. Objective Function 
    c2 = problem_dict["c2"]
    c1 = problem_dict["c1"]
    c0 = problem_dict["c0"]
    
    def cost_rule(m):
        return sum(float(c2[i]) * m.pg[i]**2 + float(c1[i]) * m.pg[i] + float(c0[i]) for i in m.GEN)
    m.cost = pyo.Objective(rule=cost_rule, sense=pyo.minimize)

    # 5. Constraints
    m.Constraints = pyo.ConstraintList()
    C_g = problem_dict["C_g"] 
    
    # --- A. Nodal Power Balance Constraints ---
    for bus_i in range(nbus):
        # Active Power Balance
        gen_p = 0.0
        for g in m.GEN:
            if C_g[bus_i, g] != 0:
                gen_p += float(C_g[bus_i, g]) * m.pg[g]
        v_Mp_v = quad_form(problem_dict["M_p"][bus_i])
        m.Constraints.add(gen_p - float(Pd_instance[bus_i]) == v_Mp_v)
        
        # Reactive Power Balance
        gen_q = 0.0
        for g in m.GEN:
            if C_g[bus_i, g] != 0:
                gen_q += float(C_g[bus_i, g]) * m.qg[g]
        v_Mq_v = quad_form(problem_dict["M_q"][bus_i])
        m.Constraints.add(gen_q - float(Qd_instance[bus_i]) == v_Mq_v)
        
        # Voltage Magnitude Limits
        v_Mv_v = quad_form(problem_dict["M_v"][bus_i])
        m.Constraints.add(pyo.inequality(float(Vmin2[bus_i]), v_Mv_v, float(Vmax2[bus_i])))

    # --- B. Branch Constraints ---
    nbranch = problem_dict["M_pf"].shape[0]
    for br in range(nbranch):
        # 1. Thermal Limits (Apparent Power)
        smax_val = float(smax2[br])
        
        p_from = quad_form(problem_dict["M_pf"][br])
        q_from = quad_form(problem_dict["M_qf"][br])
        m.Constraints.add(p_from**2 + q_from**2 <= smax_val)
        
        p_to = quad_form(problem_dict["M_pt"][br])
        q_to = quad_form(problem_dict["M_qt"][br])
        m.Constraints.add(p_to**2 + q_to**2 <= smax_val)

        # 2. Angle Difference Stability (Safely Handled)
        angmax_rad = float(problem_dict["angmax"][br])
        angmin_rad = float(problem_dict["angmin"][br])
        
        if angmax_rad < np.pi/2 and angmin_rad > -np.pi/2:
            v_Mc_v = quad_form(problem_dict["M_c"][br])
            v_Ms_v = quad_form(problem_dict["M_s"][br])
            
            m.Constraints.add(float(np.tan(angmin_rad)) * v_Mc_v <= v_Ms_v)
            m.Constraints.add(v_Ms_v <= float(np.tan(angmax_rad)) * v_Mc_v)
    
    # 6. Solve the Model
    solver = pyo.SolverFactory('ipopt')
    # Optional: Pass specific Ipopt tolerances (useful for non-convex QCQP)
    solver.options['tol'] = 1e-6
    solver.options['max_iter'] = 3000
    
    start_time = time.time()
    results = solver.solve(m, tee=True) # tee=True streams Ipopt terminal output
    solve_time = time.time() - start_time
    
    # 7. Extract Results
    status = results.solver.termination_condition
    obj_val = pyo.value(m.cost)
    
    pg_res = np.array([pyo.value(m.pg[i]) for i in m.GEN])
    qg_res = np.array([pyo.value(m.qg[i]) for i in m.GEN])
    v_res = np.array([pyo.value(m.v[i]) for i in m.BUS2])
    
    return status, obj_val, solve_time, v_res, pg_res, qg_res

if __name__ == "__main__":
    # === Setup Command Line Arguments ===
    parser = argparse.ArgumentParser(description="Evaluate ACOPF baseline using Pyomo and IPOPT.")
    parser.add_argument('--case_name', type=str, required=True, help="Name of the grid case (e.g., pglib_opf_case300_ieee)")
    parser.add_argument('--eval_limit', type=int, default=1, help="Number of instances to solve (default: 1)")
    parser.add_argument('--total_samples', type=int, default=10000, help="Total samples in the loaded dataset (default: 10000)")
    args = parser.parse_args()

    # 1. Load the Dataset
    case_name = args.case_name
    total_samples = args.total_samples
    dataset_path = f'./dataset/{case_name}_{total_samples}.pt'
    
    print(f"Loading dataset from {dataset_path}...")
    problem_pt = torch.load(dataset_path, map_location='cpu')

    # 2. Strip PyTorch bindings to create a pure NumPy dictionary
    problem_np = {}
    for key, value in problem_pt.items():
        if isinstance(value, torch.Tensor):
            problem_np[key] = value.numpy()
        else:
            problem_np[key] = value

    # 3. Extract the Validation/Test Slices
    # Matching the 80/10/10 split logic from your PINN script
    actual_total_samples = problem_np["Pd_all"].shape[0] 
    train_size = int(0.8 * actual_total_samples)
    val_size = int(0.1 * actual_total_samples)

    # We will use the validation split as the evaluation baseline
    test_Pd = problem_np["Pd_all"][train_size + val_size:]
    test_Qd = problem_np["Qd_all"][train_size + val_size:]

    # Dynamically find the slack bus imaginary index
    slack_imag_idx = int(np.where(problem_np["a_ref"] == 1)[0][0])

    # 4. Setup the Loop Iteration

    # Data structures to capture benchmark metrics and physical variables
    metrics = {
        "status": [],
        "obj_val": [],
        "solve_time": []
    }
    
    solutions = {
        "v_optimal": [],
        "pg_optimal": [],
        "qg_optimal": []
    }

    # 5. The Execution Loop
    total_start_time = time.time()
    
    # Safely assign eval_limit (preventing out-of-bounds errors)
    num_test_samples = test_Pd.shape[0] 
    eval_limit = min(args.eval_limit, num_test_samples)
    
    print(f"\nStarting Ipopt baseline evaluation over {eval_limit} samples...")
    for i in range(eval_limit):
        Pd_instance = problem_np["Pd"] #test_Pd[i]
        Qd_instance = problem_np["Qd"] #test_Qd[i]
        
        print(f"[{i+1}/{eval_limit}] Solving Instance...")
        
        # Dispatch the single instance to Pyomo/Ipopt
        status, obj_val, solve_time, v_res, pg_res, qg_res = create_and_solve_acopf_ipopt(
            problem_dict=problem_np, 
            Pd_instance=Pd_instance, 
            Qd_instance=Qd_instance, 
            slack_imag_idx=slack_imag_idx
        )
        
        # Log the optimizer metrics
        metrics["status"].append(str(status))
        metrics["solve_time"].append(solve_time)
        metrics["obj_val"].append(obj_val if obj_val is not None else np.nan)
        
        # Log the mathematical state variables for later MSE comparison with the PINN
        solutions["v_optimal"].append(v_res)
        solutions["pg_optimal"].append(pg_res)
        solutions["qg_optimal"].append(qg_res)
        
    total_time = time.time() - total_start_time

    # 6. Aggregate Baseline Metrics
    # Pyomo typically returns 'ok' or 'optimal' for successful solves
    successful_solves = sum(1 for s in metrics["status"] if "ok" in s.lower() or "optimal" in s.lower())
    avg_solve_time = np.mean(metrics["solve_time"])
    
    # Calculate average cost only for instances that successfully converged
    valid_costs = [obj for stat, obj in zip(metrics["status"], metrics["obj_val"]) 
                   if ("ok" in stat.lower() or "optimal" in stat.lower()) and not np.isnan(obj)]
    avg_cost = np.mean(valid_costs) if valid_costs else float('inf')

    print("\n" + "="*50)
    print("IPOPT BASELINE EVALUATION COMPLETE")
    print(f"Total Computation Time: {total_time:.2f}s")
    print(f"Convergence Success Rate: {successful_solves}/{eval_limit} ({(successful_solves/eval_limit)*100:.2f}%)")
    print(f"Average Solve Time: {avg_solve_time:.4f}s per instance")
    print(f"Average Optimal Cost: {avg_cost:.2f}")
    print("="*50)
    
    # Optional: Save the ground-truth solutions to disk to plot against PINN outputs
    np.savez(f"result/ipopt_baseline_{case_name}_{eval_limit}_instances.npz", **solutions, **metrics)