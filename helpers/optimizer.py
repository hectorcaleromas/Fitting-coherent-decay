"""Nelder-Mead refinement of T1, T2 and detuning."""
import numpy as np
from scipy.optimize import minimize
from .comparison import rotate_first_point
try:
    from tqdm.auto import tqdm
except ImportError:
    tqdm = None

def optimize_parameters(simulator, exp_data, decay_times, initial_T1, initial_T2,
                         initial_detuning, initial_alpha, initial_angle,
                         residue_function, fit_T1=False, fixed_T1=None, maxiter=40,
                         first_angle=None):
    fixed_T1 = initial_T1 if fixed_T1 is None else fixed_T1
    history, both = [], isinstance(exp_data, tuple)
    def objective(params):
        if fit_T1: T1, T2, detuning = params
        else: T2, detuning = params; T1 = fixed_T1
        if T1 <= 1000 or T2 <= 1000: return 1e6
        try:
            sim = simulator.simulate(decay_times, T1, T2, alpha=initial_alpha,
                detuning_MHz=detuning, angle_offset=initial_angle,
                return_real=both, return_imag=both)
            if first_angle is not None:
                sim = rotate_first_point(sim, first_angle, initial_angle)
            cost = residue_function(exp_data, sim, rotate_images=False)
        except Exception: return 1e6
        history.append((T1, T2, detuning, cost)); return cost
    guess = np.array([initial_T1, initial_T2, initial_detuning] if fit_T1 else [initial_T2, initial_detuning], dtype=float)
    progress = tqdm(total=maxiter, desc="Optimizing T1/T2") if tqdm else None
    evaluations = [0]
    def tracked_objective(params):
        value = objective(params)
        evaluations[0] += 1
        if progress and evaluations[0] <= maxiter: progress.update(1)
        return value
    result = minimize(tracked_objective, guess, method="Nelder-Mead",
                      options={"maxiter": maxiter, "disp": True, "xatol": 100, "fatol": 1e-5})
    if progress: progress.close()
    if fit_T1: best_T1, best_T2, best_detuning = result.x
    else: best_T2, best_detuning = result.x; best_T1 = fixed_T1
    sim = simulator.simulate(decay_times, best_T1, best_T2, alpha=initial_alpha,
        detuning_MHz=best_detuning, angle_offset=initial_angle,
        return_real=both, return_imag=both)
    if first_angle is not None:
        sim = rotate_first_point(sim, first_angle, initial_angle)
    return {"T1": best_T1, "T2": best_T2, "detuning": best_detuning,
            "cost": result.fun, "simulation": sim,
            "history": np.asarray(history), "result": result}
