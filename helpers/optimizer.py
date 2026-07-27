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
                         first_angle=None, component="real", fit_angle=False):
    fixed_T1 = initial_T1 if fixed_T1 is None else fixed_T1
    history, both = [], isinstance(exp_data, tuple)
    def objective(params):
        cursor = 0
        if fit_T1:
            T1, T2, detuning = params[:3]
            cursor = 3
        else:
            T2, detuning = params[:2]
            T1, cursor = fixed_T1, 2
        angle = params[cursor] if fit_angle else initial_angle
        if T1 <= 1000 or T2 <= 1000: return 1e6
        try:
            sim = simulator.simulate(decay_times, T1, T2, alpha=initial_alpha,
                detuning_MHz=detuning, angle_offset=angle,
                return_real=(both or component == "real"),
                return_imag=(both or component == "imag"))
            if first_angle is not None:
                sim = rotate_first_point(sim, first_angle, angle)
            cost = residue_function(exp_data, sim, rotate_images=False)
        except Exception: return 1e6
        history.append((T1, T2, detuning, cost)); return cost
    guess_values = ([initial_T1, initial_T2, initial_detuning]
                    if fit_T1 else [initial_T2, initial_detuning])
    if fit_angle:
        guess_values.append(initial_angle)
    guess = np.asarray(guess_values, dtype=float)
    progress = tqdm(total=maxiter, desc="Optimizing T1/T2") if tqdm else None
    def update_progress(_params):
        if progress:
            progress.update(1)
    result = minimize(objective, guess, method="Nelder-Mead",
                      callback=update_progress,
                      options={"maxiter": maxiter, "disp": True, "xatol": 100, "fatol": 1e-5})
    if progress: progress.close()
    if fit_T1:
        best_T1, best_T2, best_detuning = result.x[:3]
        cursor = 3
    else:
        best_T2, best_detuning = result.x[:2]
        best_T1, cursor = fixed_T1, 2
    best_angle = result.x[cursor] if fit_angle else initial_angle
    sim = simulator.simulate(decay_times, best_T1, best_T2, alpha=initial_alpha,
        detuning_MHz=best_detuning, angle_offset=best_angle,
        return_real=(both or component == "real"),
        return_imag=(both or component == "imag"))
    if first_angle is not None:
        sim = rotate_first_point(sim, first_angle, best_angle)
    return {"T1": best_T1, "T2": best_T2, "detuning": best_detuning,
            "angle": best_angle,
            "cost": result.fun, "simulation": sim,
            "history": np.asarray(history), "result": result}
