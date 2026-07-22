"""Affine-normalized image comparison and rotation fitting."""
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import rotate


def build_mask(x, y, radius=2.5):
    X, Y = np.meshgrid(x, y)
    return (X * X + Y * Y) <= radius * radius


def affine_fit(exp_img, sim_img, mask):
    exp, sim = exp_img[mask].ravel(), sim_img[mask].ravel()
    a, b = np.linalg.lstsq(np.column_stack((sim, np.ones_like(sim))), exp, rcond=None)[0]
    if abs(a) < np.finfo(float).eps: a = np.finfo(float).eps
    return (exp_img - b) / a, a, b


def correlation(exp_img, sim_img, mask):
    exp, sim = exp_img[mask].ravel(), sim_img[mask].ravel()
    exp, sim = exp - exp.mean(), sim - sim.mean()
    den = np.linalg.norm(exp) * np.linalg.norm(sim)
    return float(np.dot(exp, sim) / den) if den else 0.0


def best_rotated_simulation(exp_img, sim_img, mask, candidate_angles=None):
    angles = np.asarray(np.linspace(-180, 180, 361) if candidate_angles is None else candidate_angles)
    best = (-np.inf, angles[0], None, None)
    for angle in angles:
        sim_rot = rotate(sim_img, float(angle), reshape=False, order=1, mode="nearest")
        exp_fit, _, _ = affine_fit(exp_img, sim_rot, mask)
        overlap = correlation(exp_fit, sim_rot, mask)
        if overlap > best[0]: best = (overlap, float(angle), exp_fit, sim_rot)
    return best[2], best[3], best[0], best[1]


def _single(exp_data, sim_data, mask, rotate_images, rotation_angles, weights):
    costs, previous = [], None
    for exp, sim in zip(exp_data, sim_data):
        angles = rotation_angles if previous is None else rotation_angles[np.abs(rotation_angles - previous) <= 90]
        if len(angles) == 0: angles = np.array([previous])
        if rotate_images: exp_fit, sim_fit, overlap, previous = best_rotated_simulation(exp, sim, mask, angles)
        else: exp_fit, _, _ = affine_fit(exp, sim, mask); sim_fit, overlap = sim, correlation(exp_fit, sim, mask)
        diff = exp_fit[mask] - sim_fit[mask]
        scale = np.std(exp_fit[mask]) or 1.0
        costs.append(0.4 * np.sqrt(np.mean(diff * diff)) / scale + 0.6 * (1 - overlap))
    return float(np.average(costs, weights=weights))


def residue(exp_data, sim_data, mask, weights=None, rotate_images=True, rotation_angles=None):
    angles = np.linspace(-180, 180, 361) if rotation_angles is None else np.asarray(rotation_angles)
    n = len(exp_data[0] if isinstance(exp_data, tuple) else exp_data)
    weights = np.ones(n) if weights is None else np.asarray(weights)
    if isinstance(exp_data, tuple):
        return 0.5 * (_single(exp_data[0], sim_data[0], mask, rotate_images, angles, weights) +
                      _single(exp_data[1], sim_data[1], mask, rotate_images, angles, weights))
    return _single(exp_data, sim_data, mask, rotate_images, angles, weights)


def residue_no_rotation(exp_data, sim_data, mask, weights=None):
    return residue(exp_data, sim_data, mask, weights, rotate_images=False)


def extract_rotation_angles(exp_data, sim_data, mask, rotation_angles=None):
    angles = np.linspace(-180, 180, 361) if rotation_angles is None else np.asarray(rotation_angles)
    exp, sim = (exp_data[0], sim_data[0]) if isinstance(exp_data, tuple) else (exp_data, sim_data)
    out, corr, previous = [], [], None
    for e, s in zip(exp, sim):
        candidates = angles if previous is None else angles[np.abs(angles - previous) <= 90]
        if len(candidates) == 0: candidates = np.array([previous])
        _, _, c, a = best_rotated_simulation(e, s, mask, candidates)
        out.append(a); corr.append(c); previous = a
    return np.asarray(out), np.asarray(corr)


def compare_experiment_simulation(exp_data, sim_data, x_list, y_list, mask,
                                  weights=None, rotate_images=True, rotation_angles=None):
    angles, overlaps = extract_rotation_angles(exp_data, sim_data, mask, rotation_angles) if rotate_images else (np.zeros(len(exp_data[0] if isinstance(exp_data, tuple) else exp_data)), np.array([]))
    components = [("Real", exp_data[0], sim_data[0]), ("Imaginary", exp_data[1], sim_data[1])] if isinstance(exp_data, tuple) else [("", exp_data, sim_data)]
    for label, exp, sim in components:
        fig, axes = plt.subplots(len(exp), 3, figsize=(10, 3 * len(exp)), squeeze=False)
        for i, (e, s) in enumerate(zip(exp, sim)):
            if rotate_images:
                fitted, displayed, _, _ = best_rotated_simulation(e, s, mask, rotation_angles)
            else:
                fitted, _, _ = affine_fit(e, s, mask); displayed = s
            images = (fitted, displayed, fitted - displayed)
            for j, image in enumerate(images):
                im = axes[i, j].pcolormesh(x_list[i], y_list[i], image, shading="auto", cmap="bwr")
                axes[i, j].set_aspect("equal"); plt.colorbar(im, ax=axes[i, j])
        fig.suptitle(label or "Experiment vs simulation"); fig.tight_layout(); plt.show()
    return angles, overlaps
