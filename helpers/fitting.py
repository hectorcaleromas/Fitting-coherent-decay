import numpy as np
from scipy.ndimage import rotate
from .comparison import affine_fit


def find_best_alpha(
    simulator,
    exp_data,
    decay_times,
    alpha_values,
    rotation_values,
    mask,
    component="real",
):

    both = isinstance(exp_data, tuple)

    if both:
        exp_real = exp_data[0][0]
        exp_imag = exp_data[1][0]
    else:
        exp_img = exp_data[0]

    best = {
        "alpha": None,
        "angle": None,
        "cost": np.inf,
        "sim": None,
    }

    print("Searching alpha...")

    for alpha in alpha_values:

        sim = simulator.simulate(
            decay_times=decay_times,
            T1=1e9,
            T2=1e9,
            alpha=alpha,
            return_real=(both or component == "real"),
            return_imag=(both or component == "imag"),
        )

        if both:
            sim_real = sim[0][0]
            sim_imag = sim[1][0]
        else:
            sim_img = sim

        for angle in rotation_values:

            if both:

                sr = rotate(
                    sim_real,
                    angle,
                    reshape=False,
                    order=1,
                    mode="nearest",
                )

                si = rotate(
                    sim_imag,
                    angle,
                    reshape=False,
                    order=1,
                    mode="nearest",
                )

                sr, _, _ = affine_fit(
                    exp_real,
                    sr,
                    mask,
                )

                si, _, _ = affine_fit(
                    exp_imag,
                    si,
                    mask,
                )

                diff_real = exp_real[mask] - sr[mask]
                diff_imag = exp_imag[mask] - si[mask]

                cost = 0.5 * (
                    np.sqrt(np.mean(diff_real**2))
                    +
                    np.sqrt(np.mean(diff_imag**2))
                )

            else:

                s = rotate(
                    sim_img,
                    angle,
                    reshape=False,
                    order=1,
                    mode="nearest",
                )

                s, _, _ = affine_fit(
                    exp_img,
                    s,
                    mask,
                )

                diff = exp_img[mask] - s[mask]

                cost = np.sqrt(
                    np.mean(diff**2)
                )

            if cost < best["cost"]:

                best["cost"] = cost
                best["alpha"] = alpha
                best["angle"] = angle
                best["sim"] = sim

    return best
