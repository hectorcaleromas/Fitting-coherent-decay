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
            # ``simulate`` returns one image per requested decay time.  This
            # alpha fit uses only the first decay time, so select its 2-D
            # image before applying the 2-D boolean mask.
            sim_img = sim[0]

        for angle in rotation_values:

            if both:

                sr_rot = rotate(
                    sim_real,
                    angle,
                    reshape=False,
                    order=1,
                    mode="nearest",
                )

                si_rot = rotate(
                    sim_imag,
                    angle,
                    reshape=False,
                    order=1,
                    mode="nearest",
                )

                sr, _, _ = affine_fit(
                    exp_real,
                    sr_rot,
                    mask,
                )

                si, _, _ = affine_fit(
                    exp_imag,
                    si_rot,
                    mask,
                )

                diff_real = sr[mask] - sr_rot[mask]
                diff_imag = si[mask] - si_rot[mask]

                cost = 0.5 * (
                    np.sqrt(np.mean(diff_real**2))
                    +
                    np.sqrt(np.mean(diff_imag**2))
                )

            else:

                s_rot = rotate(
                    sim_img,
                    angle,
                    reshape=False,
                    order=1,
                    mode="nearest",
                )

                s, _, _ = affine_fit(
                    exp_img,
                    s_rot,
                    mask,
                )

                diff = s[mask] - s_rot[mask]

                cost = np.sqrt(
                    np.mean(diff**2)
                )

            if cost < best["cost"]:

                best["cost"] = cost
                best["alpha"] = alpha
                best["angle"] = angle
                best["sim"] = sim

    return best
