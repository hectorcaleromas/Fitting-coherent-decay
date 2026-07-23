"""Discovery and loading of coherent-decay data."""
import glob, os, re
from datetime import datetime
from pathlib import Path
import h5py
import numpy as np

def find_data_files(data_dir, decay_times, experiment_start="00-00-00"):
    start = datetime.strptime(experiment_start, "%H-%M-%S")
    files = []
    for path in glob.glob(os.path.join(data_dir, "*.hdf5")):
        name = os.path.basename(path)
        if "Char_Func_2D_decay_sweep" not in name: continue
        try: t = datetime.strptime(name.split("_")[0], "%H-%M-%S")
        except ValueError: continue
        files.append(((t - start).total_seconds() % 86400, path))
    files = [p for _, p in sorted(files)]
    if len(files) != len(decay_times):
        raise ValueError(f"Expected {len(decay_times)} files, found {len(files)}.")
    paired = sorted(zip(np.asarray(decay_times), files), key=lambda p: p[0])
    return [p for _, p in paired], np.asarray([t for t, _ in paired])

def load_real_imag(real_files=None, imag_files=None, beta_scale=1.0):
    def load(files):
        states, xs, ys = [], [], []
        for path in files:
            with h5py.File(path, "r") as data:
                states.append(np.mean(np.asarray(data["I"]), axis=0))
                xs.append(beta_scale * np.asarray(data["x_displace"]))
                ys.append(beta_scale * np.asarray(data["y_displace"]))
        return np.asarray(states), np.asarray(xs), np.asarray(ys)
    real = imag = x = y = None
    if real_files is not None: real, x, y = load(real_files)
    if imag_files is not None:
        imag, xi, yi = load(imag_files)
        if x is None: x, y = xi, yi
        elif not (np.allclose(x, xi) and np.allclose(y, yi)): raise RuntimeError("Real and Imag beta grids differ.")
    return real, imag, x, y

def load_experiment(real_dir=None, imag_dir=None, real_decays=None, imag_decays=None,
                    beta_scale=1.0, experiment_start="00-00-00"):
    rf = inf = ref = None
    if real_dir is not None: rf, ref = find_data_files(real_dir, real_decays, experiment_start)
    if imag_dir is not None:
        inf, di = find_data_files(imag_dir, imag_decays, experiment_start)
        if ref is None: ref = di
        elif not np.allclose(ref, di): raise RuntimeError("Real and Imag decay times differ.")
    real, imag, x, y = load_real_imag(rf, inf, beta_scale)
    return real, imag, x, y, ref


def load_processed_npz(processed_data_dir, delay_times_ns, delay_indices="all",
                       beta_scale=1.0):
    """Load calibrated ``delay_*_averaged_cf.npz`` files.

    Each file must contain ``alpha_real``, ``alpha_imag``, ``chi_real`` and
    ``chi_imag``.  The returned tuple has the same shape and ordering as
    :func:`load_experiment`, so downstream fitting code can use either source.
    """
    processed_data_dir = Path(processed_data_dir)
    delay_times_ns = np.asarray(delay_times_ns, dtype=float)

    def delay_index(file_path):
        match = re.search(r"delay_(\d+)_averaged_cf\.npz$", Path(file_path).name)
        if match is None:
            raise ValueError(f"Could not identify delay index from:\n{file_path}")
        return int(match.group(1))

    if not processed_data_dir.exists():
        raise FileNotFoundError(f"Processed NPZ folder does not exist:\n{processed_data_dir}")

    npz_files = sorted(processed_data_dir.glob("delay_*_averaged_cf.npz"), key=delay_index)
    if not npz_files:
        raise FileNotFoundError(
            f"No delay_*_averaged_cf.npz files were found in:\n{processed_data_dir}"
        )

    available = {delay_index(path): path for path in npz_files}
    selected = list(available) if delay_indices == "all" else [int(index) for index in delay_indices]
    missing = [index for index in selected if index not in available]
    if missing:
        raise FileNotFoundError(f"No NPZ file was found for delay indices:\n{missing}")

    real, imag, x_list, y_list, decay_times = [], [], [], [], []
    for index in selected:
        if index >= len(delay_times_ns):
            raise IndexError(
                f"Delay index {index} is present, but delay_times_ns only contains "
                f"{len(delay_times_ns)} values."
            )

        path = available[index]
        with np.load(path, allow_pickle=False) as data:
            required = {"alpha_real", "alpha_imag", "chi_real", "chi_imag"}
            missing_keys = required - set(data.files)
            if missing_keys:
                raise KeyError(f"{path.name} is missing keys: {sorted(missing_keys)}")
            alpha_real = np.asarray(data["alpha_real"], dtype=float)
            alpha_imag = np.asarray(data["alpha_imag"], dtype=float)
            chi_real = np.asarray(data["chi_real"], dtype=float)
            chi_imag = np.asarray(data["chi_imag"], dtype=float)

        expected_shape = (len(alpha_imag), len(alpha_real))
        if chi_real.shape != expected_shape or chi_imag.shape != expected_shape:
            raise ValueError(
                f"{path.name}: characteristic-function arrays must have shape "
                f"{expected_shape}."
            )

        x_list.append(beta_scale * alpha_real)
        y_list.append(beta_scale * alpha_imag)
        real.append(chi_real)
        imag.append(chi_imag)
        decay_times.append(delay_times_ns[index])

    return (
        np.asarray(real),
        np.asarray(imag),
        np.asarray(x_list),
        np.asarray(y_list),
        np.asarray(decay_times),
    )
