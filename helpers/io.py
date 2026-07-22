"""Discovery and loading of coherent-decay HDF5 data."""
import glob, os
from datetime import datetime
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
