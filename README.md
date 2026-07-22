# Fitting coherent T1/T2 decay

This project compares measured characteristic-function data with simulations and estimates:

- `T1` and `T2` decay times
- the simulation parameter `alpha`
- detuning from image rotation

The main working example is [`Fitting_T1&T2.ipynb`](Fitting_T1%26T2.ipynb). The reusable code is in the `helpers/` folder.

## Very simple setup

1. Install Python and the required packages:

   ```bash
   pip install numpy matplotlib scipy h5py qutip jupyter
   ```

2. Put your experimental data on your computer. The data files are not included in this repository.

3. Open the notebook:

   ```bash
   jupyter notebook "Fitting_T1&T2.ipynb"
   ```

4. In the first code cell, update `real_dir` and `imag_dir` so they point to your data folders. Also adjust the decay times and other switches if needed.

5. Run the notebook cells from top to bottom. The notebook will load the data, simulate the decay, search for approximate `T1`/`T2` values, estimate detuning, and perform a final optimization.

## What the main files do

- `helpers/io.py` — loads the experimental data.
- `helpers/simulation.py` — simulates characteristic functions.
- `helpers/comparison.py` — compares measured and simulated images.
- `helpers/fitting.py` — searches for a good `alpha` and rotation.
- `helpers/optimizer.py` — refines `T1`, `T2`, and detuning.
- `helpers/plotting.py` — creates the plots.

The coherent-decay notebook is intentionally excluded from Git by `.gitignore`; use the included `Fitting_T1&T2.ipynb` as the starting point.
