# Fitting coherent T1/T2 decay

This project aims to extract cavity T1 and T2 from fitting the decay of a coherent state in time.
Experimentally, we measure the characteristic function, and we try to match our Master equation simulations. Real, imaginary, or both kinds of data can be used in the fit. This script finds:

- `T1` and `T2` decay times
- the simulation parameter `alpha`
- detuning from image rotation

The workflow is quite simple. After extracting the experimental data we fit the first point (where the coherent state has ideally not decayed yet) to a coherent state. Since preparing the state takes some time, even a small detuning causes the characteristic function to rotate in phase space. For this reason, we also extract (by brute force) the angle that better adjusts to the experimental data. 

The next step is to perform a grid search. That is, we try different values of T1 and T2 and measure how much they overlap with the actual data. For each of these points we also find the best angle. This provides an estimate of the detuning and an initial guess for a multi-parameter optimization of T1 and T2.

Finally, using the standard scipy.optimize we fine tune the parameters T1 and T2. In case we already have measured T1 independently, the script can also optimize only over T2.

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
