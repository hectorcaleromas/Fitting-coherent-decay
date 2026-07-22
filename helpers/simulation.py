"""QuTiP characteristic-function simulation."""
import numpy as np
import qutip as qt
from scipy.ndimage import rotate


class CharacteristicFunctionSimulator:
    def __init__(self, N_c, x, y):
        self.N_c = int(N_c)
        self.x, self.y = np.asarray(x), np.asarray(y)
        self.Nx, self.Ny = len(self.x), len(self.y)
        self.a = qt.destroy(self.N_c)
        self.D_ops = [qt.displace(self.N_c, xr + 1j * yi)
                      for yi in self.y for xr in self.x]

    def simulate(self, decay_times, T1, T2, alpha=3.0, detuning_MHz=0.0,
                 angle_offset=0.0, return_real=True, return_imag=True, dt=100):
        if T1 <= 0 or T2 <= 0:
            raise ValueError("T1 and T2 must be positive")
        times = np.asarray(decay_times, dtype=float)
        if times.ndim != 1 or len(times) == 0 or np.any(times < 0):
            raise ValueError("decay_times must be a non-empty 1-D array")
        rho0 = qt.coherent_dm(self.N_c, alpha)
        gamma1 = 1.0 / T1
        gamma_phi = max(0.0, 1.0 / T2 - 1.0 / (2.0 * T1))
        c_ops = [np.sqrt(gamma1) * self.a]
        if gamma_phi > 0:
            c_ops.append(np.sqrt(2.0 * gamma_phi) * self.a.dag() * self.a)
        H = (2.0 * np.pi * detuning_MHz / 1000.0) * self.a.dag() * self.a
        dense = np.arange(0.0, times.max() + max(float(dt), 1e-12), dt)
        result = qt.mesolve(H, rho0, dense, c_ops=c_ops,
                            options={"nsteps": 50000})
        real, imag = ([] if return_real else None), ([] if return_imag else None)
        for t in times:
            rho = result.states[int(np.argmin(np.abs(dense - t)))]
            vals = np.asarray(qt.expect(self.D_ops, rho)).reshape(self.Ny, self.Nx)
            if return_real:
                image = -np.real(vals)
                if abs(angle_offset) > 1e-12:
                    image = rotate(image, angle_offset, reshape=False, order=1, mode="nearest")
                real.append(image)
            if return_imag:
                image = np.imag(vals)
                if abs(angle_offset) > 1e-12:
                    image = rotate(image, angle_offset, reshape=False, order=1, mode="nearest")
                imag.append(image)
        real = np.asarray(real) if return_real else None
        imag = np.asarray(imag) if return_imag else None
        if return_real and return_imag: return real, imag
        return real if return_real else imag
