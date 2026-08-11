import numpy as np
import matplotlib.pyplot as plt


def plot_characteristic_functions(
    charfuncs,
    x_list,
    y_list,
    decay_times,
    title="Characteristic Functions",
    vmin=None,
    vmax=None,
):

    n = len(charfuncs)

    cols = 4
    rows = int(np.ceil(n / cols))

    fig, axes = plt.subplots(
        rows,
        cols,
        figsize=(18, 12),
        sharex=True,
        sharey=True,
    )

    axes = np.ravel(axes)

    for i in range(n):

        ax = axes[i]

        im = ax.pcolormesh(
            x_list[i],
            y_list[i],
            charfuncs[i],
            shading="auto",
            cmap="bwr",
            vmin=vmin,
            vmax=vmax,
        )

        ax.axvline(
            0,
            color="k",
            ls="--",
            lw=0.8,
        )

        ax.axhline(
            0,
            color="k",
            ls="--",
            lw=0.8,
        )

        ax.set_aspect("equal")

        decay = decay_times[i]

        if decay >= 1000:
            label = f"{decay/1000:g} μs"
        else:
            label = f"{decay:g} ns"

        ax.set_title(label)

        plt.colorbar(
            im,
            ax=ax,
            fraction=0.046,
            pad=0.04,
        )

    for j in range(n, len(axes)):
        fig.delaxes(axes[j])

    plt.suptitle(title)

    plt.tight_layout()

    plt.show()


def plot_landscape(
    cost,
    T1_values,
    T2_values,
    best_T1,
    best_T2,
    title="Grid Search Residue",
):

    plt.figure(figsize=(7, 6))

    plt.imshow(
        cost,
        origin="lower",
        aspect="auto",
        extent=[
            T2_values[0] / 1000,
            T2_values[-1] / 1000,
            T1_values[0] / 1000,
            T1_values[-1] / 1000,
        ],
    )

    plt.scatter(
        best_T2 / 1000,
        best_T1 / 1000,
        marker="x",
        s=120,
        c="red",
        label="Best",
    )

    plt.xlabel("T2 (μs)")
    plt.ylabel("T1 (μs)")
    plt.title(title)

    plt.colorbar(label="Residue")

    plt.legend()

    plt.tight_layout()

    plt.show()
