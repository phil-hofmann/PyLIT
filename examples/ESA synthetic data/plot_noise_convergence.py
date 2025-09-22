import numpy as np

pi = np.pi
import matplotlib.pyplot as plt
from matplotlib.cm import viridis, plasma
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset

import sys
import time
import glob

if len(sys.argv) != 4:
    print("Usage: python script.py <kernel_name> <method_name> <kindex>")
    sys.exit(1)

# -------------------------------------------------- #
# 0) Handle function arguments and declare variables #
# -------------------------------------------------- #
kernel_name = sys.argv[1]  # selects kernels, must be either 'Gauss' or 'Uniform'
method_name = sys.argv[
    2
]  # selects regularizer, must be 'l2_fit', 'max_entropy_fit', 'cdf_l2_fit'
kindex = sys.argv[3]  # selects the wavenumber (k) index to plot
omega_max = 3.0

# Import data
noise0p001_file = (
    "synthetic_example/outputs/chi2kink/Skw_"
    + kernel_name
    + "_"
    + method_name
    + "_synthetic_noise_0p001_k"
    + kindex
    + "_rs10.csv"
)
noise0p001 = np.loadtxt(noise0p001_file, delimiter=",", skiprows=1)
noise0p01_file = (
    "synthetic_example/outputs/chi2kink/Skw_"
    + kernel_name
    + "_"
    + method_name
    + "_synthetic_noise_0p01_k"
    + kindex
    + "_rs10.csv"
)
noise0p01 = np.loadtxt(noise0p01_file, delimiter=",", skiprows=1)
noise0p1_file = (
    "synthetic_example/outputs/chi2kink/Skw_"
    + kernel_name
    + "_"
    + method_name
    + "_synthetic_noise_0p1_k"
    + kindex
    + "_rs10.csv"
)
noise0p1 = np.loadtxt(noise0p1_file, delimiter=",", skiprows=1)
prior_file = "synthetic_example/inputs/D_k" + kindex + "_rs10.csv"
prior = np.loadtxt(prior_file, delimiter=",")
exact_file = (
    "synthetic_example/true_solutions/Skw_static-Lindhard_N34_rs10_theta1_k"
    + kindex
    + ".dat"
)
Nomega = len(prior[:, 0])
true_solution = np.loadtxt(exact_file, delimiter=",", skiprows=1)[:Nomega, :]
ana_cont_rec_file = (
    "synthetic_example/Bryan_chi2k_solutions/S_sigma0p001_static-Lindhard_N34_rs10_theta1_num"
    + kindex
    + ".dat"
)
ana_cont_rec_noise0p001 = np.loadtxt(ana_cont_rec_file, delimiter=",", skiprows=1)[
    :Nomega, :
]

print(
    "shapes of arrays (some of error column)",
    np.shape(noise0p001),
    np.shape(noise0p01),
    np.shape(noise0p1),
    np.shape(prior),
    np.shape(true_solution),
    np.shape(ana_cont_rec_noise0p001),
)
# Units
q_file = "synthetic_example/inputs/chi0q_ideal-Lindhard_N34_rs10_theta1.dat"
qs_kFe = np.loadtxt(
    exact_file, delimiter=",", skiprows=1
)  # wavenumbers normalized by Fermi wavenumber
r_s = 10
w_pe_Hartree = np.sqrt(3 / float(r_s) ** 3)

# plots
colors = plasma(np.flip(np.linspace(0.0, 0.75, 3, endpoint=True)))
fig, ax = plt.subplots(figsize=(5, 5))
ax.plot(
    prior[:, 0] / w_pe_Hartree, prior[:, 1], c="black", marker="x", ls="-", markevery=35
)
for Skw, color in zip([noise0p1, noise0p01, noise0p001], colors):
    ax.plot(
        Skw[:, 0] / w_pe_Hartree,
        Skw[:, 1],
        color=color,
        marker="v",
        ls="-",
        markevery=40,
        alpha=0.5,
    )
ax.plot(
    ana_cont_rec_noise0p001[:, 0] / w_pe_Hartree,
    ana_cont_rec_noise0p001[:, 1],
    c="green",
    marker="o",
    ls="--",
    markevery=50,
)
ax.plot(
    true_solution[:, 0] / w_pe_Hartree,
    true_solution[:, 1],
    c="black",
    marker="o",
    ls="--",
    markevery=50,
)
for Skw, color in zip([noise0p1, noise0p01, noise0p001], colors):
    ax.fill_between(
        Skw[:, 0] / w_pe_Hartree,
        Skw[:, 1] - Skw[:, 2],
        Skw[:, 1] + Skw[:, 2],
        alpha=0.5,
        color=color,
    )

ax.set_title(
    r"$S(q,\omega)$ at $r_s=10$ $\theta=1$ $q = $"
    + str(np.round(qs_kFe[int(kindex), 0], 2))
    + "$ q_F$\n regularized with CDF $L_2$ distance",
    fontsize=14,
)
ax.set_xlabel(r"$\omega/\omega_{p,e}$", fontsize=16)
ax.set_xlim([0, omega_max])
ax.set_ylim(
    [
        0,
        1.1
        * np.amax(
            [
                prior[:, 1],
                noise0p1[:, 1],
                noise0p01[:, 1],
                noise0p001[:, 1],
                true_solution[:, 1],
            ]
        ),
    ]
)
ax.tick_params(axis="both", which="major", labelsize=14, width=2)  # Major ticks
ax.tick_params(
    axis="both", which="minor", labelsize=12, width=1
)  # Minor ticks (if any)
ax.legend(
    [
        "RPA (default)",
        r"$\sigma_0=0.1$",
        r"$\sigma_0=0.01$",
        r"$\sigma_0=0.001$",
        r"$\chi^2$k $\sigma_0=0.001$",
        "static (exact)",
    ],
    fontsize=12,
)

plt.savefig(
    "Skw_noise_convergence_LFCRPA_rs10_theta1_q"
    + str(kindex)
    + "_"
    + kernel_name
    + "_"
    + method_name
    + ".pdf",
    bbox_inches="tight",
    format="pdf",
)
print(
    "Skw_noise_convergence_LFCRPA_rs10_theta1_q"
    + str(kindex)
    + "_"
    + kernel_name
    + "_"
    + method_name
    + ".pdf"
)
