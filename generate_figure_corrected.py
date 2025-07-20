import numpy as np
import matplotlib.pyplot as plt

# Define parameter space
mu = np.logspace(4, 6, 600)  # mu ∈ [1e4, 1e6]
lmbda = np.logspace(-14.5, -11.5, 600)  # lambda ∈ [1e-14.5, 1e-11.5]
mu_grid, lmbda_grid = np.meshgrid(mu, lmbda)

# Compute M_node in eV
M_node = mu_grid**2 / lmbda_grid

# LIGO-like detectability threshold
ligo_M_node = 5.6e11
lmbda_ligo = mu**2 / ligo_M_node

# Create figure
plt.figure(figsize=(10, 7))

# Contour plot with proper levels
contour_levels = np.logspace(19, 26.5, 50)
cp = plt.contourf(mu_grid, lmbda_grid, M_node, levels=contour_levels, cmap='plasma', extend='both')
plt.contour(mu_grid, lmbda_grid, M_node, levels=contour_levels, colors='white', linewidths=0.5)

# Add colorbar
cbar = plt.colorbar(cp)
cbar.set_label(r'$M_{\rm node}$ (eV/c$^2$)', fontsize=14)

# Plot LIGO detectability threshold line
mask = (lmbda_ligo >= 10**-14.5) & (lmbda_ligo <= 10**-11.5)
plt.plot(mu[mask], lmbda_ligo[mask], 'r--', linewidth=2, label='LIGO-like Detectability Threshold')

# Axes and labels
plt.xscale('log')
plt.yscale('log')
plt.xlim(1e4, 1e6)
plt.ylim(10**-14.5, 10**-11.5)
plt.xlabel(r'$\mu$ (eV)', fontsize=14)
plt.ylabel(r'$\lambda$', fontsize=14)
plt.title(r'Figure 9: Parameter Space of $M_{\rm node}$ in $(\mu, \lambda)$', fontsize=16)
plt.legend(fontsize=12)
plt.tight_layout()

# Save figure
plt.savefig('figure_9_corrected.png', dpi=300)
