import numpy as np
import matplotlib.pyplot as plt

# Define parameter space
mu = np.logspace(4, 6, 600)
lmbda = np.logspace(-14.5, -11.5, 600)
mu_grid, lmbda_grid = np.meshgrid(mu, lmbda)

# Compute M_node
M_node = mu_grid**2 / lmbda_grid

# LIGO-like detectability threshold
ligo_M_node = 5.6e11
lambda_vals = np.logspace(-14.5, -11.5, 500)
mu_vals = np.sqrt(lambda_vals * ligo_M_node)

# Mask: only keep visible range
mask = (mu_vals >= 1e4) & (mu_vals <= 1e6)

# Plot
plt.figure(figsize=(10, 7))

# Filled contours
levels = np.logspace(19, 26.5, 50)
cp = plt.contourf(mu_grid, lmbda_grid, M_node, levels=levels, cmap='plasma', extend='both')
plt.contour(mu_grid, lmbda_grid, M_node, levels=levels, colors='white', linewidths=0.5)

# Colorbar
cbar = plt.colorbar(cp)
cbar.set_label(r'$M_{\rm node}$ (eV/c$^2$)', fontsize=14)

# Plot threshold line with high visibility
plt.plot(mu_vals[mask], lambda_vals[mask], color='cyan', linestyle='--', linewidth=4, label='LIGO-like Threshold')

# Debug dots
plt.scatter(mu_vals[mask], lambda_vals[mask], color='white', s=10)

# Log scales and labels
plt.xscale('log')
plt.yscale('log')
plt.xlim(1e4, 1e6)
plt.ylim(10**-14.5, 10**-11.5)
plt.xlabel(r'$\mu$ (eV)', fontsize=14)
plt.ylabel(r'$\lambda$', fontsize=14)
plt.title(r'Figure 9: Parameter Space of $M_{\rm node}$ in $(\mu, \lambda)$', fontsize=16)
plt.legend(fontsize=12)
plt.tight_layout()

# Save or show
plt.savefig('figure_9_with_threshold.png', dpi=300)
plt.show()
