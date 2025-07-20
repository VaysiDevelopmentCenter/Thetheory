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
lambda_vals = mu**2 / ligo_M_node

# Adjust y-limits to make the LIGO line visible
new_ylim_min = 1e-5
new_ylim_max = 1e1

# Mask for points within visible y-range
mask = (lambda_vals >= new_ylim_min) & (lambda_vals <= new_ylim_max)

# Extract only the M_node values in the visible region for dynamic level selection
visible_mask = (lmbda_grid >= new_ylim_min) & (lmbda_grid <= new_ylim_max)
if np.any(visible_mask):
    min_val = np.min(M_node[visible_mask])
    max_val = np.max(M_node[visible_mask])
    level_min = np.floor(np.log10(min_val))
    level_max = np.ceil(np.log10(max_val))
    levels = np.logspace(level_min, level_max, 40)
else:
    levels = np.logspace(7, 17, 50) # Fallback levels

# Create the figure
plt.figure(figsize=(10, 7))

# Contour plot with adjusted levels
cp = plt.contourf(mu_grid, lmbda_grid, M_node, levels=levels, cmap='plasma', extend='both')
plt.contour(mu_grid, lmbda_grid, M_node, levels=levels, colors='white', linewidths=0.5)

# Colorbar
cbar = plt.colorbar(cp)
cbar.set_label(r'$M_{\rm node}$ (eV/c$^2$)', fontsize=14)

# Plot LIGO line (with scatter markers too)
plt.plot(mu[mask], lambda_vals[mask], color='cyan', linestyle='--', linewidth=4, label='LIGO-like Threshold', zorder=100)
plt.scatter(mu[mask], lambda_vals[mask], color='red', marker='X', s=50, zorder=101)

# Axes settings
plt.xscale('log')
plt.yscale('log')
plt.xlim(1e4, 1e6)
plt.ylim(new_ylim_min, new_ylim_max)
plt.xlabel(r'$\mu$ (eV)', fontsize=14)
plt.ylabel(r'$\lambda$', fontsize=14)
plt.title(r'Figure 9: Parameter Space of $M_{\rm node}$ in $(\mu, \lambda)$', fontsize=16)
plt.legend(fontsize=12)
plt.tight_layout()

# Save and show
plt.savefig("figure_9_final_resolved.png", dpi=300)
plt.show()
