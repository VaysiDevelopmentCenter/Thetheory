import numpy as np
import matplotlib.pyplot as plt

# Define the parameter space
mu = np.logspace(4, 6, 500)  # mu in eV
lmbda = np.logspace(-14.5, -11.5, 500)  # lambda is unitless
mu_grid, lmbda_grid = np.meshgrid(mu, lmbda)

# Calculate M_node in eV
M_node = mu_grid**2 / lmbda_grid

# LIGO-like detectability threshold
ligo_M_node = 5.6e11  # eV

# Create the figure
plt.figure(figsize=(10, 7))

# Plot the contour levels for M_node
contour_levels = np.logspace(10, 14, 5)
cp = plt.contourf(mu_grid, lmbda_grid, M_node, levels=contour_levels, cmap='viridis')
plt.contour(mu_grid, lmbda_grid, M_node, levels=contour_levels, colors='white', linewidths=0.5)

# Add a colorbar
cbar = plt.colorbar(cp)
cbar.set_label(r'$M_{node}$ (eV/c$^2$)', fontsize=14)

# Plot the LIGO detectability threshold line
lmbda_ligo = mu**2 / ligo_M_node
plt.plot(mu, lmbda_ligo, 'r--', linewidth=2, label='LIGO-like Detectability Threshold')

# Set plot scales, labels, and title
plt.xscale('log')
plt.yscale('log')
plt.xlabel(r'$\mu$ (eV)', fontsize=14)
plt.ylabel(r'$\lambda$', fontsize=14)
plt.title(r'Figure 9: Contour Plot of $M_{\rm node}$ in $(\mu, \lambda)$ Space', fontsize=16)
plt.legend(fontsize=12)
plt.tight_layout()

# Save the figure
plt.savefig('figure_9.png', dpi=300)

print("Figure 9 has been regenerated and saved as figure_9.png")
