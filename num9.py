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

# Adjust the y-axis limits to ensure the LIGO threshold line is visible
# The original lambda_vals range from approx 1.78e-4 to 1.78
# So, we set the new y-limits to encompass this range comfortably.
new_ylim_min = 10**-5
new_ylim_max = 10**1

# Mask for the LIGO line within the new y-axis limits
mask = (lambda_vals >= new_ylim_min) & (lambda_vals <= new_ylim_max)

# Create the figure
plt.figure(figsize=(10, 7))

# --- تغییرات اصلی اینجا اعمال شد ---
# Contour plot
# Adjust levels to match the new visible M_node range (from ~10^7 to ~10^17)
# This will make the colored background visible again.
levels = np.logspace(7, 17, 50) # Adjusted levels for M_node based on new ylim
cp = plt.contourf(mu_grid, lmbda_grid, M_node, levels=levels, cmap='plasma', extend='both')
plt.contour(mu_grid, lmbda_grid, M_node, levels=levels, colors='white', linewidths=0.5)

# Colorbar
cbar = plt.colorbar(cp)
cbar.set_label(r'$M_{\rm node}$ (eV/c$^2$)', fontsize=14)

# Highlighted LIGO threshold line - Now it should be VERY visible!
# Changed color to cyan, added zorder for plot, changed scatter color/marker for emphasis
plt.plot(mu[mask], lambda_vals[mask], color='cyan', linestyle='--', linewidth=5, label='LIGO-like Threshold', zorder=100)
plt.scatter(mu[mask], lambda_vals[mask], color='red', s=50, zorder=101, marker='X')
# --- پایان تغییرات ---

# Axes and labels
plt.xscale('log')
plt.yscale('log')
plt.xlim(1e4, 1e6)
plt.ylim(new_ylim_min, new_ylim_max) # Apply the new y-limits

plt.xlabel(r'$\mu$ (eV)', fontsize=14)
plt.ylabel(r'$\lambda$', fontsize=14)
plt.title(r'Figure 9: Parameter Space of $M_{\rm node}$ in $(\mu, \lambda)$', fontsize=16)
plt.legend(fontsize=12)
plt.tight_layout()

# Save with a new filename to avoid confusion with previous outputs
output_path = "figure_9_final_highlightedx.png"
plt.savefig(output_path, dpi=300)
plt.show()
