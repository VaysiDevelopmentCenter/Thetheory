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

# --- تغییرات اینجا اعمال شد ---
# Adjust the y-axis limits to ensure the LIGO threshold line is visible
# The original lambda_vals range from approx 1.78e-4 to 1.78
# So, we need to extend the ylim beyond the original 10^-14.5 to 10^-11.5
# We'll set it to a range that comfortably includes the relevant lambda_vals
# For example, from 1e-5 to 1e0 or 1e1
new_ylim_min = 10**-5 # A lower bound that includes the LIGO line
new_ylim_max = 10**1  # An upper bound that includes the LIGO line or more
# --- پایان تغییرات ---

# Mask for the LIGO line within the new y-axis limits
mask = (lambda_vals >= new_ylim_min) & (lambda_vals <= new_ylim_max)

# Create the figure
plt.figure(figsize=(10, 7))

# Contour plot
# Adjust levels if necessary, based on the new visible M_node range
levels = np.logspace(19, 26.5, 50) # Keep current levels, adjust if plot looks empty
cp = plt.contourf(mu_grid, lmbda_grid, M_node, levels=levels, cmap='plasma', extend='both')
plt.contour(mu_grid, lmbda_grid, M_node, levels=levels, colors='white', linewidths=0.5)

# Colorbar
cbar = plt.colorbar(cp)
cbar.set_label(r'$M_{\rm node}$ (eV/c$^2$)', fontsize=14)

# Highlighted LIGO threshold line - now it should be visible!
plt.plot(mu[mask], lambda_vals[mask], color='orange', linestyle='--', linewidth=5, label='LIGO-like Threshold')
plt.scatter(mu[mask], lambda_vals[mask], color='white', s=20, zorder=10) # Add zorder for scatter too

# Axes and labels
plt.xscale('log')
plt.yscale('log')
plt.xlim(1e4, 1e6)

# --- اعمال محدوده Y جدید ---
plt.ylim(new_ylim_min, new_ylim_max)
# --- پایان اعمال محدوده Y جدید ---

plt.xlabel(r'$\mu$ (eV)', fontsize=14)
plt.ylabel(r'$\lambda$', fontsize=14)
plt.title(r'Figure 9: Parameter Space of $M_{\rm node}$ in $(\mu, \lambda)$', fontsize=16)
plt.legend(fontsize=12)
plt.tight_layout()

# Save
output_path = "figure_9_final_highlightedx.png"
plt.savefig(output_path, dpi=300)
plt.show()
