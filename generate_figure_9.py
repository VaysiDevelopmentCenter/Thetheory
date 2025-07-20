import numpy as np
import matplotlib.pyplot as plt

mu = np.logspace(4, 6, 600)
lmbda = np.logspace(-14.5, -11.5, 600)
mu_grid, lmbda_grid = np.meshgrid(mu, lmbda)
M_node = mu_grid**2 / lmbda_grid

ligo_M_node = 5.6e11
lambda_vals = mu**2 / ligo_M_node
mask = (lambda_vals >= 10**-14.5) & (lambda_vals <= 10**-11.5)

plt.figure(figsize=(10, 7))
levels = np.logspace(19, 26.5, 50)
cp = plt.contourf(mu_grid, lmbda_grid, M_node, levels=levels, cmap='plasma', extend='both')
plt.contour(mu_grid, lmbda_grid, M_node, levels=levels, colors='white', linewidths=0.5)
cbar = plt.colorbar(cp)
cbar.set_label(r'$M_{\rm node}$ (eV/c$^2$)', fontsize=14)
plt.plot(mu[mask], lambda_vals[mask], 'r--', linewidth=3, label='LIGO-like Threshold')
plt.xscale('log')
plt.yscale('log')
plt.xlim(1e4, 1e6)
plt.ylim(10**-14.5, 10**-11.5)
plt.xlabel(r'$\mu$ (eV)', fontsize=14)
plt.ylabel(r'$\lambda$', fontsize=14)
plt.title(r'Figure 9: Parameter Space of $M_{\rm node}$ in $(\mu, \lambda)$', fontsize=16)
plt.legend(fontsize=12)
plt.tight_layout()
plt.savefig('figure_9_final_fixed.png', dpi=300)
