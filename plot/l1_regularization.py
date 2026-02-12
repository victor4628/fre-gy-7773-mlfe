# Geometric interpretation of Lasso regression (L1 regularization)

import numpy as np
import matplotlib.pyplot as plt

# Set grid for theta_1 and theta_2
theta1 = np.linspace(-1.5, 4.5, 400)
theta2 = np.linspace(-1.5, 4.5, 400)
T1, T2 = np.meshgrid(theta1, theta2)

# Define the OLS loss function (ellipsoid)
# We assume the OLS solution is at (3, 3) and features are correlated
center = np.array([3, 3])
A = np.array([[2, -0.8], [-0.8, 1]])  # Precision matrix defining the ellipse tilt


def ols_loss(t1, t2):
    diff1 = t1 - center[0]
    diff2 = t2 - center[1]
    return A[0, 0] * diff1**2 + (A[0, 1] + A[1, 0]) * diff1 * diff2 + A[1, 1] * diff2**2


Z = ols_loss(T1, T2)

# Create the plot
plt.figure(figsize=(10, 8))

# Plot contours of the OLS loss with colormap
levels = np.linspace(Z.min(), Z.max(), 100)
contourf = plt.contourf(T1, T2, Z, levels=levels, cmap="viridis", alpha=0.8)
plt.colorbar(contourf, label="Loss Value")

# Plot the L1 ball (diamond constraint)
# In Lasso: ||theta||_1 <= T
T_l1 = 2.0
diamond = np.array(
    [
        [T_l1, 0.0],
        [0.0, T_l1],
        [-T_l1, 0.0],
        [0.0, -T_l1],
        [T_l1, 0.0],
    ]
)

plt.plot(
    diamond[:, 0],
    diamond[:, 1],
    color="deepskyblue",
    linewidth=3,
    label=r"$\ell_1$ Constraint Ball",
)

# Fill for visibility
plt.fill(diamond[:, 0], diamond[:, 1], color="deepskyblue", alpha=0.4)

# Highlight axes and solutions
plt.axhline(0, color="black", linewidth=1)
plt.axvline(0, color="black", linewidth=1)

# OLS solution
plt.plot(center[0], center[1], "ro", markersize=8, label="OLS Solution")

# Lasso solution (tangency at a corner → sparsity)
lasso_sol = np.array([2.0, 0.0])  # illustrative corner solution
plt.plot(
    lasso_sol[0],
    lasso_sol[1],
    ".",
    color="orange",
    markersize=15,
    label="Lasso Solution",
)

# Formatting
plt.title(r"Geometric Interpretation of Lasso ($\ell_1$ ball)")
plt.xlabel(r"$\theta_1$")
plt.ylabel(r"$\theta_2$")
plt.xlim(-1.5, 4.5)
plt.ylim(-1.5, 4.5)
plt.legend(loc="upper left")
# plt.savefig("figures/l1_lasso.pdf", bbox_inches="tight")
plt.show()
