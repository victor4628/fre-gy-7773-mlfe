# Geometric interpretation of Ridge regression (L2 regularization)

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

# Plot the L2 ball (circle constraint)
# In Ridge regression, the constraint is ||theta||_2^2 <= T
radius = np.sqrt(2.0)
circle = plt.Circle(
    (0, 0), radius, color="deepskyblue", alpha=0.6, label=r"$\ell_2$ Constraint Ball"
)
plt.gca().add_patch(circle)
plt.axhline(0, color="black", linewidth=1)
plt.axvline(0, color="black", linewidth=1)

# Mark the unconstrained OLS solution
plt.plot(center[0], center[1], "ro", markersize=8, label="OLS Solution")

# Mark the Ridge solution (where the ellipse is tangent to the circle)
ridge_sol = np.array([0.9, 1.1])  # Approximate tangency point for visualization
plt.plot(
    ridge_sol[0],
    ridge_sol[1],
    ".",
    color="orange",
    markersize=15,
    label="Ridge Solution",
)

# Formatting
plt.title(r"Geometric Interpretation of Ridge Regression ($\ell_2$ ball)")
plt.xlabel(r"$\theta_1$")
plt.ylabel(r"$\theta_2$")
plt.xlim(-1.5, 4.5)
plt.ylim(-1.5, 4.5)
plt.legend(loc="upper left")
# plt.savefig("figures/l2_ridge.pdf", bbox_inches="tight")
plt.show()
