import numpy as np
import matplotlib.pyplot as plt

# Constants
G = 6.67430e-11  # gravitational constant

# Masses and positions of two objects
m1, m2 = 5e24, 7e24  # kg
pos1 = np.array([-1.0, 0.0])  # object 1
pos2 = np.array([1.0, 0.0])   # object 2

# Create a lattice/grid
x = np.linspace(-3, 3, 30)
y = np.linspace(-2, 2, 20)
X, Y = np.meshgrid(x, y)

# Initialize force components
Fx = np.zeros(X.shape)
Fy = np.zeros(Y.shape)

# Calculate gravitational force at each grid point
for i in range(X.shape[0]):
    for j in range(X.shape[1]):
        point = np.array([X[i, j], Y[i, j]])
        
        # Force from object 1
        r1 = point - pos1
        r1_mag = np.linalg.norm(r1)
        if r1_mag != 0:
            F1 = -G * m1 * r1 / r1_mag**3  # vector force per unit mass
        else:
            F1 = np.array([0.0, 0.0])
        
        # Force from object 2
        r2 = point - pos2
        r2_mag = np.linalg.norm(r2)
        if r2_mag != 0:
            F2 = -G * m2 * r2 / r2_mag**3
        else:
            F2 = np.array([0.0, 0.0])
        
        # Total force
        F = F1 + F2
        Fx[i, j] = F[0]
        Fy[i, j] = F[1]

# Normalize for quiver plotting
F_mag = np.sqrt(Fx**2 + Fy**2)
Fx_norm = Fx / F_mag
Fy_norm = Fy / F_mag

# Plot
plt.figure(figsize=(8, 6))
plt.quiver(X, Y, Fx_norm, Fy_norm, F_mag, pivot='middle', cmap='plasma')
plt.plot(pos1[0], pos1[1], 'ro', markersize=10, label='Object 1')
plt.plot(pos2[0], pos2[1], 'bo', markersize=10, label='Object 2')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Gravitational Field Between Two Objects')
plt.colorbar(label='Force magnitude')
plt.legend()
plt.grid(True)
plt.show()
