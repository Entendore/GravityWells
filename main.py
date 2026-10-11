import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Constants
G = 6.67430e-11
dt = 0.01  # time step

# Two objects with masses, initial positions and velocities
m1, m2 = 5e24, 7e24
pos1 = np.array([-1.0, 0.0])
pos2 = np.array([1.0, 0.0])
vel1 = np.array([0.0, 0.1])
vel2 = np.array([0.0, -0.1])

# Lattice/grid setup
x = np.linspace(-3, 3, 25)
y = np.linspace(-2, 2, 20)
X, Y = np.meshgrid(x, y)

# Prepare the figure
fig, ax = plt.subplots(figsize=(8, 6))
ax.set_xlim(-3, 3)
ax.set_ylim(-2, 2)
ax.set_aspect('equal')
ax.grid(True)

# Plot objects
obj1_dot, = ax.plot([], [], 'ro', markersize=10)
obj2_dot, = ax.plot([], [], 'bo', markersize=10)
trail1, = ax.plot([], [], 'r-', linewidth=1)
trail2, = ax.plot([], [], 'b-', linewidth=1)

# Initial quiver (force field)
Fx_init = np.zeros_like(X)
Fy_init = np.zeros_like(Y)
quiver = ax.quiver(X, Y, Fx_init, Fy_init, np.sqrt(Fx_init**2 + Fy_init**2), cmap='plasma', scale=20)

# Store trails
x1_trail, y1_trail = [], []
x2_trail, y2_trail = [], []

def compute_field(pos1, pos2, m1, m2):
    Fx = np.zeros(X.shape)
    Fy = np.zeros(Y.shape)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            point = np.array([X[i, j], Y[i, j]])
            
            # Force from object 1
            r1 = point - pos1
            r1_mag = np.linalg.norm(r1)
            F1 = -G * m1 * r1 / r1_mag**3 if r1_mag != 0 else np.array([0.0, 0.0])
            
            # Force from object 2
            r2 = point - pos2
            r2_mag = np.linalg.norm(r2)
            F2 = -G * m2 * r2 / r2_mag**3 if r2_mag != 0 else np.array([0.0, 0.0])
            
            F = F1 + F2
            Fx[i, j] = F[0]
            Fy[i, j] = F[1]
    
    # Normalize for direction
    F_mag = np.sqrt(Fx**2 + Fy**2)
    Fx_norm = Fx / (F_mag + 1e-20)
    Fy_norm = Fy / (F_mag + 1e-20)
    
    return Fx_norm, Fy_norm, F_mag

def update(frame):
    global pos1, pos2, vel1, vel2
    
    # Compute mutual gravitational forces
    r12 = pos2 - pos1
    r12_mag = np.linalg.norm(r12)
    F12 = G * m1 * m2 * r12 / r12_mag**3
    
    # Update velocities
    vel1 += F12 / m1 * dt
    vel2 -= F12 / m2 * dt
    
    # Update positions
    pos1 += vel1 * dt
    pos2 += vel2 * dt
    
    # Update trails
    x1_trail.append(pos1[0])
    y1_trail.append(pos1[1])
    x2_trail.append(pos2[0])
    y2_trail.append(pos2[1])
    
    # Compute gravitational field
    Fx_norm, Fy_norm, F_mag = compute_field(pos1, pos2, m1, m2)
    
    # Update quiver with magnitude-based color
    quiver.set_UVC(Fx_norm, Fy_norm, F_mag)
    
    # Update objects and trails
    obj1_dot.set_data(pos1[0], pos1[1])
    obj2_dot.set_data(pos2[0], pos2[1])
    trail1.set_data(x1_trail, y1_trail)
    trail2.set_data(x2_trail, y2_trail)
    
    return obj1_dot, obj2_dot, trail1, trail2, quiver

ani = FuncAnimation(fig, update, frames=500, interval=50, blit=True)
plt.colorbar(quiver, ax=ax, label='Force magnitude')
plt.title("Dynamic Gravitational Field Between Two Objects")
plt.show()
