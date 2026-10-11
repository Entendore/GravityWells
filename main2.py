import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.cm as cm

# Constants
G = 6.67430e-11
dt = 0.01

# Objects: mass, position, velocity, trail, color
objects = [
    {"mass": 5e24, "pos": np.array([-1.0, 0.0]), "vel": np.array([0.0, 0.2]), "trail_x": [], "trail_y": [], "color": "r"},
    {"mass": 7e24, "pos": np.array([1.0, 0.0]), "vel": np.array([0.0, -0.15]), "trail_x": [], "trail_y": [], "color": "b"},
    {"mass": 1e24, "pos": np.array([0.0, 1.5]), "vel": np.array([-0.2, 0.0]), "trail_x": [], "trail_y": [], "color": "g"}
]

# Grid for field visualization
x = np.linspace(-3, 3, 40)
y = np.linspace(-3, 3, 40)
X, Y = np.meshgrid(x, y)

# Figure setup
fig, ax = plt.subplots(figsize=(8, 8))
ax.set_xlim(-3, 3)
ax.set_ylim(-3, 3)
ax.set_aspect('equal')
ax.grid(True)

# Initialize quiver for gravitational field
Fx_init = np.zeros_like(X)
Fy_init = np.zeros_like(Y)
quiver = ax.quiver(X, Y, Fx_init, Fy_init, np.sqrt(Fx_init**2 + Fy_init**2), cmap='plasma', scale=20)

# Initialize dots, trails, and velocity arrows
dots, trails, vel_arrows = [], [], []
cmap = cm.plasma

for obj in objects:
    dot, = ax.plot([], [], 'o', color=obj["color"], markersize=10)
    trail, = ax.plot([], [], '-', color=obj["color"], linewidth=1)
    arrow = ax.quiver(obj["pos"][0], obj["pos"][1], 0, 0, angles='xy', scale_units='xy', scale=1, color=obj["color"])
    dots.append(dot)
    trails.append(trail)
    vel_arrows.append(arrow)

# Dragging state
dragging = {"active": False, "obj_idx": None}

# Interactive mouse events
def on_press(event):
    if event.inaxes != ax:
        return
    for idx, obj in enumerate(objects):
        if np.hypot(event.xdata - obj["pos"][0], event.ydata - obj["pos"][1]) < 0.1:
            dragging["active"] = True
            dragging["obj_idx"] = idx
            break

def on_motion(event):
    if not dragging["active"] or event.inaxes != ax:
        return
    idx = dragging["obj_idx"]
    objects[idx]["pos"] = np.array([event.xdata, event.ydata])
    objects[idx]["vel"] = np.array([0.0, 0.0])
    objects[idx]["trail_x"].append(event.xdata)
    objects[idx]["trail_y"].append(event.ydata)

def on_release(event):
    dragging["active"] = False
    dragging["obj_idx"] = None

fig.canvas.mpl_connect("button_press_event", on_press)
fig.canvas.mpl_connect("motion_notify_event", on_motion)
fig.canvas.mpl_connect("button_release_event", on_release)

# Compute gravitational field (vectorized)
def compute_field_vectorized(objects):
    Fx_total = np.zeros(X.shape)
    Fy_total = np.zeros(Y.shape)
    for obj in objects:
        dx = X - obj["pos"][0]
        dy = Y - obj["pos"][1]
        r2 = dx**2 + dy**2
        r = np.sqrt(r2)
        r3 = np.where(r != 0, r2 * r, np.inf)
        Fx_total += -G * obj["mass"] * dx / r3
        Fy_total += -G * obj["mass"] * dy / r3
    F_mag = np.sqrt(Fx_total**2 + Fy_total**2)
    Fx_norm = Fx_total / (F_mag + 1e-20)
    Fy_norm = Fy_total / (F_mag + 1e-20)
    return Fx_norm, Fy_norm, F_mag

# Animation update
def update(frame):
    # Update velocities
    for i, obj_i in enumerate(objects):
        F_total = np.zeros(2)
        for j, obj_j in enumerate(objects):
            if i != j:
                r = obj_j["pos"] - obj_i["pos"]
                r_mag = np.linalg.norm(r)
                if r_mag != 0:
                    F_total += G * obj_i["mass"] * obj_j["mass"] * r / r_mag**3
        obj_i["vel"] += F_total / obj_i["mass"] * dt

    # Update positions and trails
    trail_length = 200
    for obj in objects:
        obj["pos"] += obj["vel"] * dt
        obj["trail_x"].append(obj["pos"][0])
        obj["trail_y"].append(obj["pos"][1])
        obj["trail_x"] = obj["trail_x"][-trail_length:]
        obj["trail_y"] = obj["trail_y"][-trail_length:]

    # Update gravitational field
    Fx_norm, Fy_norm, F_mag = compute_field_vectorized(objects)
    F_mag_norm = F_mag / (F_mag.max() + 1e-20)
    quiver.set_UVC(Fx_norm, Fy_norm, F_mag_norm)

    # Max speed for scaling arrows
    speeds = np.array([np.linalg.norm(obj["vel"]) for obj in objects])
    max_speed = speeds.max() if speeds.max() != 0 else 1e-10

    # Update objects and arrows
    for idx, obj in enumerate(objects):
        dots[idx].set_data([obj["pos"][0]], [obj["pos"][1]])
        trails[idx].set_data(obj["trail_x"], obj["trail_y"])
        scale_factor = 0.5 / max_speed
        vel_arrows[idx].set_offsets([obj["pos"][0], obj["pos"][1]])
        vel_arrows[idx].set_UVC(obj["vel"][0]*scale_factor, obj["vel"][1]*scale_factor)
        vel_arrows[idx].set_color(cmap(np.linalg.norm(obj["vel"]) / max_speed))

    return dots + trails + [quiver] + vel_arrows

# Run animation
ani = FuncAnimation(fig, update, frames=1000, interval=50, blit=True)
plt.colorbar(quiver, ax=ax, label="Gravitational Field Magnitude")
plt.title("Interactive Gravity Simulation with Velocity Vectors")
plt.show()
