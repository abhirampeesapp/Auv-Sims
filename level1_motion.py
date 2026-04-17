import numpy as np

# -----------------------------
# Simulation parameters
# -----------------------------
dt = 0.1          # Time step (s)
num_steps = 20    # Number of iterations

# Initial state [x, y] in meters
state = np.array([0.0, 0.0], dtype=float)

# Constant velocity [vx, vy] in m/s
velocity = np.array([0.5, 0.2], dtype=float)

print("Step |    x (m) |    y (m)")
print("-----------------------------")

for step in range(1, num_steps + 1):
    state += velocity * dt
    print(f"{step:4d} | {state[0]:7.3f} | {state[1]:7.3f}")
