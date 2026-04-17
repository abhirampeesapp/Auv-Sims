import numpy as np
import matplotlib.pyplot as plt

# =========================
# PID CONTROLLER
# =========================
class PID:
    def __init__(self, kp, ki, kd, limit=1.0):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.integral = 0.0
        self.prev_error = 0.0
        self.limit = limit

    def compute(self, error, dt):
        self.integral += error * dt
        derivative = (error - self.prev_error) / dt
        output = self.kp * error + self.ki * self.integral + self.kd * derivative
        self.prev_error = error
        return np.clip(output, -self.limit, self.limit)


# =========================
# OCEAN CURRENT MODEL
# =========================
def ocean_current(x, y, z, t):
    """
    Returns ocean current velocity [vx, vy, vz]
    """
    vx = 0.3 * np.sin(0.1 * y) + 0.1 * np.cos(0.05 * t)
    vy = 0.2 * np.cos(0.1 * x)
    vz = 0.05 * np.sin(0.2 * t)

    # Small random turbulence
    noise = np.random.normal(0, 0.02, 3)
    return np.array([vx, vy, vz]) + noise


# =========================
# AUV MODEL (WITH DISTURBANCE)
# =========================
class AUV:
    def __init__(self):
        self.x = 0.0
        self.y = 0.0
        self.z = 0.0
        self.yaw = 0.0
        self.speed = 1.2
        self.history = []

    def step(self, yaw_rate, vertical_speed, current, dt):
        self.yaw += yaw_rate * dt

        vx = self.speed * np.cos(self.yaw) + current[0]
        vy = self.speed * np.sin(self.yaw) + current[1]
        vz = vertical_speed + current[2]

        self.x += vx * dt
        self.y += vy * dt
        self.z += vz * dt

        self.history.append([self.x, self.y, self.z])


# =========================
# SIMULATION
# =========================
def simulate():
    dt = 0.05
    total_time = 150

    waypoints = [
        (10, 0, -5),
        (25, 10, -8),
        (40, -5, -10),
        (55, 0, -6)
    ]

    auv = AUV()
    yaw_pid = PID(2.5, 0.0, 0.5)
    depth_pid = PID(1.5, 0.0, 0.4)

    wp_index = 0

    for t in np.arange(0, total_time, dt):
        if wp_index >= len(waypoints):
            break

        wx, wy, wz = waypoints[wp_index]

        dx = wx - auv.x
        dy = wy - auv.y
        dist = np.sqrt(dx**2 + dy**2)

        if dist < 1.2:
            wp_index += 1
            continue

        desired_yaw = np.arctan2(dy, dx)
        yaw_error = np.arctan2(
            np.sin(desired_yaw - auv.yaw),
            np.cos(desired_yaw - auv.yaw)
        )

        depth_error = wz - auv.z

        yaw_rate = yaw_pid.compute(yaw_error, dt)
        vertical_speed = depth_pid.compute(depth_error, dt)

        current = ocean_current(auv.x, auv.y, auv.z, t)
        auv.step(yaw_rate, vertical_speed, current, dt)

    plot(auv.history, waypoints)


# =========================
# VISUALIZATION
# =========================
def plot(history, waypoints):
    history = np.array(history)
    wps = np.array(waypoints)

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')

    ax.plot(history[:,0], history[:,1], history[:,2], label="AUV Path")
    ax.scatter(wps[:,0], wps[:,1], wps[:,2], c='red', s=60, label="Waypoints")

    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Depth (m)")
    ax.invert_zaxis()
    ax.set_title("Level 8: Waypoint Navigation with Ocean Currents")
    ax.legend()

    plt.show()


# =========================
if __name__ == "__main__":
    simulate()
