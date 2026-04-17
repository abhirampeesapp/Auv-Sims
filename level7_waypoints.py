import numpy as np
import matplotlib.pyplot as plt


# =========================
# PID CONTROLLER (with basic anti-windup)
# =========================
class PID:
    def __init__(self, kp, ki, kd, limit=1.0, integral_limit=5.0):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.limit = limit
        self.integral_limit = integral_limit      # prevents windup
        self.integral = 0.0
        self.prev_error = 0.0

    def compute(self, error, dt):
        # Integral with basic clamping (anti-windup)
        self.integral += error * dt
        self.integral = np.clip(self.integral, -self.integral_limit, self.integral_limit)

        derivative = (error - self.prev_error) / dt if dt > 0 else 0.0
        u = self.kp * error + self.ki * self.integral + self.kd * derivative
        u_clipped = np.clip(u, -self.limit, self.limit)

        # If output is saturated → don't increase integral further (simple anti-windup)
        if u_clipped != u and np.sign(error) == np.sign(u):
            self.integral -= error * dt   # back-calculate

        self.prev_error = error
        return u_clipped


# =========================
# AUV KINEMATIC MODEL
# =========================
class AUV:
    def __init__(self):
        self.x = 0.0
        self.y = 0.0
        self.z = 0.0
        self.yaw = 0.0
        self.speed = 1.2          # constant forward speed (m/s)
        self.history = []

    def step(self, yaw_rate_cmd, vertical_speed_cmd, dt):
        # Update heading
        self.yaw += yaw_rate_cmd * dt
        self.yaw = np.arctan2(np.sin(self.yaw), np.cos(self.yaw))  # normalize to [-π, π]

        # Update position (simple unicycle model)
        self.x += self.speed * np.cos(self.yaw) * dt
        self.y += self.speed * np.sin(self.yaw) * dt
        self.z += vertical_speed_cmd * dt

        # Save state
        self.history.append([self.x, self.y, self.z, self.yaw])


# =========================
# LINE-OF-SIGHT GUIDANCE with adaptive lookahead
# =========================
def los_guidance(current_pos, wp_prev, wp_next, lookahead_base=5.0):
    """
    Adaptive LOS: lookahead grows with larger cross-track error
    Returns desired yaw angle
    """
    x, y = current_pos
    x1, y1 = wp_prev
    x2, y2 = wp_next

    # Path segment direction
    dx = x2 - x1
    dy = y2 - y1
    path_angle = np.arctan2(dy, dx)

    # Cross-track error (positive = starboard side)
    cross_track_error = (
        -(x - x1) * np.sin(path_angle) +
         (y - y1) * np.cos(path_angle)
    )

    # Adaptive lookahead: larger error → larger lookahead (smoother), but bounded
    lookahead = max(3.0, min(15.0, lookahead_base + 6.0 * abs(cross_track_error)))

    # LOS angle
    desired_yaw = path_angle - np.arctan2(cross_track_error, lookahead)

    return desired_yaw, cross_track_error, lookahead


# =========================
# MAIN SIMULATION
# =========================
def simulate():
    dt = 0.05
    T = 180.0               # longer time in case path is slow
    acceptance_radius = 2.5  # improved from 1.0

    waypoints = [
        (0.0,   0.0,   0.0),
        (10.0,  0.0,  -5.0),
        (20.0, 10.0,  -8.0),
        (30.0,  0.0,  -6.0),
        (40.0, -10.0, -10.0)
    ]

    auv = AUV()

    # Tuned controllers
    yaw_pid   = PID(kp=2.5, ki=0.12, kd=0.7,  limit=1.3, integral_limit=4.0)
    depth_pid = PID(kp=1.3, ki=0.08, kd=0.35, limit=0.6, integral_limit=3.0)

    wp_idx = 1  # start heading toward second waypoint

    while auv.history[-1][3] if auv.history else 0 < T:  # rough time check
        if wp_idx >= len(waypoints):
            break

        wp_prev = waypoints[wp_idx - 1][:2]
        wp_next = waypoints[wp_idx][:2]
        target_depth = waypoints[wp_idx][2]

        # Distance to next waypoint (horizontal only)
        dx = wp_next[0] - auv.x
        dy = wp_next[1] - auv.y
        dist_to_wp = np.hypot(dx, dy)

        # Waypoint reached → advance
        if dist_to_wp < acceptance_radius:
            wp_idx += 1
            if wp_idx >= len(waypoints):
                break
            # Update to next segment
            wp_prev = waypoints[wp_idx - 1][:2]
            wp_next = waypoints[wp_idx][:2]
            target_depth = waypoints[wp_idx][2]

        # LOS guidance
        desired_yaw, cte, current_lookahead = los_guidance(
            (auv.x, auv.y), wp_prev, wp_next, lookahead_base=5.0
        )

        # Heading error (shortest angle)
        yaw_error = np.arctan2(np.sin(desired_yaw - auv.yaw),
                               np.cos(desired_yaw - auv.yaw))

        # Depth control
        depth_error = target_depth - auv.z

        # Compute controls
        yaw_rate_cmd   = yaw_pid.compute(yaw_error, dt)
        vertical_speed_cmd = depth_pid.compute(depth_error, dt)

        # Update vehicle
        auv.step(yaw_rate_cmd, vertical_speed_cmd, dt)

    plot_results(auv.history, waypoints)


# =========================
# PLOTTING
# =========================
def plot_results(history, waypoints):
    history = np.array(history)
    wps = np.array(waypoints)

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Trajectory
    ax.plot(history[:, 0], history[:, 1], history[:, 2],
            label="AUV path", color='royalblue', linewidth=2.2, zorder=3)

    # Waypoints
    ax.scatter(wps[:, 0], wps[:, 1], wps[:, 2],
               color='red', s=80, marker='*', label="Waypoints", zorder=5)

    # Start & end markers
    ax.scatter([history[0,0]], [history[0,1]], [history[0,2]],
               color='green', s=120, marker='o', label="Start", zorder=10)
    ax.scatter([history[-1,0]], [history[-1,1]], [history[-1,2]],
               color='darkorange', s=120, marker='^', label="End", zorder=10)

    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Depth (m)")
    ax.invert_zaxis()           # depth increases downward
    ax.legend(loc='upper left', bbox_to_anchor=(1.05, 1))
    ax.set_title("Improved AUV Waypoint Following\n(Adaptive LOS + PID with anti-windup)")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    simulate()