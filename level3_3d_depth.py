import numpy as np
import matplotlib.pyplot as plt


# =========================
# PID CONTROLLER (ANTI-WINDUP)
# =========================
class PID:
    def __init__(self, kp, ki, kd, limit=1.0):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.limit = limit
        self.integral = 0.0
        self.prev_error = 0.0

    def compute(self, error, dt):
        derivative = (error - self.prev_error) / dt
        self.integral += error * dt

        output = (
            self.kp * error +
            self.ki * self.integral +
            self.kd * derivative
        )

        # Anti-windup
        output = np.clip(output, -self.limit, self.limit)
        if abs(output) == self.limit:
            self.integral -= error * dt

        self.prev_error = error
        return output


# =========================
# SIMPLIFIED AUV DYNAMICS
# =========================
class AUV:
    """
    State vector:
    [x, z, pitch, u, w]
    """

    def __init__(self):
        self.state = np.array([
            0.0,   # x position
            0.0,   # depth (z)
            0.0,   # pitch (rad)
            1.5,   # surge velocity u (m/s)
            0.0    # heave velocity w (m/s)
        ])
        self.history = [self.state.copy()]

    def dynamics(self, state, elevator):
        x, z, pitch, u, w = state

        # Hydrodynamic parameters (simplified)
        drag_u = 0.15
        drag_w = 0.25
        pitch_damping = 0.4

        # Forces & moments
        du = -drag_u * u
        dw = elevator - drag_w * w
        dpitch = elevator - pitch_damping * pitch

        # Kinematics
        dx = u * np.cos(pitch) - w * np.sin(pitch)
        dz = u * np.sin(pitch) + w * np.cos(pitch)

        return np.array([dx, dz, dpitch, du, dw])

    def step(self, elevator, dt):
        # RK4 Integration
        k1 = self.dynamics(self.state, elevator)
        k2 = self.dynamics(self.state + 0.5 * dt * k1, elevator)
        k3 = self.dynamics(self.state + 0.5 * dt * k2, elevator)
        k4 = self.dynamics(self.state + dt * k3, elevator)

        self.state += (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

        # Pitch saturation (realistic)
        self.state[2] = np.clip(self.state[2], -0.6, 0.6)

        self.history.append(self.state.copy())


# =========================
# SIMULATION
# =========================
def simulate():
    dt = 0.05
    T = 60.0
    target_depth = -10.0  # meters

    auv = AUV()
    pid = PID(kp=0.9, ki=0.03, kd=0.35, limit=0.4)

    time = np.arange(0, T, dt)

    for _ in time:
        depth_error = target_depth - auv.state[1]
        elevator = pid.compute(depth_error, dt)
        auv.step(elevator, dt)

    plot_results(time, auv.history, target_depth)


# =========================
# PLOTTING
# =========================
def plot_results(time, history, target_depth):
    history = np.array(history)

    fig, ax = plt.subplots(2, 1, figsize=(10, 7), sharex=True)

    # Depth
    ax[0].plot(time, history[1:, 1])
    ax[0].axhline(target_depth, linestyle="--")
    ax[0].invert_yaxis()
    ax[0].set_ylabel("Depth (m)")
    ax[0].set_title("AUV Depth Control using PID")
    ax[0].grid()

    # Pitch
    ax[1].plot(time, history[1:, 2])
    ax[1].set_ylabel("Pitch (rad)")
    ax[1].set_xlabel("Time (s)")
    ax[1].grid()

    plt.show()


# =========================
if __name__ == "__main__":
    simulate()
