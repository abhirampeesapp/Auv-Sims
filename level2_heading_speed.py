import numpy as np
import matplotlib.pyplot as plt


class AUV2D:
    """
    2D AUV kinematic model (unicycle-style)
    State: [x, y, heading]
    """

    def __init__(self, position=(0.0, 0.0), heading=0.0, speed=1.0):
        self.position = np.array(position, dtype=float)
        self.heading = heading                  # radians
        self.speed = speed                      # m/s
        self.history = [self.position.copy()]

    def step(self, yaw_rate, dt, current=(0.0, 0.0), noise_std=0.0):
        """
        Propagate AUV state by one time step.

        yaw_rate : heading rate (rad/s)
        dt       : time step (s)
        current  : water current velocity (vx, vy)
        noise_std: Gaussian noise std (m)
        """

        # Update heading
        self.heading += yaw_rate * dt
        self.heading = np.arctan2(np.sin(self.heading), np.cos(self.heading))  # wrap

        # Body-frame velocity
        v_body = np.array([
            self.speed * np.cos(self.heading),
            self.speed * np.sin(self.heading)
        ])

        # Add current disturbance
        v_total = v_body + np.array(current)

        # Update position
        self.position += v_total * dt

        # Add measurement/process noise
        if noise_std > 0:
            self.position += np.random.normal(0, noise_std, size=2)

        self.history.append(self.position.copy())


def simulate_auv(duration=60.0, dt=0.1):
    auv = AUV2D(position=(0, 0), heading=0.0, speed=1.5)

    time = np.arange(0, duration, dt)

    for t in time:
        # Sinusoidal yaw-rate command (smooth turns)
        yaw_rate = 0.15 * np.sin(0.05 * t)

        # Slowly varying ocean current
        current = (
            0.2 * np.sin(0.01 * t),
            0.1 * np.cos(0.01 * t)
        )

        auv.step(
            yaw_rate=yaw_rate,
            dt=dt,
            current=current,
            noise_std=0.01
        )

    plot_trajectory(auv.history)


def plot_trajectory(history):
    history = np.array(history)

    plt.figure(figsize=(8, 6))
    plt.plot(history[:, 0], history[:, 1], linewidth=2, label="AUV Trajectory")
    plt.scatter(history[0, 0], history[0, 1], c='green', s=80, label="Start")
    plt.scatter(history[-1, 0], history[-1, 1], c='red', s=80, label="End")

    plt.xlabel("X (m)")
    plt.ylabel("Y (m)")
    plt.title("2D AUV Motion with Currents and Noise")
    plt.axis("equal")
    plt.grid(True)
    plt.legend()
    plt.show()


if __name__ == "__main__":
    simulate_auv()
