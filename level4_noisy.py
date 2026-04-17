import numpy as np
import matplotlib.pyplot as plt

# ==================================================
# BASE AUV CLASS (NO NOISE)
# ==================================================
class AUV3D:
    def __init__(self,
                 initial_position=(0, 0, 0),
                 initial_yaw=0.0,
                 initial_pitch=0.0,
                 speed=1.0):

        self.position = np.array(initial_position, dtype=float)
        self.yaw = initial_yaw
        self.pitch = initial_pitch
        self.speed = speed

        self.history = [self.position.copy()]

    def update(self, delta_yaw, delta_pitch, dt):
        # Update orientation
        self.yaw += delta_yaw
        self.pitch += delta_pitch

        # Direction vector (3D)
        dx = self.speed * np.cos(self.pitch) * np.cos(self.yaw)
        dy = self.speed * np.cos(self.pitch) * np.sin(self.yaw)
        dz = self.speed * np.sin(self.pitch)

        # Update position
        self.position += np.array([dx, dy, dz]) * dt
        self.history.append(self.position.copy())


# ==================================================
# AUV WITH SENSOR NOISE
# ==================================================
class AUV3DNoisy(AUV3D):
    def __init__(self, *args,
                 position_noise_std=0.5,
                 depth_noise_std=0.2,
                 **kwargs):

        super().__init__(*args, **kwargs)
        self.position_noise_std = position_noise_std
        self.depth_noise_std = depth_noise_std
        self.noisy_history = [self.get_noisy_measurement()]

    def get_noisy_measurement(self):
        noise_xy = np.random.normal(0, self.position_noise_std, 2)
        noise_z = np.random.normal(0, self.depth_noise_std)
        return self.position + np.array([noise_xy[0], noise_xy[1], noise_z])

    def update(self, *args, **kwargs):
        super().update(*args, **kwargs)
        self.noisy_history.append(self.get_noisy_measurement())


# ==================================================
# SIMULATION FUNCTION
# ==================================================
def simulate_auv_noisy(duration=60, dt=0.1, target_depth=-10):

    auv = AUV3DNoisy(
        initial_position=(0, 0, 0),
        initial_yaw=0.0,
        initial_pitch=0.0,
        speed=1.5,
        position_noise_std=0.5,
        depth_noise_std=0.2
    )

    times = np.arange(0, duration, dt)

    for t in times:
        # Simple depth controller (IDEAL – uses true depth)
        depth_error = target_depth - auv.position[2]
        delta_pitch = 0.05 * depth_error

        # Small yaw oscillation
        delta_yaw = 0.01 * np.sin(0.1 * t)

        auv.update(delta_yaw, delta_pitch, dt)

    # ==================================================
    # PLOTTING
    # ==================================================
    true_history = np.array(auv.history)
    noisy_history = np.array(auv.noisy_history)

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    ax.plot(true_history[:, 0],
            true_history[:, 1],
            true_history[:, 2],
            label="True Path",
            color="blue")

    ax.plot(noisy_history[:, 0],
            noisy_history[:, 1],
            noisy_history[:, 2],
            label="Noisy Measurements",
            color="red",
            alpha=0.5)

    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Depth (m)")
    ax.set_title("Level 4: 3D AUV with Sensor Noise")
    ax.legend()

    ax.invert_zaxis()
    plt.show()


# ==================================================
# RUN
# ==================================================
if __name__ == "__main__":
    simulate_auv_noisy()
