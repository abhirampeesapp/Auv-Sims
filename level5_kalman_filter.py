import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class AUV3DEKF:
    def __init__(self, initial_position=(0, 0, 0), initial_velocity=(0, 0, 0), initial_yaw=0, initial_pitch=0,
                 process_noise_std=0.1, measurement_noise_std=0.5):
        """
        AUV with EKF for state estimation.
        State: [x, y, z, vx, vy, vz, yaw, pitch]
        """
        self.state = np.array([*initial_position, *initial_velocity, initial_yaw, initial_pitch], dtype=float)
        self.P = np.eye(8) * 1.0  # Covariance
        self.Q = np.eye(8) * process_noise_std**2  # Process noise
        self.R = np.eye(3) * measurement_noise_std**2  # Measurement noise (only position measured)
        self.history_true = [self.state[:3].copy()]
        self.history_est = [self.state[:3].copy()]
        self.history_noisy = []

    def predict(self, u, dt):
        """
        Predict step: u = [speed, delta_yaw, delta_pitch]
        """
        speed, delta_yaw, delta_pitch = u
        yaw = self.state[6] + delta_yaw
        pitch = self.state[7] + delta_pitch
        
        # Update velocities
        self.state[3] = speed * np.cos(pitch) * np.cos(yaw)  # vx
        self.state[4] = speed * np.cos(pitch) * np.sin(yaw)  # vy
        self.state[5] = -speed * np.sin(pitch)  # vz (depth convention)
        
        # Update positions
        self.state[:3] += self.state[3:6] * dt
        
        # Update angles
        self.state[6] = yaw
        self.state[7] = pitch
        
        # Jacobian F (approximate for EKF)
        F = np.eye(8)
        F[:3, 3:6] = dt * np.eye(3)
        # Nonlinear parts approximated as identity for simplicity
        
        self.P = F @ self.P @ F.T + self.Q

    def update(self, measurement):
        """
        Correction step: measurement = [x, y, z] noisy
        """
        H = np.zeros((3, 8))
        H[:, :3] = np.eye(3)  # Observe positions only
        
        y = measurement - H @ self.state
        S = H @ self.P @ H.T + self.R
        K = self.P @ H.T @ np.linalg.inv(S)
        
        self.state += K @ y
        self.P = (np.eye(8) - K @ H) @ self.P

    def get_noisy_measurement(self, noise_std=0.5):
        return self.state[:3] + np.random.normal(0, noise_std, 3)

def simulate_auv_ekf(duration=60, dt=0.1, speed=1.5, target_depth=-10):
    auv = AUV3DEKF(initial_position=(0, 0, 0), initial_velocity=(speed, 0, 0))
    times = np.arange(0, duration, dt)
    for t in times:
        # Control inputs
        depth_error = target_depth - auv.state[2]  # Use estimated depth
        delta_pitch = 0.05 * depth_error
        delta_yaw = 0.01 * np.sin(0.1 * t)
        u = [speed, delta_yaw, delta_pitch]
        
        auv.predict(u, dt)
        
        # Simulate measurement
        noisy_meas = auv.get_noisy_measurement()
        auv.update(noisy_meas)
        
        auv.history_true.append(auv.state[:3].copy())  # True is estimated here (simplified)
        auv.history_est.append(auv.state[:3].copy())
        auv.history_noisy.append(noisy_meas)
    
    # Plot
    true_history = np.array(auv.history_true)  # In sim, true == est without separate true model
    est_history = np.array(auv.history_est)
    noisy_history = np.array(auv.history_noisy)
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(est_history[:, 0], est_history[:, 1], est_history[:, 2], label='Estimated Path', color='green')
    ax.plot(noisy_history[:, 0], noisy_history[:, 1], noisy_history[:, 2], label='Noisy Measurements', color='red', alpha=0.3)
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Depth (m)')
    ax.set_title('AUV with EKF State Estimation')
    ax.legend()
    plt.show()

# Run the simulation
if __name__ == "__main__":
    simulate_auv_ekf()