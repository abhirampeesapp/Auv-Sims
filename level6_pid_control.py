import numpy as np
import matplotlib.pyplot as plt

# ==================================
# PID CONTROLLER (CLEAN & STABLE)
# ==================================
class PID:
    def __init__(self, kp, ki, kd, limit=5.0):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.limit = limit
        self.integral = 0.0
        self.prev_error = 0.0

    def compute(self, error, dt):
        self.integral += error * dt
        derivative = (error - self.prev_error) / dt

        u = self.kp * error + self.ki * self.integral + self.kd * derivative
        u = np.clip(u, -self.limit, self.limit)

        self.prev_error = error
        return u


# ==================================
# 6-DOF AUV MODEL (BASELINE)
# ==================================
class AUV6DOF:
    """
    State vector:
    [u, v, w, p, q, r, x, y, z, phi, theta, psi]
    """

    def __init__(self):
        self.state = np.zeros(12)
        self.state[6:9] = [0, 0, 0]  # position
        self.state[0] = 1.2          # forward speed

        self.history = []

        # Parameters
        self.m = 150.0
        self.I = np.diag([20, 30, 30])
        self.drag_lin = np.diag([8, 12, 15])
        self.drag_ang = np.diag([5, 6, 6])
        self.g = 9.81
        self.W = self.m * self.g
        self.B = self.W * 1.02   # slightly buoyant

    def rotation_matrix(self, phi, theta, psi):
        cφ, sφ = np.cos(phi), np.sin(phi)
        cθ, sθ = np.cos(theta), np.sin(theta)
        cψ, sψ = np.cos(psi), np.sin(psi)

        return np.array([
            [cψ*cθ, cψ*sθ*sφ - sψ*cφ, cψ*sθ*cφ + sψ*sφ],
            [sψ*cθ, sψ*sθ*sφ + cψ*cφ, sψ*sθ*cφ - cψ*sφ],
            [-sθ,    cθ*sφ,            cθ*cφ]
        ])

    def dynamics(self, x, u_ctrl):
        u, v, w, p, q, r, X, Y, Z, φ, θ, ψ = x
        thrust, elevator, rudder = u_ctrl

        # Forces
        F = np.array([
            thrust - self.drag_lin[0,0]*u,
            -self.drag_lin[1,1]*v,
            (self.B - self.W) - self.drag_lin[2,2]*w + elevator
        ])

        # Moments
        M = np.array([
            -self.drag_ang[0,0]*p,
            elevator - self.drag_ang[1,1]*q,
            rudder - self.drag_ang[2,2]*r
        ])

        # Accelerations
        lin_acc = F / self.m
        ang_acc = np.linalg.inv(self.I) @ M

        # Kinematics
        R = self.rotation_matrix(φ, θ, ψ)
        pos_dot = R @ np.array([u, v, w])

        euler_dot = np.array([
            p + q*np.sin(φ)*np.tan(θ) + r*np.cos(φ)*np.tan(θ),
            q*np.cos(φ) - r*np.sin(φ),
            q*np.sin(φ)/np.cos(θ) + r*np.cos(φ)/np.cos(θ)
        ])

        return np.hstack((lin_acc, ang_acc, pos_dot, euler_dot))

    def step(self, u_ctrl, dt):
        k1 = self.dynamics(self.state, u_ctrl)
        k2 = self.dynamics(self.state + 0.5*dt*k1, u_ctrl)
        k3 = self.dynamics(self.state + 0.5*dt*k2, u_ctrl)
        k4 = self.dynamics(self.state + dt*k3, u_ctrl)

        self.state += (dt/6)*(k1 + 2*k2 + 2*k3 + k4)
        self.history.append(self.state.copy())


# ==================================
# SIMULATION
# ==================================
def simulate():
    auv = AUV6DOF()
    pid_depth = PID(3.0, 0.1, 1.5)

    dt = 0.05
    T = 40
    z_ref = -10

    for _ in np.arange(0, T, dt):
        depth_error = z_ref - auv.state[8]
        elevator = pid_depth.compute(depth_error, dt)

        control = [15.0, elevator, 0.0]  # thrust, elevator, rudder
        auv.step(control, dt)

    history = np.array(auv.history)
    plt.plot(history[:, 6], history[:, 8])
    plt.gca().invert_yaxis()
    plt.xlabel("X (m)")
    plt.ylabel("Depth (m)")
    plt.title("6-DOF AUV Depth Control")
    plt.grid()
    plt.show()


if __name__ == "__main__":
    simulate()
