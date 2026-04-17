import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Tuple, List, Optional

# ────────────────────────────────────────────────
# Configuration / Parameters
# ────────────────────────────────────────────────
@dataclass
class Params:
    dt: float = 0.1          # [s]
    T_total: float = 40.0    # total simulation time [s]
    v_nom: float = 1.2       # nominal forward speed [m/s]
    yaw_rate: float = 0.04   # constant yaw rate [rad/s]  → circular-ish path
    sigma_range: float = 0.35  # std dev of range measurement [m]
    sigma_process_xy: float = 0.06
    sigma_process_yaw: float = 0.008

    # For nicer visualization
    plot_traj: bool = True
    plot_cov: bool = True
    plot_every: int = 5


# ────────────────────────────────────────────────
# 1. Ocean current model (smooth & time-varying)
# ────────────────────────────────────────────────
def ocean_current(t: float) -> np.ndarray:
    """Returns [cx, cy] in m/s — world frame"""
    # Combination of different frequencies + slow trend
    cx = 0.35 * np.sin(0.008 * t) + 0.12 * np.sin(0.025 * t + 1.1)
    cy = 0.28 * np.cos(0.011 * t) + 0.09 * np.cos(0.038 * t + 0.7)
    return np.array([cx, cy])


# ────────────────────────────────────────────────
# 2. Measurement model — range-only to multiple beacons
# ────────────────────────────────────────────────
def simulate_ranges(true_state: np.ndarray, landmarks: List[Tuple[float, float]],
                    sigma: float = 0.3) -> np.ndarray:
    """
    true_state : [x, y, yaw]
    Returns noisy range measurements (one per landmark)
    """
    x, y = true_state[:2]
    ranges = []
    for lx, ly in landmarks:
        r_true = np.hypot(lx - x, ly - y)
        r_noisy = r_true + np.random.normal(0, sigma)
        ranges.append(max(r_noisy, 0.01))   # avoid numerical issues
    return np.array(ranges)


def predicted_ranges(state: np.ndarray, landmarks: List[Tuple[float, float]]) -> np.ndarray:
    """Noise-free predicted ranges (used in EKF)"""
    x, y = state[:2]
    return np.array([np.hypot(lx - x, ly - y) for lx, ly in landmarks])


# ────────────────────────────────────────────────
# 3. EKF core functions
# ────────────────────────────────────────────────
def ekf_predict(x: np.ndarray, P: np.ndarray, u: Tuple[float, float, np.ndarray],
                dt: float, Q: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    u = (v_forward, yaw_rate, current_xy)
    Motion model:  v_world = v * heading + current
    """
    v, ω, current = u
    θ = x[2]

    # State prediction (simple Euler integration)
    x_new = x.copy()
    x_new[0] += (v * np.cos(θ) + current[0]) * dt
    x_new[1] += (v * np.sin(θ) + current[1]) * dt
    x_new[2] += ω * dt
    x_new[2] = np.arctan2(np.sin(x_new[2]), np.cos(x_new[2]))  # wrap to [-π, π]

    # Jacobian F (wrt state — control & current treated as known)
    F = np.eye(3)
    F[0, 2] = -v * np.sin(θ) * dt
    F[1, 2] =  v * np.cos(θ) * dt

    P_new = F @ P @ F.T + Q
    return x_new, P_new


def ekf_update(x: np.ndarray, P: np.ndarray, z: np.ndarray,
               landmarks: List[Tuple[float, float]], R: np.ndarray
               ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Range-only update — multiple independent range measurements
    """
    z_pred = predicted_ranges(x, landmarks)

    # Measurement Jacobian H (∂range / ∂state)
    H_rows = []
    for lx, ly in landmarks:
        dx, dy = lx - x[0], ly - x[1]
        r = np.hypot(dx, dy) + 1e-9   # avoid division by zero
        H_rows.append([-dx/r, -dy/r, 0.0])

    H = np.array(H_rows)           # shape (n_landmarks, 3)
    innovation = z - z_pred        # shape (n_landmarks,)

    S = H @ P @ H.T + R
    K = P @ H.T @ np.linalg.inv(S)   # can be numerically unstable for ill-conditioned S

    x_upd = x + K @ innovation
    x_upd[2] = np.arctan2(np.sin(x_upd[2]), np.cos(x_upd[2]))

    # Joseph form — more stable when K is large
    I_KH = np.eye(3) - K @ H
    P_upd = I_KH @ P @ I_KH.T + K @ R @ K.T

    return x_upd, P_upd


# ────────────────────────────────────────────────
# Main simulation & visualization
# ────────────────────────────────────────────────
def run_simulation(params: Params):
    # ── Setup ─────────────────────────────────────
    landmarks = [
        (0, 40),
        (50, 10),
        (35, -25),
        (-10, -15),
        (60, 35)
    ]

    n_beacons = len(landmarks)

    # Noise covariances
    Q = np.diag([params.sigma_process_xy**2,
                 params.sigma_process_xy**2,
                 params.sigma_process_yaw**2])

    R = np.eye(n_beacons) * params.sigma_range**2

    # Initial conditions
    x_true = np.array([0.0, 0.0, np.deg2rad(45.0)])
    x_est  = x_true.copy() + np.random.normal(0, [0.8, 0.8, 0.15])
    P = np.diag([4.0, 4.0, 0.25])

    # Logging
    times     = []
    true_traj = []
    est_traj  = []
    est_3sigma = []   # for uncertainty ellipse visualization

    # ── Simulation loop ───────────────────────────
    for t in np.arange(0.0, params.T_total + params.dt/2, params.dt):
        current = ocean_current(t)

        # ── Ground truth motion ───────────────────
        v_body = params.v_nom
        x_true[0] += (v_body * np.cos(x_true[2]) + current[0]) * params.dt
        x_true[1] += (v_body * np.sin(x_true[2]) + current[1]) * params.dt
        x_true[2] += params.yaw_rate * params.dt
        x_true[2] = np.arctan2(np.sin(x_true[2]), np.cos(x_true[2]))

        # ── Measurement ───────────────────────────
        z = simulate_ranges(x_true, landmarks, params.sigma_range)

        # ── EKF ───────────────────────────────────
        x_est, P = ekf_predict(x_est, P,
                               u=(v_body, params.yaw_rate, current),
                               dt=params.dt, Q=Q)

        x_est, P = ekf_update(x_est, P, z, landmarks, R)

        # ── Logging ───────────────────────────────
        if int(t / params.dt) % params.plot_every == 0:
            times.append(t)
            true_traj.append(x_true[:2].copy())
            est_traj.append(x_est[:2].copy())
            est_3sigma.append(3 * np.sqrt(np.diag(P)[:2]))

    # ── Visualization ─────────────────────────────
    if not params.plot_traj:
        return

    true_traj = np.array(true_traj)
    est_traj  = np.array(est_traj)
    est_3sigma = np.array(est_3sigma)

    plt.figure(figsize=(11, 9))
    ax = plt.gca()

    # Landmarks
    lx, ly = zip(*landmarks)
    ax.plot(lx, ly, 'r^', ms=12, label='Landmarks / Beacons')

    # True & estimated trajectory
    ax.plot(true_traj[:,0], true_traj[:,1], 'C0-', lw=2.2, label='True path')
    ax.plot(est_traj[:,0],  est_traj[:,1],  'C1--', lw=1.8, label='EKF estimate')

    # Uncertainty ellipses (simple axis-aligned for clarity)
    if params.plot_cov:
        for pos, sig in zip(est_traj[::max(1, len(est_traj)//40)], est_3sigma[::max(1, len(est_traj)//40)]):
            rect = plt.Rectangle(pos - sig, 2*sig[0], 2*sig[1],
                                 fc='orange', ec='none', alpha=0.13, zorder=1)
            ax.add_patch(rect)

    ax.set_aspect('equal')
    ax.grid(alpha=0.3, ls=':')
    ax.legend(loc='upper left', fontsize=10)
    ax.set_title("Range-only EKF localization with time-varying ocean current")
    ax.set_xlabel("East [m]"), ax.set_ylabel("North [m]")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    cfg = Params()
    # Feel free to tune:
    # cfg.T_total = 60
    # cfg.yaw_rate = 0.07
    # cfg.sigma_range = 0.5
    # cfg.plot_every = 3

    run_simulation(cfg)