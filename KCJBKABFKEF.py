import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import List, Tuple

# ────────────────────────────────────────────────
# Configuration / Parameters
# ────────────────────────────────────────────────
@dataclass
class SonarParams:
    sound_speed: float = 1500.0  # m/s in water
    max_range: float = 150.0     # Maximum detection range [m]
    range_resolution: float = 0.2  # Range bin size [m]
    vehicle_speed: float = 1.5   # AUV speed along track [m/s]
    ping_rate: float = 5.0       # Pings per second
    beam_width_deg: float = 60.0  # Approximate beam width (for simple gating)
    noise_level: float = 0.02    # Gaussian noise std dev
    attenuation_factor: float = 0.01  # dB/m (simple absorption)
    n_pings: int = 300           # Number of pings to simulate
    log_compress: bool = True    # Apply log compression for display


# ────────────────────────────────────────────────
# Simple point reflector targets
# ────────────────────────────────────────────────
@dataclass
class Target:
    x: float  # Along-track position [m]
    y: float  # Cross-track position [m] (positive port, negative starboard)
    strength: float = 1.0  # Relative reflectivity


# ────────────────────────────────────────────────
# Sonar raw signal simulation
# ────────────────────────────────────────────────
def simulate_sonar_ping(veh_x: float, veh_y: float, targets: List[Target],
                        params: SonarParams) -> np.ndarray:
    """
    Simulate raw echo intensity vs range for one ping.
    - Simple model: Point reflectors with 1/r^2 spreading + absorption
    - Gated by crude beam (ignore if |angle| > beam_width/2)
    - Returns intensity array (one side, e.g., port/starboard combined in one axis)
    """
    n_bins = int(params.max_range / params.range_resolution)
    signal = np.zeros(n_bins)
    beam_half_rad = np.deg2rad(params.beam_width_deg / 2)

    for tgt in targets:
        dx = tgt.x - veh_x
        dy = tgt.y - veh_y
        r = np.sqrt(dx**2 + dy**2)
        if r > params.max_range or r < 0.1:
            continue

        # Simple beam gating: angle from heading (assume heading along x)
        angle = np.arctan2(dy, dx)
        if abs(angle) > beam_half_rad:
            continue  # Outside beam

        # Intensity: target strength / (spreading + absorption)
        absorption = np.exp(-params.attenuation_factor * r * 2)  # Round-trip
        intensity = tgt.strength * absorption / (r**2 + 1e-6)

        # Add to nearest bin (simple, no pulse length simulation)
        bin_idx = int(r / params.range_resolution)
        if bin_idx < n_bins:
            signal[bin_idx] += intensity

    # Add background noise (reverberation + thermal)
    signal += np.abs(np.random.normal(0, params.noise_level, n_bins))

    return signal


# ────────────────────────────────────────────────
# Main simulation: Form raw sonar image
# ────────────────────────────────────────────────
def run_sonar_imaging_simulation(params: SonarParams):
    # ── Setup ─────────────────────────────────────
    dt = 1.0 / params.ping_rate  # Time between pings [s]
    n_bins = int(params.max_range / params.range_resolution)

    # Define scene: targets (point reflectors, e.g., rocks, wrecks)
    targets = [
        Target(50, 40, 1.5),   # Strong reflector port side
        Target(100, -60, 1.0), # Weaker starboard
        Target(150, 30, 2.0),
        Target(200, -20, 0.8),
        Target(250, 50, 1.2),
    ]

    # Add simple seabed: weak reflectors along y=0 (but in side-scan, nadir is blank)
    # For simplicity, skip detailed seabed; focus on highlights

    # Raw image: rows = pings (along-track), cols = range bins (cross-track)
    # For side-scan, typically separate port/starboard, but here combined (y>0 port, y<0 starb as negative range?)
    # Simplify: Use one image per side, or abs(y) as range, but for demo: assume one side, range = |slant range|
    raw_image = np.zeros((params.n_pings, n_bins))

    # Vehicle path: straight line along x, y=0 (above flat seabed)
    veh_y = 0.0
    for ping in range(params.n_pings):
        veh_x = ping * params.vehicle_speed * dt

        # Simulate ping
        signal = simulate_sonar_ping(veh_x, veh_y, targets, params)

        raw_image[ping] = signal

    # ── Post-processing for display ───────────────
    # Raw is amplitude; often display as intensity (squared) or log-compressed
    display_image = raw_image ** 2  # Intensity
    if params.log_compress:
        display_image = np.log(1 + display_image)  # Log compress for better viz
        display_image = (display_image - display_image.min()) / (display_image.max() - display_image.min() + 1e-6)

    # ── Visualization ─────────────────────────────
    extent = [0, params.max_range, 0, params.n_pings * params.vehicle_speed * dt]
    plt.figure(figsize=(12, 8))
    plt.imshow(display_image, aspect='auto', cmap='gray', origin='lower', extent=extent)
    plt.title("Simulated Raw Side-Scan Sonar Image")
    plt.xlabel("Range (cross-track) [m]")
    plt.ylabel("Along-track distance [m]")
    plt.colorbar(label='Normalized Intensity (log-compressed)')
    plt.grid(alpha=0.2)
    plt.tight_layout()
    plt.show()

    # Optional: Save raw data
    # np.save('raw_sonar_image.npy', raw_image)


if __name__ == "__main__":
    cfg = SonarParams()
    # Tune as needed:
    # cfg.max_range = 200
    # cfg.vehicle_speed = 2.0
    # cfg.ping_rate = 10.0
    # cfg.log_compress = False

    run_sonar_imaging_simulation(cfg)