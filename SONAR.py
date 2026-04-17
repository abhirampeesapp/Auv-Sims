import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import List
import noise  # pip install noise (Perlin)

# ────────────────────────────────────────────────
# Configuration
# ────────────────────────────────────────────────
@dataclass
class SonarParams:
    sound_speed: float = 1500.0
    max_range: float = 100.0             # slant [m]
    range_resolution: float = 0.075
    vehicle_speed: float = 1.8
    ping_rate: float = 8.0
    horizontal_beam_width_deg: float = 0.8   # narrow along-track
    vertical_beam_width_deg: float = 50.0
    altitude: float = 8.0
    nadir_blank_m: float = 5.0
    absorption_db_per_m: float = 0.04    # ~500 kHz
    spreading_loss_exp: float = 2.0
    shadow_attenuation: float = 0.90
    n_pings: int = 900
    apply_tvg: bool = True
    apply_slant_correction: bool = True
    log_compress: bool = True
    
    pulse_length_m: float = 0.4          # → range blur
    seabed_texture_scale: float = 15.0
    multipath_attenuation: float = 0.22
    sidelobe_level: float = 0.07
    
    # Vehicle motion (demo)
    vehicle_roll_amp_deg: float = 5.0
    vehicle_yaw_amp_deg: float = 4.0


# ────────────────────────────────────────────────
# Seabed type enums & angle-dependent BS models
# ────────────────────────────────────────────────
SEABED_TYPES = {
    'mud':    {'roughness': 0.4, 'volume_scatter': 0.008, 'impedance_ratio': 1.05},
    'sand':   {'roughness': 1.1, 'volume_scatter': 0.018, 'impedance_ratio': 1.25},
    'gravel': {'roughness': 2.2, 'volume_scatter': 0.035, 'impedance_ratio': 1.45},
    'rock':   {'roughness': 3.5, 'volume_scatter': 0.06,  'impedance_ratio': 1.80},
}

def backscattering_strength(grazing_deg: float, seabed_type: str) -> float:
    """Semi-empirical angle-dependent BS (dB) — APL-UW / ESAB inspired"""
    if grazing_deg < 1.0:
        grazing_deg = 1.0
    grazing_rad = np.deg2rad(grazing_deg)
    
    p = SEABED_TYPES[seabed_type]
    # Interface (facet + Bragg approx) — strong at high grazing
    bs_interface = 10 * np.log10(p['impedance_ratio']**2 * np.cos(grazing_rad)**2 + 0.01)
    
    # Roughness term — falls off at low grazing
    roughness_term = p['roughness'] * np.sin(grazing_rad)**1.5
    
    # Volume scattering — more constant, but angle-dependent path
    bs_volume = 10 * np.log10(p['volume_scatter'] * (1 + 3 * np.sin(grazing_rad)))
    
    # Total (dB) — blend interface → volume at low grazing
    weight_interface = np.exp(- (90 - grazing_deg)/25.0)
    bs_total_db = weight_interface * bs_interface + (1 - weight_interface) * bs_volume + roughness_term - 12.0
    
    return bs_total_db


# ────────────────────────────────────────────────
# Target
# ────────────────────────────────────────────────
@dataclass
class Target:
    x: float
    y: float          # >0 port, <0 starboard
    z: float = 0.8
    strength: float = 1.0


# ────────────────────────────────────────────────
# Seabed reverberation with type-dependent BS + Perlin texture
# ────────────────────────────────────────────────
def get_seabed_type(horiz_dist: float) -> str:
    """Simple zoned seabed types (demo)"""
    if horiz_dist < 25: return 'mud'
    if horiz_dist < 55: return 'sand'
    if horiz_dist < 80: return 'gravel'
    return 'rock'


def seabed_reverberation(n_bins: int, horiz_dist: np.ndarray, grazing_deg: np.ndarray,
                         params: SonarParams, ping_seed: int) -> np.ndarray:
    rev = np.zeros(n_bins)
    for i in range(n_bins):
        if horiz_dist[i] < 1e-3:
            continue
        seabed = get_seabed_type(horiz_dist[i])
        bs_db = backscattering_strength(grazing_deg[i], seabed)
        bs_lin = 10 ** (bs_db / 10)
        
        # Perlin texture modulation
        x = i * 0.35 + 150
        per = noise.pnoise2(x / params.seabed_texture_scale,
                            horiz_dist[i] / (params.seabed_texture_scale * 0.8),
                            octaves=4, persistence=0.55, lacunarity=2.1,
                            base=ping_seed % 100)
        per = 0.7 + 0.6 * (per + 1)/2
        rev[i] = bs_lin * per * 0.015  # scale to reasonable level
    
    return rev


# ────────────────────────────────────────────────
# Realistic beam pattern (sinc horizontal, cos vertical approx)
# ────────────────────────────────────────────────
def beam_pattern_horizontal(angle_along_rad: float, params: SonarParams) -> float:
    """Sinc-like for line array directivity (typical SSS)"""
    kD = np.pi * np.sin(angle_along_rad) / np.deg2rad(params.horizontal_beam_width_deg / 2)
    if abs(kD) < 1e-6:
        return 1.0
    return (np.sin(kD) / kD) ** 2


def beam_pattern_vertical(grazing_rad: float, params: SonarParams) -> float:
    """Wide fan — cosine approx"""
    angle_from_nadir = np.pi/2 - grazing_rad
    return np.cos(angle_from_nadir * 2.0) ** 2   # broad lobe


# ────────────────────────────────────────────────
# Simplified 2.5D ray multipath (direct + bottom bounce)
# ────────────────────────────────────────────────
def add_multipath(side: np.ndarray, params: SonarParams, eff_alt: float):
    # First bottom reflection — virtual source below
    mp_delay_m = 2 * eff_alt * 1.08  # slight extra path
    mp_bins = int(mp_delay_m / params.range_resolution)
    if mp_bins < 1 or mp_bins >= len(side) // 2:
        return
    atten = params.multipath_attenuation * 0.7  # extra bottom loss
    side[mp_bins:] += atten * side[:-mp_bins]


# ────────────────────────────────────────────────
# Simulate ping
# ────────────────────────────────────────────────
def simulate_sonar_ping(veh_x: float, ping_idx: int, targets: List[Target],
                        params: SonarParams) -> tuple[np.ndarray, np.ndarray]:
    n_bins = int(params.max_range / params.range_resolution)
    port = np.zeros(n_bins)
    starb = np.zeros(n_bins)

    hor_half_rad = np.deg2rad(params.horizontal_beam_width_deg / 2)

    # Vehicle attitude
    roll_deg = params.vehicle_roll_amp_deg * np.sin(2 * np.pi * ping_idx / 140)
    yaw_deg  = params.vehicle_yaw_amp_deg * np.sin(2 * np.pi * ping_idx / 320)
    roll_rad = np.deg2rad(roll_deg)
    yaw_rad  = np.deg2rad(yaw_deg)
    eff_alt = params.altitude * np.cos(roll_rad)

    slant_arr = np.arange(n_bins) * params.range_resolution + 1e-3
    horiz_arr = np.sqrt(np.maximum(slant_arr**2 - eff_alt**2, 0))
    grazing_rad = np.arctan2(eff_alt, horiz_arr + 1e-6)
    grazing_deg = np.rad2deg(grazing_rad)

    # ── Seabed reverberation ─────────────────────
    rev_port = seabed_reverberation(n_bins, horiz_arr, grazing_deg, params, ping_idx + 10)
    rev_starb = seabed_reverberation(n_bins, horiz_arr, grazing_deg, params, ping_idx + 20)
    port += rev_port
    starb += rev_starb

    # ── Targets + beam pattern ───────────────────
    for tgt in targets:
        dy = tgt.y
        dx = tgt.x - veh_x
        # Yaw rotation
        dy_rot = dy * np.cos(yaw_rad) - dx * np.sin(yaw_rad)
        dx_rot = dy * np.sin(yaw_rad) + dx * np.cos(yaw_rad)

        slant = np.sqrt(dx_rot**2 + dy_rot**2 + eff_alt**2)
        if slant > params.max_range or slant < eff_alt * 0.7:
            continue

        angle_along = np.arctan2(dx_rot, eff_alt)
        if abs(angle_along) > hor_half_rad * 1.4:
            continue

        grazing = np.arctan2(eff_alt, abs(dy_rot))
        if grazing < 0.03:
            continue

        bp_h = beam_pattern_horizontal(angle_along, params)
        bp_v = beam_pattern_vertical(grazing, params)
        beam_gain = bp_h * bp_v

        # Target strength simplified
        intensity = tgt.strength * beam_gain * 1.5

        spreading = 1.0 / (slant ** params.spreading_loss_exp)
        absorption = 10 ** (-params.absorption_db_per_m * slant * 2 / 20)
        intensity *= spreading * absorption

        bin_idx = int(slant / params.range_resolution)
        if bin_idx >= n_bins:
            continue

        side = port if dy_rot > 0 else starb
        side[bin_idx] += intensity

        # Shadow (slant projected)
        shadow_bins = int(tgt.z / np.sin(grazing) / params.range_resolution * 1.6)
        for s in range(1, min(shadow_bins + 1, n_bins - bin_idx)):
            side[bin_idx + s] *= params.shadow_attenuation

    # ── Multipath (geometric) ────────────────────
    add_multipath(port, params, eff_alt)
    add_multipath(starb, params, eff_alt)

    # ── Sidelobes (simple rings) ─────────────────
    for r_m in [28.0, 62.0]:
        sl_bin = int(r_m / params.range_resolution)
        if 4 < sl_bin < n_bins - 4:
            win = np.hanning(9) * params.sidelobe_level
            port[sl_bin-4:sl_bin+5] += np.max(port) * win
            starb[sl_bin-4:sl_bin+5] += np.max(starb) * win

    # ── Pulse smearing ───────────────────────────
    pulse_bins = max(1, int(params.pulse_length_m / params.range_resolution))
    if pulse_bins > 1:
        kernel = np.ones(pulse_bins) / pulse_bins
        port = np.convolve(port, kernel, mode='same')
        starb = np.convolve(starb, kernel, mode='same')

    # ── Speckle noise (Rayleigh envelope) ────────
    # Multiplicative on amplitude (realistic for coherent imaging)
    speckle_port = np.random.rayleigh(scale=1.0, size=n_bins)
    speckle_starb = np.random.rayleigh(scale=1.0, size=n_bins)
    port *= (speckle_port / np.sqrt(2))   # normalize mean ≈1
    starb *= (speckle_starb / np.sqrt(2))

    # Additive thermal noise
    port += np.random.normal(0, 0.008, n_bins)
    starb += np.random.normal(0, 0.008, n_bins)

    # TVG
    if params.apply_tvg:
        r = slant_arr
        tvg_db = 20 * np.log10(r + 1e-3) + 2 * params.absorption_db_per_m * r
        gain = 10 ** (tvg_db / 20)
        port *= gain
        starb *= gain

    return port, starb


# ────────────────────────────────────────────────
# Main
# ────────────────────────────────────────────────
def run_sonar_imaging_simulation(params: SonarParams):
    dt = 1.0 / params.ping_rate
    n_bins = int(params.max_range / params.range_resolution)

    targets = [
        Target(140,  42, 1.6, 2.1),
        Target(260, -58, 0.7, 1.2),
        Target(400,  70, 2.5, 2.6),
        Target(540, -35, 1.0, 1.1),
        Target(680,  60, 1.5, 1.8),
    ]

    raw_image = np.zeros((params.n_pings, 2 * n_bins))

    for ping in range(params.n_pings):
        veh_x = ping * params.vehicle_speed * dt
        port, starb = simulate_sonar_ping(veh_x, ping, targets, params)

        nadir_bins = int(params.nadir_blank_m / params.range_resolution)
        port[:nadir_bins] *= 0.15
        starb[:nadir_bins] *= 0.15

        raw_image[ping, :n_bins] = port
        raw_image[ping, n_bins:] = starb[::-1]

    # Display prep
    display_image = np.maximum(raw_image, 0)**1.35

    if params.log_compress:
        display_image = np.log1p(display_image)
        display_image = (display_image - display_image.min()) / (display_image.max() - display_image.min() + 1e-8)

    # Slant → horizontal correction
    if params.apply_slant_correction:
        horiz_image = np.zeros_like(display_image)
        for row in range(display_image.shape[0]):
            slant = np.arange(n_bins) * params.range_resolution
            horiz = np.sqrt(np.maximum(slant**2 - params.altitude**2, 0))
            for col in range(n_bins):
                if horiz[col] > params.max_range:
                    continue
                new_col = int(horiz[col] / params.range_resolution)
                if new_col < n_bins:
                    horiz_image[row, new_col] = max(horiz_image[row, new_col], display_image[row, col])
                    horiz_image[row, 2*n_bins-1-new_col] = max(
                        horiz_image[row, 2*n_bins-1-new_col], display_image[row, 2*n_bins-1-col])
        display_image = horiz_image

    # Plot
    along_m = params.n_pings * params.vehicle_speed * dt
    extent = [-params.max_range, params.max_range, 0, along_m]

    plt.figure(figsize=(15, 10))
    plt.imshow(display_image, aspect='auto', cmap='gray_r', origin='lower', extent=extent)
    plt.title("Advanced Side-Scan Sonar Simulation\n(realistic beam, angle-dep BS, seabed zones, Rayleigh speckle, ray multipath)")
    plt.xlabel("Across-track [m] (horizontal if corrected)")
    plt.ylabel("Along-track [m]")
    plt.colorbar(label='Normalized Intensity')
    plt.grid(alpha=0.1)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    cfg = SonarParams()
    # Tuning ideas:
    # cfg.altitude = 10.0
    # cfg.horizontal_beam_width_deg = 0.6
    # cfg.apply_slant_correction = False
    # cfg.pulse_length_m = 0.8
    run_sonar_imaging_simulation(cfg)