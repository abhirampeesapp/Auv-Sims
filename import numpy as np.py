import numpy as np
import matplotlib.pyplot as plt
import cv2
from scipy.ndimage import gaussian_filter

# ===============================
# 1. BIO-INSPIRED AUV MOTION (3D)
# ===============================
dt = 0.1
steps = 1000
v = 0.8           # forward speed (m/s)
yaw_amp = np.deg2rad(15)   # fish-like yaw amplitude
yaw_freq = 0.05
depth_amp = 2.0   # depth oscillation amplitude (m)
depth_freq = 0.03

x, y, z, yaw = 0.0, 0.0, -10.0, 0.0
trajectory = []

for t in range(steps):
    yaw = yaw_amp * np.sin(yaw_freq * t)
    z = -10 + depth_amp * np.sin(depth_freq * t)
    x += v * np.cos(yaw) * dt
    y += v * np.sin(yaw) * dt
    trajectory.append((x, y, z, yaw))

trajectory = np.array(trajectory)

# ===============================
# 2. VIRTUAL SEABED with rods & corrosion
# ===============================
H, W = 300, 300

# Base seabed terrain (smooth rolling surface)
base_noise = np.random.normal(0, 1, (H, W))
seabed = gaussian_filter(base_noise, sigma=12)
seabed = (seabed - seabed.min()) / (seabed.max() - seabed.min() + 1e-8)
seabed = seabed * 0.35 + 0.25   # range ~[0.25, 0.6] for sandy/muddy look

# Add rod-like / cylindrical objects
for _ in range(6):
    cx = np.random.randint(50, 250)
    cy = np.random.randint(50, 250)
    length = np.random.randint(25, 55)
    width  = np.random.randint(5, 11)
    angle  = np.random.uniform(0, 360)

    # Force Python floats → fixes boxPoints parsing error in many OpenCV versions
    rect = (
        (float(cx), float(cy)),           # center
        (float(length), float(width)),    # size (width, height)
        float(angle)                      # angle in degrees
    )

    box = cv2.boxPoints(rect).astype(np.int32)
    cv2.fillPoly(seabed, [box], 0.88)     # bright value for object

# Add realistic corrosion / pitting to objects
obj_mask = seabed > 0.75
if np.any(obj_mask):
    # Small random pits (corrosion holes)
    pit_y, pit_x = np.where(obj_mask)
    for i in np.random.choice(len(pit_y), size=min(80, len(pit_y)), replace=False):
        py, px = pit_y[i], pit_x[i]
        r = np.random.randint(1, 4)
        cv2.circle(seabed, (px, py), r, 0.55, -1)  # darker pits

    # Surface roughness / corrosion noise on objects
    noise = np.random.normal(0, 0.06, seabed.shape)
    seabed[obj_mask] += noise[obj_mask]
    seabed = np.clip(seabed, 0, 1)

# ===============================
# 3. CAMERA IMAGE CAPTURE
# ===============================
def capture_camera(seabed, pose):
    px, py, pz, yaw = pose
    px = int(px) % W
    py = int(py) % H

    half = 60
    crop = seabed[max(py-half,0):py+half, max(px-half,0):px+half]

    # Pad if crop is smaller than expected (near edges)
    if crop.shape[0] != 2*half or crop.shape[1] != 2*half:
        crop = np.pad(crop, ((0, 2*half - crop.shape[0]), (0, 2*half - crop.shape[1])), mode='edge')

    crop = cv2.resize(crop, (128, 128))

    # Slight perspective / viewpoint tilt simulation
    M = cv2.getAffineTransform(
        np.float32([[0,0], [128,0], [0,128]]),
        np.float32([[5,8], [123,-3], [-2,125]])   # small random-like distortion
    )
    crop = cv2.warpAffine(crop, M, (128,128), borderMode=cv2.BORDER_REPLICATE)
    return crop

frames = [capture_camera(seabed, pose) for pose in trajectory[::20]]
raw_image = frames[-1]   # last frame for visualization

# ===============================
# 4. REALISTIC UNDERWATER DEGRADATION
# ===============================
def underwater_degrade(img):
    img = img.astype(np.float32)
    
    # Base color cast (blue-green dominance + red absorption)
    red   = img * 0.55
    green = img * 0.85
    blue  = img * 1.10
    color = np.stack([red, green, blue], axis=-1)

    # Exponential depth-dependent attenuation
    att = np.exp(-0.07 * np.abs(color))   # stronger for red
    color *= att

    # Forward scattering + backscattering (haze/murk)
    haze = gaussian_filter(color, sigma=10)
    degraded = 0.48 * color + 0.52 * haze

    # Particle / marine snow noise
    noise = np.random.normal(0, 0.07, degraded.shape)
    degraded += noise

    # Corner vignette (lens + light falloff)
    yy, xx = np.ogrid[:128, :128]
    dist = np.sqrt((xx-64)**2 + (yy-64)**2) / 100
    vignette = np.clip(1.1 - dist**1.4, 0.3, 1.0)
    degraded *= vignette[..., np.newaxis]

    return np.clip(degraded, 0, 1)

raw_underwater = underwater_degrade(raw_image)

# ===============================
# 5. MULTI-SCALE RETINEX (simple MSRCR-like)
# ===============================
def multi_scale_retinex(img, sigmas=[10, 60, 180]):
    img = img + 1e-6
    retinex = np.zeros_like(img)
    for sigma in sigmas:
        blur = gaussian_filter(img, sigma=(sigma, sigma, 0))
        retinex += np.log1p(img) - np.log1p(blur)
    retinex /= len(sigmas)
    return retinex

def enhance_msrcr(img):
    ret = multi_scale_retinex(img)
    enhanced = cv2.normalize(ret, None, 0, 1, cv2.NORM_MINMAX)
    # Optional: slight gamma + contrast stretch
    enhanced = np.power(enhanced, 0.95)
    enhanced = cv2.normalize(enhanced, None, 0, 1, cv2.NORM_MINMAX)
    return enhanced

enhanced = enhance_msrcr(raw_underwater)

# ===============================
# 6. SIMPLE ROD/OBJECT DETECTION
# ===============================
def detect_rods(img):
    gray = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_BGR2GRAY)
    # Enhance contrast
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    gray = clahe.apply(gray)

    thresh = cv2.adaptiveThreshold(gray, 255,
                                   cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 11, 3)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    detections = []
    img_draw = img.copy()

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 60 or area > 4000:
            continue

        x,y,w,h = cv2.boundingRect(cnt)
        aspect = w / float(h) if h > 0 else 1

        # Rod-like = elongated shape
        if aspect > 2.2 or aspect < 0.45:
            cv2.rectangle(img_draw, (x,y), (x+w,y+h), (0,1,0), 2)
            detections.append((x,y,w,h,aspect))

    return img_draw, detections

detected_img, rods = detect_rods(enhanced)
print(f"Detected {len(rods)} rod-like / elongated objects.")

# ===============================
# 7. VISUALIZATION
# ===============================
fig = plt.figure(figsize=(15, 11))

plt.subplot(2,3,1)
plt.title("Raw Camera View")
plt.imshow(raw_image, cmap='gray')
plt.axis('off')

plt.subplot(2,3,2)
plt.title("Underwater Degraded")
plt.imshow(raw_underwater)
plt.axis('off')

plt.subplot(2,3,3)
plt.title("MSR Enhanced")
plt.imshow(enhanced)
plt.axis('off')

plt.subplot(2,3,4)
plt.title("Detected Rods / Objects")
plt.imshow(detected_img)
plt.axis('off')

ax = fig.add_subplot(2,3,(5,6), projection='3d')
ax.plot(trajectory[:,0], trajectory[:,1], trajectory[:,2], lw=1.5)
ax.set_title("AUV Bio-inspired Trajectory")
ax.set_xlabel("X (m)")
ax.set_ylabel("Y (m)")
ax.set_zlabel("Depth (m)")
ax.view_init(elev=20, azim=135)

plt.tight_layout()
plt.show()