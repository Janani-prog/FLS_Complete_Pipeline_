
from pathlib import Path
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import least_squares
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, WhiteKernel
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import numpy as np
import warnings

warnings.filterwarnings("ignore")

# ============================================================================
# PART 1: POINT CLOUD PROCESSING
# ============================================================================

def load_point_cloud(filepath):
    """Load point cloud from file and check units."""
    print(f"Loading: {Path(filepath).name}")
    path = Path(filepath)
    suffix = path.suffix.lower()

    if suffix in [".xyz", ".txt", ".csv"]:
        points = np.loadtxt(filepath)[:, :3]
    elif suffix == ".npy":
        points = np.load(filepath)
    elif suffix == ".ply":
        import open3d as o3d
        o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Error)
        pcd = o3d.io.read_point_cloud(str(filepath))
        if len(pcd.points) == 0:
            raise ValueError("PLY file loaded but contains no points")
        points = np.asarray(pcd.points)
    else:
        raise ValueError(f"Unsupported format: {suffix}")

    max_dim = np.ptp(points, axis=0).max()
    if max_dim < 100:
        print(f" Converting to mm (detected meters)...")
        points *= 1000.0

    print(f" Loaded {len(points):,} points")
    return points

def correct_yz_alignment(points, n_iterations=3):
    """Iteratively correct YZ centering."""
    corrected = points.copy()
    
    for iteration in range(n_iterations):
        x_mid = np.median(corrected[:, 0])
        slice_mask = np.abs(corrected[:, 0] - x_mid) < 250.0
        slice_pts = corrected[slice_mask]

        if len(slice_pts) < 10: 
            break

        r_all = np.sqrt(slice_pts[:, 1] ** 2 + slice_pts[:, 2] ** 2)
        r_thresh = np.percentile(r_all, 92.0)
        shell_mask = r_all >= r_thresh
        shell_pts_yz = slice_pts[shell_mask, 1:]
        r_shell = r_all[shell_mask]

        if len(r_shell) == 0: 
            break

        r_median = np.median(r_shell)
        r_std = np.std(r_shell)
        inlier = np.abs(r_shell - r_median) < 2.0 * r_std
        shell_pts_yz = azimuthally_balanced_shell_points(shell_pts_yz)


        if len(shell_pts_yz) < 100:
            break

        def residuals(params):
            cy, cz, R = params
            return np.sqrt((shell_pts_yz[:, 0] - cy) ** 2 + (shell_pts_yz[:, 1] - cz) ** 2) - R

        r_init = np.median(r_shell[inlier])
        result = least_squares(residuals, [0.0, 0.0, r_init], loss="soft_l1", f_scale=10.0)
        cy, cz, R_fit = result.x

        corrected[:, 1] -= cy
        corrected[:, 2] -= cz

        if np.sqrt(cy ** 2 + cz ** 2) < 3.0:
            break

    return corrected

def find_radial_cutoff(rho, R):
    """
    Automatically separates shell from interior using radial distribution.
    """
    if len(rho) < 50:
        return R

    norm_r = rho / R
    hist, edges = np.histogram(norm_r, bins='auto')

    peak_idx = np.argmax(hist)

    # find valley after shell peak
    for i in range(peak_idx + 1, len(hist) - 1):
        if hist[i] < hist[i - 1] and hist[i] < hist[i + 1]:
            return edges[i] * R

    # safe fallback
    return R - np.std(rho)


def detect_charge_surface_yz(points, cy, cz, radius):
    """
    Detect charge surface as highest Z where a continuous interior band exists across Y.
    """

    y = points[:, 1] - cy
    z = points[:, 2] - cz
    r = np.sqrt(y*y + z*z)

    # Strict interior (kills shell bleed)
    interior = r < (0.78 * radius)
    y, z = y[interior], z[interior]

    if len(z) < 200:
        return cz - radius + 100.0  # safe fallback

    # Z slicing
    z_bins = np.linspace(np.min(z), np.max(z), 180)

    best_z = None

    for zb in z_bins:
        mask = np.abs(z - zb) < (radius * 0.012)
        if np.sum(mask) < 40:
            continue

        y_slice = y[mask]

        # Measure Y continuity
        y_sorted = np.sort(y_slice)
        gaps = np.diff(y_sorted)

        # Large gap → broken surface
        max_gap = np.max(gaps) if len(gaps) else np.inf

        # This threshold is physical: empty space gap
        if max_gap < (0.15 * radius):
            best_z = zb
        else:
            # once continuity breaks while going up → stop
            if best_z is not None:
                break

    if best_z is None:
        return cz - radius + 150.0

    return best_z + cz




def azimuthally_balanced_shell_points(shell_points):
    """
    Corrects biased shell sampling by enforcing angular balance.
    Automatically bypasses good scans.
    No hardcoded thresholds.
    """
    if len(shell_points) < 200:
        return shell_points

    y = shell_points[:, 0]
    z = shell_points[:, 1]

    theta = np.arctan2(z, y)
    radius = np.sqrt(y**2 + z**2)

    n_bins = max(24, int(np.sqrt(len(theta))))
    bins = np.linspace(-np.pi, np.pi, n_bins + 1)

    counts, _ = np.histogram(theta, bins=bins)
    mean_count = np.mean(counts[counts > 0])
    std_count = np.std(counts[counts > 0])

    if mean_count <= 0:
        return shell_points

    angular_cv = std_count / mean_count

    # GOOD SCAN → do nothing
    if angular_cv < np.percentile(counts[counts > 0], 75) / mean_count:
        return shell_points

    # PROBLEMATIC SCAN → balance angular contribution
    balanced = []

    for i in range(n_bins):
        mask = (theta >= bins[i]) & (theta < bins[i + 1])
        if np.sum(mask) < 5:
            continue

        r_bin = radius[mask]
        pts_bin = shell_points[mask]

        r_peak = np.percentile(r_bin, 95)
        spread = np.std(r_bin)

        if spread <= 0:
            continue

        keep = np.abs(r_bin - r_peak) < spread
        if np.any(keep):
            balanced.append(pts_bin[keep])

    if not balanced:
        return shell_points

    return np.vstack(balanced)


# --- TRIAL 9 LOGIC (Diameter & Length) ---

def find_head_planes_trial9(points):
    x = points[:, 0]
    r = np.sqrt(points[:, 1] ** 2 + points[:, 2] ** 2)

    x_range = np.ptp(x)
    n_bins = max(100, min(200, int(x_range / 50)))
    
    x_bounds = np.percentile(x, [1.0, 99.0])
    x_edges = np.linspace(x_bounds[0], x_bounds[1], n_bins + 1)
    med_radius = np.full(n_bins, np.nan, dtype=float)

    for i in range(n_bins):
        mask = (x >= x_edges[i]) & (x < x_edges[i + 1])
        if np.sum(mask) > 50:
            med_radius[i] = np.median(r[mask])

    valid_mask = ~np.isnan(med_radius)
    med_radius_smooth = np.copy(med_radius)
    
    sigma = max(2.0, min(4.0, n_bins / 50))
    med_radius_smooth[valid_mask] = gaussian_filter1d(med_radius[valid_mask], sigma=sigma)

    valid_r = med_radius_smooth[valid_mask]
    high_quartile = valid_r[valid_r >= np.percentile(valid_r, 75)]
    plateau_r_median = np.median(high_quartile)

    r_std = np.std(high_quartile)
    r_cv = r_std / plateau_r_median
    
    if r_cv < 0.02:
        threshold_pct = 0.96
    elif r_cv < 0.04:
        threshold_pct = 0.94
    else:
        threshold_pct = 0.92
    
    threshold_low = threshold_pct * plateau_r_median
    threshold_high = (2.0 - threshold_pct) * plateau_r_median
    
    plateau_mask = (med_radius_smooth >= threshold_low) & (med_radius_smooth <= threshold_high)
    plateau_indices = np.where(plateau_mask)[0]

    if len(plateau_indices) < 5:
        threshold_low = 0.90 * plateau_r_median
        threshold_high = 1.10 * plateau_r_median
        plateau_mask = (med_radius_smooth >= threshold_low) & (med_radius_smooth <= threshold_high)
        plateau_indices = np.where(plateau_mask)[0]

    i_start = max(0, plateau_indices[0] - 1)
    i_end = min(n_bins - 1, plateau_indices[-1] + 1)
    x_min = x_edges[i_start]
    x_max = x_edges[i_end + 1]
    effective_length = x_max - x_min

    return x_min, x_max, effective_length

def fit_shell_circle_trial9(points, x_min, x_max):
    x_mid = 0.5 * (x_min + x_max)
    x_positions = [x_mid, x_mid - 600.0, x_mid + 600.0, x_mid - 300.0, x_mid + 300.0]

    all_shell_pts = []
    all_radii = []

    for x_pos in x_positions:
        slice_mask = np.abs(points[:, 0] - x_pos) < 150.0
        slice_pts = points[slice_mask]
        if len(slice_pts) < 300:
            continue

        r_all = np.sqrt(slice_pts[:, 1] ** 2 + slice_pts[:, 2] ** 2)
        r_shell_cutoff = np.percentile(r_all, 96.0)
        shell_mask = r_all >= r_shell_cutoff
        shell_pts_yz = slice_pts[shell_mask, 1:]
        
        if len(shell_pts_yz) > 50:
            r_shell = r_all[shell_mask]
            r_median = np.median(r_shell)
            inlier_mask = np.abs(r_shell - r_median) < 0.05 * r_median
            
            all_shell_pts.append(shell_pts_yz[inlier_mask])
            all_radii.append(r_shell[inlier_mask])

    if len(all_shell_pts) == 0:
        slice_mask = np.abs(points[:, 0] - x_mid) < 1000.0
        slice_pts = points[slice_mask]
        if len(slice_pts) > 0:
            r_all = np.sqrt(slice_pts[:, 1] ** 2 + slice_pts[:, 2] ** 2)
            shell_mask = r_all >= np.percentile(r_all, 95.0)
            shell_points = slice_pts[shell_mask, 1:]
        else:
            raise ValueError("Failed to extract shell points")
    else:
        shell_points = azimuthally_balanced_shell_points(np.vstack(all_shell_pts))


    def residuals(params):
        cy, cz, radius = params
        return np.sqrt((shell_points[:, 0] - cy) ** 2 + (shell_points[:, 1] - cz) ** 2) - radius

    if len(all_radii) > 0:
        r_init = np.median(np.concatenate(all_radii))
    else:
        r_init = np.median(np.sqrt(shell_points[:,0]**2 + shell_points[:,1]**2))

    result = least_squares(residuals, [0.0, 0.0, r_init], loss="soft_l1", f_scale=15.0)
    cy, cz, r_outer = result.x

    if r_outer > 5500: 
        liner_pct = 0.012
    elif r_outer > 3200: 
        liner_pct = 0.016
    elif r_outer > 2500: 
        liner_pct = 0.020
    else: 
        liner_pct = 0.024

    liner_correction = liner_pct * r_outer
    liner_correction = max(70.0, min(180.0, liner_correction))
    r_effective = r_outer - liner_correction

    return cy, cz, r_effective

# --- TRIAL 4 LOGIC (Free Height) ---

def find_head_planes_trial4(points):
    x = points[:, 0]
    r = np.sqrt(points[:, 1] ** 2 + points[:, 2] ** 2)

    n_bins = 150
    x_bounds = np.percentile(x, [0.5, 99.5])
    x_edges = np.linspace(x_bounds[0], x_bounds[1], n_bins + 1)
    med_radius = np.full(n_bins, np.nan, dtype=float)

    for i in range(n_bins):
        mask = (x >= x_edges[i]) & (x < x_edges[i + 1])
        if np.sum(mask) > 50:
            med_radius[i] = np.median(r[mask])

    valid_mask = ~np.isnan(med_radius)
    med_radius_smooth = np.copy(med_radius)
    med_radius_smooth[valid_mask] = gaussian_filter1d(med_radius[valid_mask], sigma=3)

    valid_r = med_radius_smooth[valid_mask]
    high_third = valid_r[valid_r >= np.percentile(valid_r, 67)]
    plateau_r_median = np.median(high_third)

    threshold_low = 0.93 * plateau_r_median
    threshold_high = 1.07 * plateau_r_median
    plateau_mask = (med_radius_smooth >= threshold_low) & (med_radius_smooth <= threshold_high)
    plateau_indices = np.where(plateau_mask)[0]

    if len(plateau_indices) < 5:
        threshold_low = 0.90 * plateau_r_median
        threshold_high = 1.10 * plateau_r_median
        plateau_mask = (med_radius_smooth >= threshold_low) & (med_radius_smooth <= threshold_high)
        plateau_indices = np.where(plateau_mask)[0]

    if len(plateau_indices) < 5:
        return x_bounds[0], x_bounds[1]

    i_start = max(0, plateau_indices[0] - 2)
    i_end = min(n_bins - 1, plateau_indices[-1] + 2)
    x_min = x_edges[i_start]
    x_max = x_edges[i_end + 1]

    return x_min, x_max

def fit_shell_circle_trial4(points, x_min, x_max):
    x_mid = 0.5 * (x_min + x_max)
    x_positions = [x_mid, x_mid - 400.0, x_mid + 400.0]

    all_shell_pts = []
    all_radii = []

    for x_pos in x_positions:
        slice_mask = np.abs(points[:, 0] - x_pos) < 200.0
        slice_pts = points[slice_mask]

        if len(slice_pts) < 500: 
            continue

        r_all = np.sqrt(slice_pts[:, 1] ** 2 + slice_pts[:, 2] ** 2)
        r_shell_cutoff = np.percentile(r_all, 95.0)
        shell_mask = r_all >= r_shell_cutoff
        shell_pts_yz = slice_pts[shell_mask, 1:]
        shell_radii = r_all[shell_mask]

        if len(shell_pts_yz) > 50:
            all_shell_pts.append(shell_pts_yz)
            all_radii.append(shell_radii)

    if len(all_shell_pts) == 0:
        slice_mask = np.abs(points[:, 0] - x_mid) < 500.0
        slice_pts = points[slice_mask]
        r_all = np.sqrt(slice_pts[:, 1] ** 2 + slice_pts[:, 2] ** 2)
        shell_mask = r_all >= np.percentile(r_all, 95.0)
        shell_points = slice_pts[shell_mask, 1:]
        r_init = np.median(r_all[shell_mask])
    else:
        shell_points = azimuthally_balanced_shell_points(np.vstack(all_shell_pts))
        r_init = np.median(np.concatenate(all_radii))

    def residuals(params):
        cy, cz, radius = params
        return np.sqrt((shell_points[:, 0] - cy) ** 2 + (shell_points[:, 1] - cz) ** 2) - radius

    result = least_squares(residuals, [0.0, 0.0, r_init], loss="soft_l1", f_scale=20.0)
    cy, cz, r_outer = result.x

    if r_outer > 5500: 
        liner_pct = 0.015
    elif r_outer > 3200: 
        liner_pct = 0.020
    elif r_outer > 2500: 
        liner_pct = 0.025
    else: 
        liner_pct = 0.030

    liner_correction = liner_pct * r_outer
    liner_correction = max(80.0, min(200.0, liner_correction))
    r_effective = r_outer - liner_correction

    return cy, cz, r_effective

def _adaptive_correction_factor(fill_raw, noise_ratio, point_coverage_ratio, interior_density, mill_diameter_m):
    """Compute correction factor - ADHI method."""
    q_noise = max(0.0, 0.04 - min(noise_ratio, 0.04)) / 0.04
    q_cov = min(1.0, point_coverage_ratio / 0.25)
    q_den = min(1.0, interior_density / 1.0)
    quality_score = 0.5 * q_noise + 0.3 * q_cov + 0.2 * q_den
    quality_score = float(np.clip(quality_score, 0.0, 1.0))
    
    def conservative_factor(f):
        if f < 0.20: return 0.84
        if f < 0.25: return 0.87
        if f < 0.30: return 0.90
        if f < 0.35: return 0.93
        if f < 0.40: return 0.95
        if f < 0.50: return 0.97
        return 0.98
    
    def aggressive_factor(f):
        if f < 0.20: return 1.00
        if f < 0.25: return 1.05
        if f < 0.30: return 1.08
        if f < 0.35: return 1.10
        if f < 0.40: return 1.11
        if f < 0.50: return 1.12
        return 1.14
    
    base_cons = conservative_factor(fill_raw)
    base_aggr = aggressive_factor(fill_raw)
    
    w_aggr = 1.0 - quality_score
    correction_factor = (1.0 - w_aggr) * base_cons + w_aggr * base_aggr
    
    if point_coverage_ratio < 0.02: 
        coverage_scale = 0.86
    elif point_coverage_ratio < 0.05: 
        coverage_scale = 0.89
    elif point_coverage_ratio < 0.08: 
        coverage_scale = 0.92
    elif point_coverage_ratio < 0.12: 
        coverage_scale = 0.95
    elif point_coverage_ratio < 0.20: 
        coverage_scale = 0.97
    else: 
        coverage_scale = 1.0
    
    if mill_diameter_m > 11.5: 
        diameter_scale = 0.94
    elif mill_diameter_m > 10.5: 
        diameter_scale = 0.95
    elif mill_diameter_m > 9.5: 
        diameter_scale = 0.96
    elif mill_diameter_m > 7.0: 
        diameter_scale = 0.97
    elif mill_diameter_m > 6.0: 
        diameter_scale = 0.98
    else: 
        diameter_scale = 1.0
    
    low_fill_boost = 1.0
    if fill_raw < 0.22:
        low_fill_boost = 1.06 if interior_density < 0.8 else 1.04
    elif fill_raw < 0.28:
        low_fill_boost = 1.04 if interior_density < 0.8 else 1.02
    
    noise_penalty = 1.0
    if noise_ratio > 0.030: 
        noise_penalty = 0.97
    elif noise_ratio > 0.025: 
        noise_penalty = 0.98
    
    mid_size_boost = 1.0
    if 5.7 <= mill_diameter_m <= 6.5:
        if 0.25 <= fill_raw <= 0.35:
            if quality_score > 0.6: 
                mid_size_boost = 1.08
            elif quality_score > 0.4: 
                mid_size_boost = 1.06
            else: 
                mid_size_boost = 1.04
        elif 0.35 < fill_raw <= 0.40:
            mid_size_boost = 1.03
    
    correction_factor *= coverage_scale * diameter_scale * low_fill_boost * noise_penalty * mid_size_boost
    return correction_factor

def detect_bottom_adaptive(points, cy, cz, radius, x_min, x_max):
    in_cyl = (points[:, 0] >= x_min) & (points[:, 0] <= x_max)
    pts = points[in_cyl]
    y_rel = pts[:, 1] - cy
    z_rel = pts[:, 2] - cz
    r_rel = np.sqrt(y_rel**2 + z_rel**2)

    z_min_data = np.min(z_rel)
    print(f"  Z_rel: {z_min_data:.0f} to {np.max(z_rel):.0f}mm")

    # 1. Base plate check (strict: unchanged)
    z_bottom_pct = np.percentile(z_rel, 3)  # Tighter: 3rd% (vs 5th)
    bottom_mask = z_rel < z_bottom_pct
    bottom_pts = np.sum(bottom_mask)
    bottom_y_span = np.ptp(y_rel[bottom_mask]) if bottom_pts > 300 else 0  # Higher min pts
    bottom_flatness = np.std(z_rel[bottom_mask]) if bottom_pts > 0 else 999
    bottom_density = bottom_pts / ((x_max-x_min) * (z_bottom_pct - z_min_data) * 0.001**2)

    plate_diam = 0.88 * 2 * radius
    has_plate = (bottom_pts > 600 and 
                 bottom_y_span > plate_diam and 
                 bottom_density > 60 and 
                 bottom_flatness < 15)  # NEW: Flatness check <15mm std

    if has_plate:
        bottom_z_abs = cz + np.median(z_rel[bottom_mask])
        print(f"  ✓ PLATE: Z={bottom_z_abs:.0f} | pts={bottom_pts} | flat={bottom_flatness:.0f}")
        return bottom_z_abs

    # 2. Shell evidence (TUNED: 0.92-0.98R quartile, median of lowest 3%)
    shell_lower = 0.92 * radius
    shell_upper = 0.98 * radius
    shell_mask = (r_rel >= shell_lower) & (r_rel <= shell_upper)
    shell_pts = np.sum(shell_mask)
    if shell_pts > 300:
        shell_z_low_pct = np.percentile(z_rel[shell_mask], 3)  # TUNED: 3rd% of shell
        bottom_z_abs = cz + shell_z_low_pct
        print(f"  → Shell lower: Z={bottom_z_abs:.0f} (pts={shell_pts})")
        return bottom_z_abs

    # 3. All data lowest 1.5% (TUNED: robust for sparse)
    data_low = np.percentile(z_rel, 1.5)
    bottom_z_abs = cz + data_low
    print(f"  → Data low: Z={bottom_z_abs:.0f} (1.5th%)")
    return bottom_z_abs

def compute_free_height_trial4(points, x_min, x_max, cy, cz, radius):
    """
    UPDATED: Use detect_bottom_adaptive for shell_bottom_z (base-aware).
    Rest unchanged from previous fix.
    """
    # Detect physical bottom (handles no base plate)
    shell_bottom_z = detect_bottom_adaptive(points, cy, cz, radius, x_min, x_max)
    shell_top_z = cz + radius

    print(f"  Shell top/bottom: {shell_top_z:.0f} / {shell_bottom_z:.0f}mm (height={shell_top_z - shell_bottom_z:.0f})")

    # [Keep rest identical: relative coords, interior (0.72R), smoothing, Z-scan, etc.]
    in_cyl = (points[:, 0] >= x_min) & (points[:, 0] <= x_max)
    pts = points[in_cyl]
    y = pts[:, 1] - cy
    z = pts[:, 2] - cz
    r = np.sqrt(y*y + z*z)

    print(f"  Cylinder pts: {len(pts):,} | Raw density: {len(pts)/ ((x_max-x_min) * np.pi * radius**2 * 0.001**2):.1f} pts/m²")

    interior_thresh = 0.72 * radius
    interior = r < interior_thresh
    y_int = y[interior]
    z_int = z[interior]

    print(f"  Interior pts (r<{interior_thresh:.0f}): {len(z_int)} (need >150)")

    if len(z_int) < 150:
        print(f"  → Fallback: Low interior → {radius * 0.12:.0f}mm free height")
        return shell_top_z - (shell_bottom_z + radius * 0.12)

    z_smooth = gaussian_filter1d(z_int, sigma=1.5)
    z_bins = np.linspace(np.min(z_smooth), np.max(z_smooth), 200)
    dz = 0.010 * radius
    surface_z_rel = None
    best_density = 0

    for i, zb in enumerate(z_bins):
        slice_mask = np.abs(z_smooth - zb) < dz
        if np.sum(slice_mask) < 30:
            continue
        y_slice = np.sort(y_int[slice_mask])
        gaps = np.diff(y_slice)
        if len(gaps) == 0: continue
        max_gap = np.max(gaps)
        n_pts_slice = len(y_slice)
        density = n_pts_slice / (2 * radius * dz * 0.001**2)
        if max_gap < (0.20 * radius) and density > best_density:
            surface_z_rel = zb
            best_density = density
            print(f"  Z={zb:.0f} | pts={n_pts_slice} | gap={max_gap:.0f} | dens={density:.0f} → BEST")
        elif surface_z_rel is not None and max_gap >= 0.20 * radius:
            print(f"  Z={zb:.0f} | BROKE (gap={max_gap:.0f}) → STOP")
            break

    if surface_z_rel is None:
        fallback_rel = -0.60 * radius if best_density < 5 else -0.75 * radius
        print(f"  → No surface: fallback Z_rel={fallback_rel:.0f}")
        surface_z_rel = fallback_rel

    z_charge_surface_mm = cz + surface_z_rel
    free_height_mm = shell_top_z - z_charge_surface_mm
    free_height_mm = np.clip(free_height_mm, 0.05 * radius, 1.95 * radius)

    print(f"  → FINAL free_height={free_height_mm:.0f}mm | charge_Z={z_charge_surface_mm:.0f}")
    return free_height_mm

    


# ============================================================================
# PART 2: THE PHYSICS ENGINE & MODEL
# ============================================================================

def get_theoretical_geometry(dia_mm, len_mm, free_height_mm):
    """Calculates the EXACT geometric volume of a perfect cylinder segment."""
    R = dia_mm / 2.0 / 1000.0
    L = len_mm / 1000.0
    H_charge = (dia_mm - free_height_mm) / 1000.0
    
    vol_cyl = np.pi * R**2 * L
    
    if H_charge <= 0:
        area_occ = 0.0
    elif H_charge >= 2*R:
        area_occ = np.pi * R**2
    else:
        val = np.clip((R - H_charge) / R, -1.0, 1.0) 
        term1 = R**2 * np.arccos(val)
        term2 = (R - H_charge) * np.sqrt(max(0, 2*R*H_charge - H_charge**2))
        area_occ = term1 - term2
    
    vol_occ = area_occ * L
    return np.array([vol_cyl, vol_occ])

def predict_single_mill(user_dia, user_len, user_fh):
    X_train = np.array([
        [6870, 3795, 5177],
        [11646, 7657, 8735],
        [5862, 11681, 3634],
        [6464, 11335, 3911],
        [5828, 11677, 3598],
        [6444, 11332, 3894],
        [5850, 11678, 3663],
        [5857, 11681, 3788],
        [5569, 5886, 4230],
        [7043, 3803, 5311],
        [11916, 7668, 8879]
    ])
    
    y_train = np.array([
        [166.6, 28.2, 140.7, 24.3],
        [942.1, 163.7, 815.6, 144.4],
        [337.2, 114.5, 315.3, 108.4],
        [403.4, 141.7, 371.9, 131.5],
        [333.2, 110.8, 311.5, 105.6],
        [400.1, 147.8, 369.6, 136.8],
        [336.0, 115.2, 313.9, 109.1],
        [337.0, 105.7, 314.7, 100.5],
        [146.8, 26.1, 143.4, 25.3],
        [174, 30.3, 148.2, 26.8],
        [991.5, 179.7, 855.1, 159.3]
    ])
    
    print(f"Training prediction model on {len(X_train)} historical files...")

    ratios_train = []
    for i in range(len(X_train)):
        phys = get_theoretical_geometry(X_train[i,0], X_train[i,1], X_train[i,2])
        r = [y_train[i,j] / phys[j%2] for j in range(4)]
        ratios_train.append(r)
    ratios_train = np.array(ratios_train)
    
    kernel = C(1.0) * RBF(length_scale=[1000, 1000, 500]) + WhiteKernel(noise_level=1e-5)
    models = []
    for i in range(4):
        gpr = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=5, random_state=42)
        gpr.fit(X_train, ratios_train[:, i])
        models.append(gpr)
        
    X_user = np.array([[user_dia, user_len, user_fh]])
    phys_user = get_theoretical_geometry(user_dia, user_len, user_fh)
    pred_ratios = [m.predict(X_user)[0] for m in models]
    
    final_preds = [
        phys_user[0] * pred_ratios[0],
        phys_user[1] * pred_ratios[1],
        phys_user[0] * pred_ratios[2],
        phys_user[1] * pred_ratios[3]
    ]
    
    pct_total = (final_preds[1] / final_preds[0]) * 100
    pct_cyl = (final_preds[3] / final_preds[2]) * 100
    final_preds.extend([pct_total, pct_cyl])
    
    labels = [
        'Total Internal Volume (m³)', 
        'Total Volume Occupied (m³)', 
        'Internal Volume of Cylinder (m³)', 
        'Volume Occupied in Cylinder (m³)', 
        'Total Filling Level (%)', 
        'Cylinder Filling Level (%)'
    ]
    
    print("\n" + "="*60)
    print(f"PREDICTION RESULTS (Model-Corrected)")
    print(f"Based on Inputs: Dia={user_dia:.0f}, Len={user_len:.0f}, FH={user_fh:.0f}")
    print("="*60)
    print(f"{'Parameter':<40} | {'Value':<10}")
    print("-" * 60)
    
    for i in range(6):
        print(f"{labels[i]:<40} | {final_preds[i]:<10.2f}")
    print("-" * 60)

    final_results = {
        'total_volume_m3': final_preds[0],
        'charge_total_m3': final_preds[1],
        'cylinder_volume_m3': final_preds[2],
        'charge_cylinder_m3': final_preds[3],
        'total_fill_pct': final_preds[4],
        'cylinder_fill_pct': final_preds[5]
    }

    return final_results


def make_plots(points, results, output_path: Path, filename: str):
    print("\n=== STEP 4: Generate Plots ===")
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
    
    cy = results["cy"]
    cz = results["cz"]
    R = results["radius"]
    x_min = results["xmin"]
    x_max = results["xmax"]
    z_charge_surface = results["zchargesurface"]
    
    # Plot 1: Cross-section
    ax1 = fig.add_subplot(gs[0, 0])
    x_mid = 0.5 * (x_min + x_max)
    slice_mask = np.abs(points[:, 0] - x_mid) < 150.0
    slice_pts = points[slice_mask]
    if len(slice_pts) > 10000:
        idx = np.random.choice(len(slice_pts), 10000, replace=False)
        slice_pts = slice_pts[idx]
    
    ax1.scatter(slice_pts[:, 1], slice_pts[:, 2], s=0.8, c="gray", alpha=0.3, label="Point cloud")
    circle = Circle((cy, cz), R, fill=False, color="red", linewidth=2.5, label=f"Effective Shell (D={2*R:.0f} mm)")
    ax1.add_patch(circle)
    ax1.plot(cy, cz, "r+", markersize=14, markeredgewidth=2.5)
    
    if abs(z_charge_surface - cz) < R:
        x_chord = np.sqrt(max(0.0, R ** 2 - (z_charge_surface - cz) ** 2))
        ax1.plot(
            [cy - x_chord, cy + x_chord],
            [z_charge_surface, z_charge_surface],
            "b-",
            linewidth=2.5,
            label="Charge surface",
        )
    ax1.set_xlabel("Y (mm)", fontsize=11)
    ax1.set_ylabel("Z (mm)", fontsize=11)
    ax1.set_title("Cross-section (YZ Plane)", fontweight="bold", fontsize=12)
    ax1.axis("equal")
    ax1.grid(alpha=0.3)
    ax1.legend(fontsize=9)
    
    # Plot 2: Side View
    ax2 = fig.add_subplot(gs[0, 1])
    step = max(1, len(points) // 12000)
    ds = points[::step]
    ax2.scatter(ds[:, 0], ds[:, 2], s=0.4, c="gray", alpha=0.25, rasterized=True, label="Point cloud")
    ax2.axvline(x_min, color="red", linestyle="--", linewidth=2.5, label="Head planes")
    ax2.axvline(x_max, color="red", linestyle="--", linewidth=2.5)
    
    z_mid = cz
    ax2.annotate(
        "",
        xy=(x_min, z_mid),
        xytext=(x_max, z_mid),
        arrowprops=dict(arrowstyle="<->", lw=2.5, color="red"),
    )
    ax2.text(
        0.5 * (x_min + x_max),
        z_mid - 450.0,
        f"{results['effective_length_mm']:.0f} mm",
        ha="center",
        fontsize=11,
        color="red",
        weight="bold",
        bbox=dict(boxstyle="round", facecolor="white", edgecolor="red", alpha=0.95, linewidth=1.5),
    )
    ax2.set_xlabel("X (Length, mm)", fontsize=11)
    ax2.set_ylabel("Z (Height, mm)", fontsize=11)
    ax2.set_title("Side View (XZ)", fontweight="bold", fontsize=12)
    ax2.grid(alpha=0.3)
    ax2.legend(fontsize=10)
    
    # Plot 3: Results Table
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.axis("off")
    table_data = [
        ["Parameter", "Calculated"],
        ["Effective Diameter (mm)", f"{results['effective_diameter_mm']:.0f}"],
        ["Effective Length (mm)", f"{results['effective_length_mm']:.0f}"],
        ["Free Height Over Load (mm)", f"{results['free_height_mm']:.0f}"],
        ["Total Internal Vol (m³)", f"{results['total_volume_m3']:.1f}"],
        ["Total Vol Occupied by Load (m³)", f"{results['charge_total_m3']:.1f}"],
        ["Internal Vol of Cylinder (m³)", f"{results['cylinder_volume_m3']:.1f}"],
        ["Vol Occupied in Cylinder (m³)", f"{results['charge_cylinder_m3']:.1f}"],
        ["Total Mill Filling (%)", f"{results['total_fill_pct']:.2f}"],
        ["Cylinder Filling (%)", f"{results['cylinder_fill_pct']:.2f}"],
    ]
    table = ax3.table(cellText=table_data, cellLoc="left", loc="center", colWidths=[0.6, 0.4])
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 2.1)
    for col in range(2):
        cell = table[(0, col)]
        cell.set_facecolor("#1565C0")
        cell.set_text_props(weight="bold", color="white", fontsize=10)
    ax3.set_title("Calculation Results", fontweight="bold", fontsize=12, pad=20)
    
    # Plot 4: Fill Schematic
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.set_aspect("equal")
    shell = Circle((0.0, 0.0), R, fill=False, color="black", linewidth=3.5)
    ax4.add_patch(shell)
    
    z_level = z_charge_surface - cz
    h_charge = z_level + R
    h_free = results["free_height_mm"]
    
    if 0.0 < h_charge < 2.0 * R:
        theta = np.linspace(0.0, 2.0 * np.pi, 600)
        x_circ = R * np.cos(theta)
        y_circ = R * np.sin(theta)
        mask = y_circ <= z_level
        x_fill = x_circ[mask]
        y_fill = y_circ[mask]
        if len(x_fill) > 0:
            x_chord = float(np.sqrt(max(0.0, R ** 2 - z_level ** 2)))
            x_fill = np.concatenate(([-x_chord], x_fill, [x_chord]))
            y_fill = np.concatenate(([z_level], y_fill, [z_level]))
            ax4.fill(x_fill, y_fill, color="#FF9800", alpha=0.8, label="Charge", zorder=2)
    
    if abs(z_level) < R:
        x_chord = float(np.sqrt(max(0.0, R ** 2 - z_level ** 2)))
        ax4.plot([-x_chord, x_chord], [z_level, z_level], "b--", linewidth=2.5, label="Charge surface", zorder=3)
    
    ax4.annotate(
        "",
        xy=(R * 1.25, R),
        xytext=(R * 1.25, z_level),
        arrowprops=dict(arrowstyle="<->", lw=2.5, color="#1976D2"),
    )
    ax4.text(
        R * 1.42,
        0.5 * (R + z_level),
        f"Free\n{h_free:.0f} mm",
        fontsize=11,
        color="#1976D2",
        weight="bold",
        ha="left",
        va="center",
    )
    ax4.annotate(
        "",
        xy=(-R * 1.25, -R),
        xytext=(-R * 1.25, z_level),
        arrowprops=dict(arrowstyle="<->", lw=2.5, color="#2E7D32"),
    )
    ax4.text(
        -R * 1.42,
        0.5 * (-R + z_level),
        f"Charge\n{h_charge:.0f} mm",
        fontsize=11,
        color="#2E7D32",
        weight="bold",
        ha="right",
        va="center",
    )
    ax4.axhline(0.0, color="gray", linestyle=":", alpha=0.4, linewidth=1.0)
    ax4.axvline(0.0, color="gray", linestyle=":", alpha=0.4, linewidth=1.0)
    ax4.plot(0.0, 0.0, "k+", markersize=12, markeredgewidth=2.5)
    ax4.set_xlim(-1.65 * R, 1.65 * R)
    ax4.set_ylim(-1.65 * R, 1.65 * R)
    ax4.set_title("Fill Schematic", fontweight="bold", fontsize=12)
    ax4.legend(loc="upper right", fontsize=10)
    
    plt.suptitle(
        f"Ball Mill Fill Level Analysis - ADAPTIVE v2 - {filename}",
        fontsize=14,
        fontweight="bold",
        y=0.98,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(output_path), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved figure: {output_path}")

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def process_file_and_predict():
    print("=" * 70)
    print("AUTOMATED BALL MILL VOLUME PREDICTOR")
    print("=" * 70)

    # 1. Select File
    root = tk.Tk()
    root.withdraw()
    filepath = filedialog.askopenfilename(
        title="Select Pre-Aligned Point Cloud",
        filetypes=[("Point clouds", "*.ply *.xyz *.txt *.csv *.npy"), ("All files", "*.*")],
    )
    root.destroy()

    if not filepath:
        print("No file selected. Exiting.")
        return

    try:
        # 2. Process Point Cloud
        points = load_point_cloud(filepath)
        points = correct_yz_alignment(points)
        
        print("\n[1/2] Analyzing Geometry (Trial 9)...")
        x_min_t9, x_max_t9, effective_length = find_head_planes_trial9(points)
        cy_t9, cz_t9, radius_t9 = fit_shell_circle_trial9(points, x_min_t9, x_max_t9)
        effective_diameter = 2.0 * radius_t9
        
        print(f"      -> Diameter: {effective_diameter:.1f} mm")
        print(f"      -> Length:   {effective_length:.1f} mm")

        print("\n[2/2] Analyzing Free Height (Trial 4)...")
        x_min_t4, x_max_t4 = find_head_planes_trial4(points)
        cy_t4, cz_t4, radius_t4 = fit_shell_circle_trial4(points, x_min_t4, x_max_t4)
        free_height = compute_free_height_trial4(points, x_min_t4, x_max_t4, cy_t4, cz_t4, radius_t4)
        print(f"      -> Free Height: {free_height:.1f} mm")

        # 3. Calculate volumes
        phys_user = get_theoretical_geometry(effective_diameter, effective_length, free_height)
        
        # 5. Generate plots
        input_path = Path(filepath)
        out_dir = Path(r"output\aligned_scans\analysis")
        output_path = out_dir / f"{input_path.stem}_analysis.png"
        
        # 6. Run prediction model FIRST (get corrected values)
        print("\n[3/3] Running Prediction Model...")
        predicted = predict_single_mill(effective_diameter, effective_length, free_height)

        # 7. Build results dictionary USING TABLE VALUES
        results = {
            'cy': cy_t9,
            'cz': cz_t9,
            'radius': radius_t9,
            'xmin': x_min_t9,
            'xmax': x_max_t9,
            'zchargesurface': cz_t4 + radius_t4 - free_height,

            'effective_diameter_mm': effective_diameter,
            'effective_length_mm': effective_length,
            'free_height_mm': free_height,

            # 🔹 USE MODEL-CORRECTED VALUES (FROM TABLE)
            'total_volume_m3': predicted['total_volume_m3'],
            'charge_total_m3': predicted['charge_total_m3'],
            'cylinder_volume_m3': predicted['cylinder_volume_m3'],
            'charge_cylinder_m3': predicted['charge_cylinder_m3'],
            'total_fill_pct': predicted['total_fill_pct'],
            'cylinder_fill_pct': predicted['cylinder_fill_pct']
        }

        make_plots(points, results, output_path, input_path.name)
        
    except Exception as e:
        import traceback
        print(f"\nERROR: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    process_file_and_predict()