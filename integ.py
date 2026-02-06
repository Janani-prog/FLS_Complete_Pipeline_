import tkinter as tk
from tkinter import filedialog
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
    
    if np.sum(valid_mask) > 5:
        valid_indices = np.where(valid_mask)[0]
        med_radius_smooth[valid_mask] = gaussian_filter1d(med_radius[valid_mask], sigma=2.5)

    R_peak = np.nanmax(med_radius_smooth)
    threshold = 0.88 * R_peak
    
    core_mask = valid_mask & (med_radius_smooth >= threshold)
    if not np.any(core_mask):
        x_min = float(np.min(x))
        x_max = float(np.max(x))
    else:
        core_indices = np.where(core_mask)[0]
        i_min = core_indices[0]
        i_max = core_indices[-1]
        x_min = float(x_edges[i_min])
        x_max = float(x_edges[i_max + 1])

    effective_length = float(x_max - x_min)
    return x_min, x_max, effective_length


def fit_shell_circle_trial9(points, x_min, x_max):
    in_cyl = (points[:, 0] >= x_min) & (points[:, 0] <= x_max)
    pts = points[in_cyl]
    y = pts[:, 1]
    z = pts[:, 2]
    r = np.sqrt(y*y + z*z)

    R_coarse = np.median(r)
    shell_min = 0.82 * R_coarse
    shell_max = 1.05 * R_coarse
    shell_mask = (r >= shell_min) & (r <= shell_max)
    shell_pts = pts[shell_mask]

    if len(shell_pts) < 100:
        return 0.0, 0.0, R_coarse

    shell_pts_yz = azimuthally_balanced_shell_points(shell_pts[:, 1:])

    def residuals(params):
        cy, cz, R = params
        return np.sqrt((shell_pts_yz[:, 0] - cy)**2 + (shell_pts_yz[:, 1] - cz)**2) - R

    result = least_squares(residuals, [0.0, 0.0, R_coarse], loss="soft_l1", f_scale=5.0)
    cy, cz, R = result.x
    return float(cy), float(cz), float(R)


# --- TRIAL 4 LOGIC (Free Height - IMPROVED) ---

def find_head_planes_trial4(points):
    """Same as Trial 9 for consistency"""
    return find_head_planes_trial9(points)[:2]


def fit_shell_circle_trial4(points, x_min, x_max):
    """Same as Trial 9 for consistency"""
    return fit_shell_circle_trial9(points, x_min, x_max)


def detect_bottom_improved(points, cy, cz, radius, x_min, x_max):
    """
    IMPROVED bottom detection that works with AND without bottom plates.
    
    Strategy:
    1. Try to detect bottom plate (flat, dense, wide)
    2. If no plate found, use geometric cylinder bottom at -radius
       (This is correct because free height is measured from shell top to charge surface)
    """
    in_cyl = (points[:, 0] >= x_min) & (points[:, 0] <= x_max)
    pts = points[in_cyl]
    y_rel = pts[:, 1] - cy
    z_rel = pts[:, 2] - cz
    r_rel = np.sqrt(y_rel**2 + z_rel**2)

    z_min_data = np.min(z_rel)
    z_max_data = np.max(z_rel)
    print(f"  Z_rel range: {z_min_data:.0f} to {z_max_data:.0f}mm")

    # 1. Try to detect bottom plate (ENHANCED)
    z_bottom_pct = np.percentile(z_rel, 2)  # Stricter: 2nd percentile
    bottom_mask = z_rel < z_bottom_pct
    bottom_pts = np.sum(bottom_mask)
    
    if bottom_pts > 0:
        bottom_y_span = np.ptp(y_rel[bottom_mask])
        bottom_flatness = np.std(z_rel[bottom_mask])
        bottom_z_median = np.median(z_rel[bottom_mask])
        
        # Calculate density (points per square meter in bottom region)
        bottom_height = max(1.0, z_bottom_pct - z_min_data)
        bottom_area = (x_max - x_min) * bottom_y_span * 0.001**2  # Convert to m²
        bottom_density = bottom_pts / bottom_area if bottom_area > 0 else 0

        plate_diam_threshold = 0.85 * 2 * radius  # 85% of diameter
        
        # ENHANCED plate detection criteria
        has_plate = (
            bottom_pts > 500 and  # Sufficient points
            bottom_y_span > plate_diam_threshold and  # Wide enough
            bottom_density > 50 and  # Dense enough (pts/m²)
            bottom_flatness < 20  # Flat enough (std dev < 20mm)
        )

        if has_plate:
            bottom_z_abs = cz + bottom_z_median
            print(f"  ✓ BOTTOM PLATE DETECTED")
            print(f"    Position: Z={bottom_z_abs:.0f}mm")
            print(f"    Points: {bottom_pts}, Span: {bottom_y_span:.0f}mm")
            print(f"    Flatness: {bottom_flatness:.1f}mm, Density: {bottom_density:.1f} pts/m²")
            return bottom_z_abs, True  # Return position and plate_detected flag
        else:
            print(f"  ✗ NO PLATE (pts={bottom_pts}, span={bottom_y_span:.0f}, flat={bottom_flatness:.1f}, dens={bottom_density:.1f})")
    
    # 2. NO PLATE: Use geometric bottom at -radius
    # This is CORRECT because:
    # - Free height is measured from shell TOP to charge SURFACE
    # - The geometric bottom is only used for reporting fill % relative to physical mill height
    # - For volume calculations, we use the cylinder geometry anyway
    bottom_z_abs = cz - radius
    print(f"  → Using GEOMETRIC bottom: Z={bottom_z_abs:.0f}mm (center - radius)")
    return bottom_z_abs, False


def compute_free_height_improved(points, x_min, x_max, cy, cz, radius):
    """
    IMPROVED free height calculation that works with AND without bottom plates.
    
    The key improvement: proper charge surface detection that distinguishes between
    charge at the bottom (low fill) vs charge at the top (high fill).
    """
    # Detect physical bottom (enhanced algorithm)
    shell_bottom_z, has_plate = detect_bottom_improved(points, cy, cz, radius, x_min, x_max)
    shell_top_z = cz + radius

    physical_height = shell_top_z - shell_bottom_z
    print(f"  Physical mill height: {physical_height:.0f}mm (top={shell_top_z:.0f}, bottom={shell_bottom_z:.0f})")
    print(f"  Expected height (2*radius): {2*radius:.0f}mm")

    # Get cylinder points for charge surface detection
    in_cyl = (points[:, 0] >= x_min) & (points[:, 0] <= x_max)
    pts = points[in_cyl]
    y = pts[:, 1] - cy
    z = pts[:, 2] - cz
    r = np.sqrt(y*y + z*z)

    total_cyl_pts = len(pts)
    cyl_density = total_cyl_pts / ((x_max - x_min) * np.pi * radius**2 * 0.001**2)
    print(f"  Cylinder points: {total_cyl_pts:,} | Density: {cyl_density:.1f} pts/m²")

    # Define interior region (exclude shell)
    interior_thresh = 0.72 * radius
    interior = r < interior_thresh
    y_int = y[interior]
    z_int = z[interior]

    interior_pts = len(z_int)
    print(f"  Interior points (r<{interior_thresh:.0f}mm): {interior_pts}")

    # Enhanced fallback logic
    if interior_pts < 150:
        print(f"  → LOW INTERIOR DATA: Using percentile-based estimate")
        # When we have low interior data, use statistical approach
        # This is typical for scans without bottom plates or with sparse sampling
        
        # Get what interior points we do have
        if interior_pts > 30:
            z_p75 = np.percentile(z_int, 75)
            z_p85 = np.percentile(z_int, 85)
            # Use conservative estimate (blend of percentiles)
            surface_z_rel = 0.6 * z_p75 + 0.4 * z_p85
            free_height_mm = shell_top_z - (cz + surface_z_rel)
        else:
            # Very sparse data - use very conservative default
            free_height_mm = radius * 1.25  # 62.5% of diameter
        
        z_charge_surface_mm = shell_top_z - free_height_mm
        print(f"    Free height: {free_height_mm:.0f}mm | Charge surface: {z_charge_surface_mm:.0f}mm")
        return free_height_mm

    # Smooth the Z data for better surface detection
    z_smooth = gaussian_filter1d(z_int, sigma=1.5)
    
    # CRITICAL FIX: Determine if we have a low-fill or high-fill scenario
    # by checking the distribution of interior points
    z_median_int = np.median(z_smooth)
    z_mean_int = np.mean(z_smooth)
    
    # Calculate how many points are in upper vs lower half
    upper_half_pts = np.sum(z_smooth > 0)
    lower_half_pts = np.sum(z_smooth < 0)
    upper_fraction = upper_half_pts / len(z_smooth) if len(z_smooth) > 0 else 0
    
    # Calculate the vertical spread of interior points
    z_std = np.std(z_smooth)
    z_range = np.ptp(z_smooth)
    z_p10 = np.percentile(z_smooth, 10)
    z_p90 = np.percentile(z_smooth, 90)
    
    print(f"  Interior distribution: {lower_half_pts} below center, {upper_half_pts} above center")
    print(f"  Median Z: {z_median_int:.0f}mm, Mean Z: {z_mean_int:.0f}mm")
    print(f"  Z spread: std={z_std:.0f}mm, range={z_range:.0f}mm, p10={z_p10:.0f}, p90={z_p90:.0f}")
    
    # IMPROVED DETECTION: Multiple criteria for low/high fill
    # Low fill indicators:
    # 1. Most points in lower half (upper_fraction < 0.35)
    # 2. Median significantly below center (< -0.2 * radius)
    # 3. 90th percentile below center (concentrated at bottom)
    
    # Additional check: data quality and distribution uniformity
    z_quartiles = np.percentile(z_smooth, [25, 50, 75])
    lower_to_upper_ratio = abs(z_quartiles[0]) / (abs(z_quartiles[2]) + 1e-6)
    
    # If there's much more mass in the lower quartile, it's likely low fill
    is_heavily_weighted_low = lower_to_upper_ratio > 1.5
    
    # Check vertical compactness - low fill tends to have charge settled in a smaller Z range
    z_iqr = z_quartiles[2] - z_quartiles[0]  # Interquartile range
    is_vertically_compact = z_iqr < 0.6 * radius
    
    is_low_fill = ((upper_fraction < 0.35) or 
                   (z_median_int < -0.2 * radius) or 
                   (z_p90 < 0.1 * radius) or
                   (is_heavily_weighted_low and is_vertically_compact))
    
    print(f"  Fill detection metrics:")
    print(f"    Lower/Upper ratio: {lower_to_upper_ratio:.2f}")
    print(f"    IQR: {z_iqr:.0f}mm ({100*z_iqr/radius:.1f}% of radius)")
    print(f"    Vertically compact: {is_vertically_compact}")
    
    if is_low_fill:
        print(f"  → DETECTED: LOW FILL scenario (charge at bottom)")
        # For low fill, find the TOP of the charge pile (highest continuous surface)
        # Scan from bottom to top, find the FIRST (lowest) continuous surface that persists
        
        # Use adaptive binning based on data density
        n_bins = min(250, max(150, interior_pts // 20))
        z_bins = np.linspace(np.min(z_smooth), np.max(z_smooth), n_bins)
        dz = 0.008 * radius  # Slightly tighter slice for better precision
        
        surface_z_rel = None
        candidate_surfaces = []
        
        # Scan from BOTTOM to TOP
        for i, zb in enumerate(z_bins):
            slice_mask = np.abs(z_smooth - zb) < dz
            n_pts_slice = np.sum(slice_mask)
            
            if n_pts_slice < 25:
                continue
            
            y_slice = np.sort(y_int[slice_mask])
            gaps = np.diff(y_slice)
            
            if len(gaps) == 0:
                continue
            
            max_gap = np.max(gaps)
            
            # For low fill, we want a continuous surface
            # Adaptive threshold: larger mills need proportionally larger continuity
            continuity_threshold = 0.20 * radius
            
            if max_gap < continuity_threshold:
                # Check if this is sustained (not just a thin layer)
                # Look at points within +/- 4*dz of this level
                sustained_mask = np.abs(z_smooth - zb) < (4 * dz)
                n_sustained = np.sum(sustained_mask)
                
                # Require more substantial layer for reliable detection
                if n_sustained > max(60, interior_pts * 0.08):
                    # Store candidate with quality score
                    quality_score = n_sustained / (max_gap + 1.0)
                    candidate_surfaces.append((zb, n_pts_slice, max_gap, quality_score))
        
        # Select best candidate (highest quality in lower region)
        if candidate_surfaces:
            # Sort by position (lowest first)
            candidate_surfaces.sort(key=lambda x: x[0])
            
            # Analyze the distribution of candidates
            n_candidates = len(candidate_surfaces)
            print(f"    Found {n_candidates} candidate surfaces")
            
            # For low fill, we want the top of the charge pile
            # This is usually in the MIDDLE of candidates (not too low, not too high)
            # Or the one with best quality if there's a clear winner
            
            # Strategy: Take from middle-to-upper portion of candidates
            start_idx = max(0, n_candidates // 3)  # Skip bottom third
            end_idx = max(start_idx + 1, (2 * n_candidates) // 3)  # Take middle third
            
            selection_pool = candidate_surfaces[start_idx:end_idx]
            
            if not selection_pool:
                selection_pool = candidate_surfaces
            
            # Pick the one with best quality score from this pool
            best = max(selection_pool, key=lambda x: x[3])
            surface_z_rel = best[0]
            print(f"    Selected charge surface at Z_rel={best[0]:.0f}mm (n={best[1]}, gap={best[2]:.0f}mm, quality={best[3]:.2f})")
            print(f"    Selection: candidate {candidate_surfaces.index(best)+1} of {n_candidates}")
        
        if surface_z_rel is None:
            # Fallback: use 80th percentile (better for low fill than 75th)
            surface_z_rel = np.percentile(z_smooth, 80)
            print(f"    FALLBACK: Using 80th percentile Z_rel={surface_z_rel:.0f}mm")
    
    else:
        print(f"  → DETECTED: HIGH FILL scenario (charge fills most of mill)")
        # For high fill, find the HIGHEST continuous surface
        # This is the top of the charge
        
        n_bins = min(250, max(150, interior_pts // 20))
        z_bins = np.linspace(np.min(z_smooth), np.max(z_smooth), n_bins)
        dz = 0.008 * radius
        
        surface_z_rel = None
        best_density = 0
        candidate_surfaces = []
        
        # Scan from BOTTOM to TOP, track all good surfaces
        for i, zb in enumerate(z_bins):
            slice_mask = np.abs(z_smooth - zb) < dz
            n_pts_slice = np.sum(slice_mask)
            
            if n_pts_slice < 25:
                continue
            
            y_slice = np.sort(y_int[slice_mask])
            gaps = np.diff(y_slice)
            
            if len(gaps) == 0:
                continue
            
            max_gap = np.max(gaps)
            
            # Calculate point density
            y_span = np.ptp(y_slice)
            density = n_pts_slice / (y_span * dz) if y_span > 0 else 0
            
            # For high fill: surface should span across mill with good density
            continuity_threshold = 0.18 * radius  # Slightly tighter for high fill
            
            if max_gap < continuity_threshold:
                candidate_surfaces.append((zb, n_pts_slice, max_gap, density))
        
        # For high fill, take the HIGHEST good surface
        if candidate_surfaces:
            # Sort by Z position (highest first)
            candidate_surfaces.sort(key=lambda x: x[0], reverse=True)
            
            # Take the highest one with good quality
            for surf in candidate_surfaces[:5]:  # Check top 5 candidates
                zb, n_pts, gap, dens = surf
                # Verify it's sustained
                sustained_mask = np.abs(z_smooth - zb) < (4 * dz)
                n_sustained = np.sum(sustained_mask)
                
                if n_sustained > max(60, interior_pts * 0.08):
                    surface_z_rel = zb
                    print(f"    Found charge surface at Z_rel={zb:.0f}mm (n={n_pts}, gap={gap:.0f}mm, dens={dens:.2f})")
                    break
        
        if surface_z_rel is None:
            # Fallback for high fill: use 85th percentile
            surface_z_rel = np.percentile(z_smooth, 85)
            print(f"    FALLBACK: Using 85th percentile, Z_rel={surface_z_rel:.0f}mm")

    # Convert relative Z to absolute Z
    z_charge_surface_mm = cz + surface_z_rel
    
    # Calculate free height from TOP of shell to charge surface
    free_height_mm = shell_top_z - z_charge_surface_mm
    
    # ENHANCED SANITY CHECKS with multiple validation strategies
    min_free = 0.03 * (2 * radius)
    max_free = 0.97 * (2 * radius)
    
    # Calculate multiple percentile-based estimates for comparison
    z_p75 = np.percentile(z_smooth, 75)
    z_p80 = np.percentile(z_smooth, 80)
    z_p85 = np.percentile(z_smooth, 85)
    z_p90 = np.percentile(z_smooth, 90)
    z_p95 = np.percentile(z_smooth, 95)
    
    expected_free_from_p75 = shell_top_z - (cz + z_p75)
    expected_free_from_p80 = shell_top_z - (cz + z_p80)
    expected_free_from_p85 = shell_top_z - (cz + z_p85)
    expected_free_from_p90 = shell_top_z - (cz + z_p90)
    
    print(f"  Percentile-based free height estimates:")
    print(f"    P75: {expected_free_from_p75:.0f}mm")
    print(f"    P80: {expected_free_from_p80:.0f}mm")
    print(f"    P85: {expected_free_from_p85:.0f}mm")
    print(f"    P90: {expected_free_from_p90:.0f}mm")
    print(f"    Surface-based: {free_height_mm:.0f}mm")
    
    # STRATEGY 1: Check for large discrepancy with P85
    discrepancy_p85 = abs(free_height_mm - expected_free_from_p85)
    
    # STRATEGY 2: Use weighted average of percentiles for robust estimate
    # Weight towards higher percentiles for better surface detection
    robust_estimate = (0.15 * expected_free_from_p75 + 
                      0.20 * expected_free_from_p80 + 
                      0.30 * expected_free_from_p85 + 
                      0.25 * expected_free_from_p90 +
                      0.10 * free_height_mm)
    
    # STRATEGY 3: Check if surface detection seems unreliable
    # If free height is much larger than most estimates, it's probably wrong
    avg_percentile_estimate = (expected_free_from_p80 + expected_free_from_p85 + expected_free_from_p90) / 3.0
    
    correction_threshold = 0.25 * radius  # About 12.5% of diameter
    
    # Decision logic
    needs_correction = False
    correction_weight = 0.0
    
    if discrepancy_p85 > correction_threshold:
        print(f"  ⚠ Large discrepancy with P85: {discrepancy_p85:.0f}mm")
        needs_correction = True
        
        # Calculate how much to trust percentiles vs surface detection
        # Larger discrepancy = trust percentiles more
        if discrepancy_p85 > 0.4 * radius:
            correction_weight = 0.85  # Trust percentiles heavily
        elif discrepancy_p85 > 0.3 * radius:
            correction_weight = 0.70
        else:
            correction_weight = 0.50
    
    # Additional check: if our estimate is an outlier compared to percentile range
    percentile_spread = expected_free_from_p90 - expected_free_from_p75
    if abs(free_height_mm - avg_percentile_estimate) > percentile_spread:
        print(f"  ⚠ Surface estimate is outlier from percentile range")
        needs_correction = True
        correction_weight = max(correction_weight, 0.75)
    
    if needs_correction:
        print(f"  → Applying correction (weight={correction_weight:.2f})")
        print(f"    Original: {free_height_mm:.0f}mm")
        
        # Use blended estimate
        if is_low_fill:
            # For low fill: strongly favor P80 (captures top of pile better)
            corrected_free = correction_weight * expected_free_from_p80 + (1 - correction_weight) * free_height_mm
        else:
            # For high fill: use robust weighted estimate
            corrected_free = correction_weight * robust_estimate + (1 - correction_weight) * free_height_mm
        
        free_height_mm = corrected_free
        z_charge_surface_mm = shell_top_z - free_height_mm
        print(f"    Corrected: {free_height_mm:.0f}mm")
    
    # Final bounds check
    free_height_mm = np.clip(free_height_mm, min_free, max_free)

    print(f"  → FINAL RESULT:")
    print(f"    Charge surface: Z={z_charge_surface_mm:.0f}mm (rel={surface_z_rel:.0f}mm)")
    print(f"    Free height: {free_height_mm:.0f}mm ({100*free_height_mm/(2*radius):.1f}% of diameter)")
    print(f"    Fill height: {z_charge_surface_mm - shell_bottom_z:.0f}mm ({100*(z_charge_surface_mm - shell_bottom_z)/physical_height:.1f}% of physical height)")
    
    return free_height_mm


# ============================================================================
# PART 2: VOLUME CALCULATIONS  
# ============================================================================

def get_theoretical_geometry(diameter_mm, length_mm, free_height_mm):
    """
    Calculates the EXACT geometric volume of a perfect cylinder segment.
    Returns separate values for:
    - Total mill volume (includes cylinder + head volumes via correction)
    - Cylinder-only volume (straight cylindrical section)
    """
    R = diameter_mm / 2.0 / 1000.0  # Convert to meters
    L = length_mm / 1000.0  # Convert to meters
    H_charge = (diameter_mm - free_height_mm) / 1000.0  # Height of charge in meters
    
    # Cylinder volume (straight cylindrical section only)
    vol_cyl = np.pi * R**2 * L
    
    # Calculate occupied cross-sectional area (circular segment)
    if H_charge <= 0:
        area_occ = 0.0
    elif H_charge >= 2*R:
        area_occ = np.pi * R**2
    else:
        # Circular segment formula
        val = np.clip((R - H_charge) / R, -1.0, 1.0)
        term1 = R**2 * np.arccos(val)
        term2 = (R - H_charge) * np.sqrt(max(0, 2*R*H_charge - H_charge**2))
        area_occ = term1 - term2
    
    # Volume occupied in cylinder
    vol_occ = area_occ * L
    
    # Return as array for compatibility with legacy prediction model
    return np.array([vol_cyl, vol_occ])



# ============================================================================
# PART 3: PREDICTION MODEL (MOCK - Replace with your actual model)
# ============================================================================

def predict_single_mill(user_dia, user_len, user_fh):
    """
    Predict mill volumes using Gaussian Process Regression trained on historical data.
    Returns 6 values: 4 volumes + 2 fill percentages
    """
    # Historical training data (diameter, length, free_height)
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
    
    # Historical ground truth (total_vol, total_occ, cyl_vol, cyl_occ)
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
    
    # Calculate theoretical volumes for training data to get ratios
    ratios_train = []
    for i in range(len(X_train)):
        phys = get_theoretical_geometry(X_train[i,0], X_train[i,1], X_train[i,2])
        # phys[0] = cylinder volume, phys[1] = occupied volume in cylinder
        # Ratios: [total/cyl, total_occ/occ, cyl/cyl, cyl_occ/occ]
        r = [y_train[i,j] / phys[j%2] for j in range(4)]
        ratios_train.append(r)
    ratios_train = np.array(ratios_train)
    
    # Train 4 separate GP models for each ratio
    kernel = C(1.0) * RBF(length_scale=[1000, 1000, 500]) + WhiteKernel(noise_level=1e-5)
    models = []
    for i in range(4):
        gpr = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=5, random_state=42)
        gpr.fit(X_train, ratios_train[:, i])
        models.append(gpr)
    
    # Predict for user input
    X_user = np.array([[user_dia, user_len, user_fh]])
    phys_user = get_theoretical_geometry(user_dia, user_len, user_fh)
    pred_ratios = [m.predict(X_user)[0] for m in models]
    
    # Apply predicted ratios to theoretical volumes
    final_preds = [
        phys_user[0] * pred_ratios[0],  # Total internal volume
        phys_user[1] * pred_ratios[1],  # Total volume occupied
        phys_user[0] * pred_ratios[2],  # Cylinder volume
        phys_user[1] * pred_ratios[3]   # Cylinder occupied
    ]
    
    # Calculate fill percentages
    pct_total = (final_preds[1] / final_preds[0]) * 100
    pct_cyl = (final_preds[3] / final_preds[2]) * 100
    
    final_results = {
        'total_volume_m3': final_preds[0],
        'charge_total_m3': final_preds[1],
        'cylinder_volume_m3': final_preds[2],
        'charge_cylinder_m3': final_preds[3],
        'total_fill_pct': pct_total,
        'cylinder_fill_pct': pct_cyl
    }

    return final_results


# ============================================================================
# PART 4: VISUALIZATION
# ============================================================================

def make_plots(points, results, output_path, filename):
    """Generate analysis plots."""
    cy = results['cy']
    cz = results['cz']
    R = results['radius']
    x_min = results['xmin']
    x_max = results['xmax']
    z_charge_surface = results['zchargesurface']
    
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(2, 2, hspace=0.35, wspace=0.30)
    
    # Plot 1: Cross-section (YZ)
    ax1 = fig.add_subplot(gs[0, 0])
    x_mid = 0.5 * (x_min + x_max)
    slice_mask = np.abs(points[:, 0] - x_mid) < 150.0
    slice_pts = points[slice_mask]
    
    if len(slice_pts) > 8000:
        idx = np.random.choice(len(slice_pts), 8000, replace=False)
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
        f"Ball Mill Fill Level Analysis - IMPROVED v3 - {filename}",
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
    print("IMPROVED BALL MILL VOLUME PREDICTOR - v3")
    print("Works with AND without bottom plates!")
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
        
        print("\n[1/3] Analyzing Geometry (Diameter & Length)...")
        x_min, x_max, effective_length = find_head_planes_trial9(points)
        cy, cz, radius = fit_shell_circle_trial9(points, x_min, x_max)
        effective_diameter = 2.0 * radius
        
        print(f"      -> Diameter: {effective_diameter:.1f} mm")
        print(f"      -> Length:   {effective_length:.1f} mm")

        print("\n[2/3] Analyzing Free Height (IMPROVED Algorithm)...")
        free_height = compute_free_height_improved(points, x_min, x_max, cy, cz, radius)
        print(f"      -> Free Height: {free_height:.1f} mm")

        # 3. Calculate volumes
        print("\n[3/3] Calculating Volumes...")
        phys_user = get_theoretical_geometry(effective_diameter, effective_length, free_height)
        
        # 4. Generate plots
        input_path = Path(filepath)
        out_dir = Path(r"output\aligned_scans\analysis")
        out_dir.mkdir(parents=True, exist_ok=True)
        output_path = out_dir / f"{input_path.stem}_analysis.png"
        
        # 5. Run prediction model (using theoretical for now)
        predicted = predict_single_mill(effective_diameter, effective_length, free_height)

        # 6. Build results dictionary
        results = {
            'cy': cy,
            'cz': cz,
            'radius': radius,
            'xmin': x_min,
            'xmax': x_max,
            'zchargesurface': cz + radius - free_height,

            'effective_diameter_mm': effective_diameter,
            'effective_length_mm': effective_length,
            'free_height_mm': free_height,

            'total_volume_m3': predicted['total_volume_m3'],
            'charge_total_m3': predicted['charge_total_m3'],
            'cylinder_volume_m3': predicted['cylinder_volume_m3'],
            'charge_cylinder_m3': predicted['charge_cylinder_m3'],
            'total_fill_pct': predicted['total_fill_pct'],
            'cylinder_fill_pct': predicted['cylinder_fill_pct']
        }

        make_plots(points, results, output_path, input_path.name)
        
        print("\n" + "=" * 70)
        print("ANALYSIS COMPLETE!")
        print("=" * 70)
        
    except Exception as e:
        import traceback
        print(f"\nERROR: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    process_file_and_predict()
